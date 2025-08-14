import os
from pathlib import Path
import torch
from dataclasses import dataclass
from matplotlib import pyplot as plt
from diffusers.models.modeling_utils import ModelMixin
from torch.nn import functional as F
from accelerate import Accelerator
from tqdm import tqdm
from diffusers.schedulers import DDPMScheduler
from torchvision.transforms import v2
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate.utils import DataLoaderConfiguration, ProjectConfiguration
from torcheval.metrics.image import FrechetInceptionDistance as FID

from accelerate.tracking import TensorBoardTracker
from torchvision.utils import make_grid
from diffusion.ddpm import WaveletDiffusion

from dataloading import DatasetName
from torchvision.transforms.v2 import ToPILImage
from wavelet.dwt import DiscreteWaveletTransform
import numpy as np


@dataclass
class TrainerConfig:

    dataset_name: DatasetName = DatasetName.CIFAR10
    num_epochs: int = 10
    checkpoint_every_epochs: int = 50
    eval_every_epochs: int = 50
    eval_every_steps: int = 10_000
    eval_batch_size: int = 16
    num_inference_steps: int = 100
    seed: int = 42
    train_batch_size: int = 32
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 100
    weight_decay: float = 0.0
    save_model: bool = True

    output_dir: str = "output"
    device: str = "cuda"

    num_train_timesteps: int = 1_000
    eval_samples: int = 200
    seed: int = 42

    wavelet_level: int = 1
    normalize_wavelet: bool = True


@dataclass
class State:
    epoch: int = 0
    global_step: int = 0
    best_fid: int = 1e9

    def state_dict(self):
        return self.__dict__

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def set(self, epoch, global_step):
        self.epoch = epoch
        self.global_step = global_step

    def update_best_fid(self, fid):
        self.best_fid = fid


class Trainer:
    """
    A simple trainer for training diffusion models.

    The Trainer initializes the optimizer, learning rate scheduler, accelerator and noise scheduler but they can all be overridden if needed.
    """

    output_dir: str = "output"

    def __init__(
        self,
        model,
        train_dataset: torch.utils.data.Dataset,
        eval_dataset: torch.utils.data.Dataset,
        config: TrainerConfig,
        diffusion_model,
    ):
        self.config = config
        self.results_folder = Path(f"{self.output_dir}/{config.output_dir}")
        self.results_folder.mkdir(exist_ok=True, parents=True)
        self.accelerator = self._init_accelerator()
        self.world_size = self.accelerator.state.num_processes
        self.diffusion_model: WaveletDiffusion = diffusion_model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.state = State()
        self.optimizer = self._init_optimizer(model, config)
        self.train_batch_size = config.train_batch_size
        self.eval_batch_size = config.eval_batch_size
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=config.lr_warmup_steps,
            num_training_steps=(len(train_dataset) * config.num_epochs),
        )
        self.current_fid = float("inf")
        encoder_decoder = DiscreteWaveletTransform(
            level=config.wavelet_level,
            normalize=config.normalize_wavelet,
        )
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=config.num_train_timesteps,
        )

        self.noise_scheduler.set_timesteps(config.num_inference_steps)
        (
            self.model,
            self.diffusion_model,
            self.optimizer,
            self.lr_scheduler,
            self.train_dataloader,
            self.eval_dataloader,
            self.fid_eval_dataloader,
            self.encoder_decoder,
            self.state,
        ) = self.accelerator.prepare(
            model,
            self.diffusion_model,
            self.optimizer,
            self.lr_scheduler,
            torch.utils.data.DataLoader(
                train_dataset,
                batch_size=config.train_batch_size,
                shuffle=True,
                generator=torch.Generator().manual_seed(config.seed),
            ),
            torch.utils.data.DataLoader(
                eval_dataset,
                batch_size=config.eval_batch_size,
                shuffle=False,
                generator=torch.Generator().manual_seed(config.seed),
            ),
            torch.utils.data.DataLoader(
                torch.utils.data.Subset(
                    eval_dataset,
                    range(config.eval_samples),
                ),
                batch_size=config.eval_batch_size,
                shuffle=False,
                generator=torch.Generator().manual_seed(config.seed),
            ),
            encoder_decoder,
            self.state,
        )
        self.accelerator.register_for_checkpointing(
            self.model,
            self.optimizer,
            self.lr_scheduler,
            self.state,
        )
        self.diffusion_model = diffusion_model.to(self.accelerator.device)
        self.fid_evaluator = FID(device=self.accelerator.device)

        resume_from_checkpoint = self._get_most_recent_checkpoint(self.results_folder)
        if resume_from_checkpoint is not None:
            self.accelerator.print(f"Resumed from checkpoint: {resume_from_checkpoint}")
            self.accelerator.load_state(resume_from_checkpoint)
        # diffusion_model = self.accelerator.unwrap_model(
        #                             self.diffusion_model
        #                         )
        # diffusion_model.model.save_pretrained(
        #     os.path.join(
        #         self.results_folder,
        #         f"model/ckpt_{1310}",
        #     )
        # )
        # exit(0)

    def _get_most_recent_checkpoint(self, output_dir: str) -> str:
        # print(f"Looking for checkpoints in {output_dir}")
        # return os.path.join(output_dir, "checkpoints", "checkpoint_1310")
        path = os.path.join(output_dir, "checkpoints")
        if not os.path.exists(path):
            return None
        dirs = [os.path.join(path, f.name) for f in os.scandir(path) if f.is_dir()]
        dirs.sort(key=os.path.getctime)
        path = dirs[
            -1
        ]  # Sorts folders by date modified, most recent checkpoint is the last
        return path

    def _init_accelerator(
        self,
    ) -> Accelerator:
        # Initialize accelerator and tensorboard logging
        accelerator = Accelerator(
            mixed_precision="no",
            gradient_accumulation_steps=1,
            log_with="tensorboard",
            # project_dir=os.path.join(self.results_folder),
            project_config=ProjectConfiguration(
                project_dir=self.results_folder,
                logging_dir=os.path.join(self.results_folder, "logs"),
            ),
            dataloader_config=DataLoaderConfiguration(split_batches=True),
        )

        if accelerator.is_main_process:
            if self.results_folder is not None:
                os.makedirs(self.results_folder, exist_ok=True)
            accelerator.init_trackers(self.results_folder)
        return accelerator

    def _init_optimizer(self, model: ModelMixin, config: TrainerConfig):
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate * self.world_size**0.5,
            weight_decay=1e-2,
        )
        return optimizer

    def save_checkpoint(self, state: State):
        """Save a checkpoint of the training state."""
        if not self.accelerator.is_main_process:
            return

        checkpoint_dir = os.path.join(
            self.results_folder, "checkpoints", f"checkpoint_{state.epoch}"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.accelerator.save_state(checkpoint_dir)
        checkpoints = os.listdir(checkpoint_dir)
        previous_checkpoints = list(
            filter(lambda x: x != f"checkpoint_{state.epoch}", checkpoints)
        )
        for checkpoint in previous_checkpoints:
            if os.path.exists(checkpoint):
                self.accelerator.print(f"Deleting checkpoint: {checkpoint}")
                os.removedirs(checkpoint)

    def load_checkpoint(self, checkpoint_path: str):
        self.accelerator.load_state(checkpoint_path)

    def _save_images(self, images, step):
        image_grid = make_grid(images[:16], nrow=4)
        test_dir = os.path.join(self.results_folder, "samples")
        os.makedirs(test_dir, exist_ok=True)
        # save the image grid
        image_grid = image_grid.permute(1, 2, 0).cpu().numpy()
        plt.imsave(os.path.join(test_dir, f"sample_{step}.png"), image_grid)

    def _calculate_fid(
        self,
        real_images: torch.Tensor,
        generated_images: torch.Tensor,
    ):
        self.fid_evaluator.update(real_images, is_real=True)
        self.fid_evaluator.update(generated_images, is_real=False)
        score = self.fid_evaluator.compute()
        return score.item()


    def _image_to_tensor(self, images):
        preprocess = v2.Compose(
            [
                v2.RGB(),
                v2.PILToTensor(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )
        return torch.stack([preprocess(image) for image in images]).to(
            self.accelerator.device
        )

    def _plot_predicted_and_original_noisy(
        self, predicted_sub, original_sub, original, timesteps
    ):
        reconstructed_predicted = self.encoder_decoder.decode(
            predicted_sub.unsqueeze(0)
        )

        reconstructed_predicted = ((reconstructed_predicted + 1) / 2.0).clamp(0, 1)
        reconstructed_original = self.encoder_decoder.decode(original_sub.unsqueeze(0))
        reconstructed_original = ((reconstructed_original + 1) / 2.0).clamp(0, 1)

        original = (original + 1) / 2
        plt.figure(figsize=(20, 15))
        plt.title(f"Step {timesteps.item()}")
        plt.subplot(2, 3, 1)
        pred_subbands = predicted_sub.view(16, 3, 8, 8)
        grid = make_grid(pred_subbands, nrow=4, normalize=True)
        plt.imshow(ToPILImage()(grid))
        plt.title("Predicted")
        original_sub = original_sub.cpu()
        plt.subplot(2, 3, 2)
        original_subbands = original_sub.view(16, 3, 8, 8)
        grid = make_grid(original_subbands, nrow=4, normalize=True)
        plt.imshow(ToPILImage()(grid))
        plt.title("Original")

        plt.subplot(2, 3, 4)
        plt.imshow(ToPILImage()(reconstructed_predicted[0].cpu()))
        plt.title("Reconstructed")

        plt.subplot(2, 3, 5)
        plt.imshow(ToPILImage()(reconstructed_original[0].cpu()))
        plt.title("Reconstructed Original")

        plt.subplot(2, 3, 6)
        plt.imshow(ToPILImage()(original.cpu()))
        plt.title("Original Image")
        diff = (original - reconstructed_original[0]).abs().mean()

        plt.subplot(2, 3, 3)
        plt.imshow(ToPILImage()((reconstructed_original[0] - original).abs().cpu()))
        plt.title(f"Diff: {diff}")
        plt.show()

    def train(self):
        global_step = self.state.global_step + 1
        current_epoch = self.state.epoch + 1
        total_steps = len(self.train_dataloader) * self.config.num_epochs
        self.current_fid = self.state.best_fid if self.state.best_fid < float("inf") else float("inf")
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        timings = np.zeros((11, 1))
        with tqdm(
            initial=global_step,
            total=total_steps,
            disable=not self.accelerator.is_local_main_process,
            desc="Steps",
        ) as pbar:
            while global_step <= total_steps:
                with tqdm(
                    initial=1,
                    total=len(self.train_dataloader),
                    unit="batch",
                    desc="Epoch Progress",
                    disable=not self.accelerator.is_local_main_process,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                    leave=False,
                ) as pbar_epoch:
                    starter.record()
                    # total_loss = 0.0
                    for step, x0 in enumerate(self.train_dataloader):
                        with self.accelerator.accumulate(self.diffusion_model):
                            loss = self.diffusion_model(x0)
                            self.accelerator.backward(loss)
                            # total_loss += loss.detach().item()
                            if self.accelerator.sync_gradients:
                                self.accelerator.clip_grad_norm_(
                                    self.diffusion_model.parameters(), 1.0
                                )
                            self.optimizer.step()
                            self.lr_scheduler.step()
                            self.optimizer.zero_grad()
                        logs = {
                            "loss": loss.detach().item(),
                            "lr": self.lr_scheduler.get_last_lr()[0],
                            "epoch": current_epoch,
                        }
                        pbar.update(1)
                        pbar.set_postfix(
                            {
                                "loss": loss.detach().item(),
                                "step": global_step,
                                "lr": self.lr_scheduler.get_last_lr()[0],
                                "epoch": current_epoch,
                                "FID": self.current_fid,
                            }
                        )
                        self.accelerator.log(logs, step=global_step)
                        global_step += 1
                        pbar_epoch.update(1)
                    ender.record()
                    torch.cuda.synchronize()
                    elapsed_time = starter.elapsed_time(ender)
                    if current_epoch <= 3:
                        timings[current_epoch] = elapsed_time
                    if current_epoch == 3 and self.accelerator.is_main_process:
                        print("mean time: ", np.mean(timings[1:10]))
                        print("std time: ", np.std(timings[1:10]))
                    current_epoch += 1

                if self.accelerator.is_main_process:
                    if (
                        current_epoch % self.config.checkpoint_every_epochs == 0
                        or current_epoch == self.config.num_epochs
                    ):
                        self.state.set(current_epoch, global_step)
                        self.save_checkpoint(self.state)
                    if (
                        current_epoch % self.config.eval_every_epochs == 0
                        or current_epoch == self.config.num_epochs
                        or global_step % self.config.eval_every_steps == 0
                    ):
                        score = self.evaluate(step=global_step)
                        self.current_fid = score
                        if score < self.state.best_fid:
                            self.state.update_best_fid(score)
                        if self.config.save_model:
                            diffusion_model = self.accelerator.unwrap_model(
                                self.diffusion_model
                            )
                            diffusion_model.model.save_pretrained(
                                os.path.join(
                                    self.results_folder,
                                    f"model/ckpt_{current_epoch}",
                                )
                            )
                        pbar.set_postfix({"FID": score})
                        pbar_epoch.set_postfix({"FID": score})
        self.accelerator.end_training()

    def evaluate(self, step: int) -> float:
        images = self.diffusion_model.sample(
            self.config.eval_samples,
            generator=torch.Generator(device=self.accelerator.device).manual_seed(
                self.config.seed
            ),
        )
        val_loss_total = 0
        for batch in self.eval_dataloader:
            val_loss = self.diffusion_model.validation_loss(
                batch,
            )
            val_loss_total += val_loss.item()
        val_loss = val_loss_total / len(self.eval_dataloader)
        self.accelerator.print(f"\nValidation loss: {val_loss: .2f}")
        self.accelerator.log(
            {
                "val_loss": val_loss,
            },
            step=step,
        )

        eval_samples = torch.cat([batch for batch in self.fid_eval_dataloader], dim=0)
        score = self._calculate_fid(eval_samples, images)
        self.accelerator.print(f"FID: {score}")
        self.accelerator.log(
            {
                "FID": score,
            },
            step=step,
        )

        tracker: TensorBoardTracker = self.accelerator.trackers[0]
        tracker.log_images({"generated_images": images[:16]}, step=step)
        self._save_images(images, step)
        return score
