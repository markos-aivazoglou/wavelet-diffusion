import copy
from dataclasses import dataclass
from enum import Enum
from torch import nn
import torch
from torch.nn import functional as F
from tqdm import tqdm
from wavelet.dwt import DiscreteWaveletTransform
from diffusers import DDPMScheduler
from diffusers import UNet2DModel

from diffusers import DDPMScheduler, DDIMScheduler
from wavelet.dwt import DiscreteWaveletTransform
from torchvision.utils import save_image, make_grid


class SamplingMode(str, Enum):
    DDPM = "ddpm"
    DDIM = "ddim"
    PNDM = "pndm"


class WaveletDiffusion(nn.Module):

    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 1000,
        prediction_type: str = "epsilon",
        wavelet_level: int = 1,
        image_resolution: int = 32,
        normalize_wavelet: bool = True,
        with_reconstruction_loss: bool = True,
        sampling_mode: SamplingMode = SamplingMode.DDPM,
        sampling_steps: int = 50,
        sampling_eta: float = 0.0, # only deterministic for DDIM
        channels: int = 3,
    ):
        super().__init__()
        self.model = model
        self.noise_scheduler: DDPMScheduler = DDPMScheduler(
            num_train_timesteps=timesteps,
            prediction_type=prediction_type,
            beta_schedule="squaredcos_cap_v2",
        )
        self.sampling_steps = sampling_steps
        self.sampling_eta = sampling_eta
        self.sampling_mode = sampling_mode
        match sampling_mode:
            case SamplingMode.DDPM:
                self.sampling_scheduler = self.noise_scheduler
            case SamplingMode.DDIM:
                self.sampling_scheduler = DDIMScheduler(
                    num_train_timesteps=timesteps,
                    prediction_type=prediction_type,
                    beta_schedule="squaredcos_cap_v2",
                )
        self.prediction_type = prediction_type
        self.encoder_decoder = DiscreteWaveletTransform(
            level=wavelet_level,
            normalize=normalize_wavelet,
        )
        self.wavelet_levels = wavelet_level
        self.with_reconstruction_loss = with_reconstruction_loss
        self.image_resolution = image_resolution
        self.channels = (lambda x: 4**wavelet_level * x)(channels)
        self.sample_size = (lambda x: x // (2**wavelet_level))(image_resolution)

    @property
    def device(self):
        return next(self.model.parameters()).device

    @torch.inference_mode()
    def sample(
        self,
        batch_size: int = 1,
        generator: torch.Generator = None,
    ) -> torch.Tensor:
        # Sample gaussian noise to begin loop
        image_batch = torch.randn(
            (
                batch_size,
                self.channels,
                self.sample_size,
                self.sample_size,
            ),
            generator=generator,
            device=self.device,
        )
        # set step values
        self.sampling_scheduler.set_timesteps(self.sampling_steps, device=self.device)
        for t in tqdm(
            self.sampling_scheduler.timesteps,
            desc="sampling loop time step",
            total=len(self.sampling_scheduler.timesteps),
            leave=False,
        ):
            # 1. predict noise for current timestep
            if isinstance(self.model, UNet2DModel):
                model_output = self.model(image_batch, t).sample
            else:
                timesteps = (
                    image_batch.new_ones(
                        (image_batch.shape[0],), device=self.device, dtype=torch.int64
                    )
                    * t
                )
                model_output = self.model(image_batch, timesteps)
            # 2. predict previous mean of image x_t-1 and add variance depending on eta
            # eta corresponds to η in paper and should be between [0, 1]
            # do x_t -> x_t-1
            image_batch = self.denoising_step(
                model_output=model_output,
                timestep=t,
                sample=image_batch,
                generator=generator,
            )

            # save_image(
            #     make_grid((image_batch[0].view(-1, 3, self.sample_size, self.sample_size) + 1) / 2),
            #     f"intermediate-steps/all_sample_{t}.png",
            # )

        reconstructed_batch = (
            self.encoder_decoder.decode(image_batch)
            if self.wavelet_levels > 0
            else image_batch
        )
        reconstructed_batch = ((reconstructed_batch + 1) / 2).clamp(0, 1)
        return reconstructed_batch

    def denoising_step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator: torch.Generator = None,
    ) -> torch.Tensor:
        match self.sampling_mode:
            case SamplingMode.DDPM:
                return self.sampling_scheduler.step(
                    model_output=model_output,
                    timestep=timestep,
                    sample=sample,
                    generator=generator,
                ).prev_sample
            case SamplingMode.DDIM:
                return self.sampling_scheduler.step(
                    model_output=model_output,
                    timestep=timestep,
                    sample=sample,
                    eta=self.sampling_eta,
                    generator=generator,
                ).prev_sample
            case SamplingMode.PNDM:
                return self.sampling_scheduler.step(
                    model_output=model_output,
                    timestep=timestep,
                    sample=sample,
                    generator=generator,
                ).prev_sample

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (x0.shape[0],),
            device=x0.device,
            dtype=torch.int64,
        )
        x0_wavelet = self.encoder_decoder.encode(x0) if self.wavelet_levels > 0 else x0
        assert x0_wavelet.shape[1:] == (
            self.channels,
            self.sample_size,
            self.sample_size,
        ), (
            f"Expected transformed images to have shape "
            f"({self.channels}, {self.sample_size}, {self.sample_size}), "
            f"but got {x0_wavelet.shape[1:]}"
        )
        noise = torch.randn(
            x0_wavelet.shape,
            device=x0.device,
        )
        xt_wavelet = self.noise_scheduler.add_noise(x0_wavelet, noise, timesteps)

        prediction = (
            self.model(xt_wavelet, timesteps).sample
            if isinstance(self.model, UNet2DModel)
            else self.model(xt_wavelet, timesteps)
        )
        true_label = noise if self.prediction_type == "epsilon" else x0_wavelet
        loss = F.mse_loss(prediction, true_label)

        if self.with_reconstruction_loss:
            x0_wavelet_predicted = prediction
            if self.prediction_type == "epsilon":
                x0_wavelet_predicted = self.noise_scheduler.step(
                    prediction, 0, xt_wavelet
                ).pred_original_sample
            x0_reconstructed = (
                self.encoder_decoder.decode(x0_wavelet_predicted)
                if self.wavelet_levels > 0
                else x0_wavelet_predicted
            )
            loss += F.smooth_l1_loss(x0_reconstructed, x0)
        return loss

    @torch.inference_mode()
    def validation_loss(self, x0: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(x0)

    @torch.inference_mode()
    def sample_synchronized_bands(
        self,
        batch_size: int = 1,
        generator: torch.Generator = None,
        low_freq_steps: int = None,
        high_freq_steps: int = None,
        freq_split_channel: int = 3,
    ) -> torch.Tensor:
        """
        Alternative approach: Process both frequency bands simultaneously but advance them
        at different rates through their respective complete denoising schedules.
        """

        # Set default step counts
        if low_freq_steps is None:
            low_freq_steps = self.sampling_steps
        if high_freq_steps is None:
            high_freq_steps = max(self.sampling_steps // 4, 10)

        print(
            f"Synchronized denoising: {low_freq_steps} low-freq steps, {high_freq_steps} high-freq steps"
        )

        # Sample gaussian noise
        image_batch = torch.randn(
            (batch_size, self.channels, self.sample_size, self.sample_size),
            generator=generator,
            device=self.device,
        )

        # Create schedulers
        low_freq_scheduler = copy.deepcopy(self.sampling_scheduler)
        high_freq_scheduler = copy.deepcopy(self.sampling_scheduler)

        low_freq_scheduler.set_timesteps(low_freq_steps, device=self.device)
        high_freq_scheduler.set_timesteps(high_freq_steps, device=self.device)

        # Create iterators for each schedule
        low_freq_iter = iter(low_freq_scheduler.timesteps)
        high_freq_iter = iter(high_freq_scheduler.timesteps)

        # Track current timesteps for each band
        try:
            current_low_t = next(low_freq_iter)
        except StopIteration:
            current_low_t = None

        try:
            current_high_t = next(high_freq_iter)
        except StopIteration:
            current_high_t = None

        # Determine total iterations needed
        max_steps = max(low_freq_steps, high_freq_steps)
        low_freq_interval = max_steps / low_freq_steps
        high_freq_interval = max_steps / high_freq_steps

        low_freq_next_step = low_freq_interval
        high_freq_next_step = high_freq_interval

        for step in tqdm(range(max_steps), desc="synchronized denoising"):
            process_low = False
            process_high = False

            # Check if it's time to process low frequency
            if current_low_t is not None and step >= low_freq_next_step - 1:
                process_low = True
                low_freq_next_step += low_freq_interval

            # Check if it's time to process high frequency
            if current_high_t is not None and step >= high_freq_next_step - 1:
                process_high = True
                high_freq_next_step += high_freq_interval

            if not (process_low or process_high):
                continue

            # Process the bands that are due for updates
            if process_low and current_low_t is not None:
                # Get model prediction
                if isinstance(self.model, UNet2DModel):
                    model_output = self.model(image_batch, current_low_t).sample
                else:
                    timesteps = (
                        image_batch.new_ones(
                            (image_batch.shape[0],),
                            device=self.device,
                            dtype=torch.int64,
                        )
                        * current_low_t
                    )
                    model_output = self.model(image_batch, timesteps)

                # Apply denoising to low frequency channels only
                temp_result = self._denoising_step_with_scheduler(
                    model_output=model_output,
                    timestep=current_low_t,
                    sample=image_batch,
                    scheduler=low_freq_scheduler,
                )

                # Update only low frequency channels
                image_batch[:, :freq_split_channel] = temp_result[
                    :, :freq_split_channel
                ]

                # Advance to next timestep
                try:
                    current_low_t = next(low_freq_iter)
                except StopIteration:
                    current_low_t = None

            if process_high and current_high_t is not None:
                # Get model prediction (if not already computed)
                if not process_low:
                    if isinstance(self.model, UNet2DModel):
                        model_output = self.model(image_batch, current_high_t).sample
                    else:
                        timesteps = (
                            image_batch.new_ones(
                                (image_batch.shape[0],),
                                device=self.device,
                                dtype=torch.int64,
                            )
                            * current_high_t
                        )
                        model_output = self.model(image_batch, timesteps)

                # Apply denoising to high frequency channels only
                temp_result = self._denoising_step_with_scheduler(
                    model_output=model_output,
                    timestep=current_high_t,
                    sample=image_batch,
                    scheduler=high_freq_scheduler,
                    generator=generator,
                )

                # Update only high frequency channels
                image_batch[:, freq_split_channel:] = temp_result[
                    :, freq_split_channel:
                ]

                # Advance to next timestep
                try:
                    current_high_t = next(high_freq_iter)
                except StopIteration:
                    current_high_t = None
                save_image(
                    make_grid((image_batch[0].view(4, 3, 128, 128) + 1) / 2),
                    f"intermediate-steps/split_all_sample_{step}.png",
                )

        # Decode and return
        reconstructed_batch = (
            self.encoder_decoder.decode(image_batch)
            if self.wavelet_levels > 0
            else image_batch
        )
        reconstructed_batch = ((reconstructed_batch + 1) / 2).clamp(0, 1)
        return reconstructed_batch

    def _denoising_step_with_scheduler(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        scheduler,
        generator: torch.Generator = None,
    ) -> torch.Tensor:
        """
        Perform a denoising step using a specific scheduler.
        This assumes your denoising_step method can work with different schedulers.
        """
        # Store original scheduler
        original_scheduler = self.sampling_scheduler

        try:
            # Temporarily use the specified scheduler
            self.sampling_scheduler = scheduler

            # Apply the denoising step
            result = self.denoising_step(
                model_output=model_output,
                timestep=timestep,
                sample=sample,
                generator=generator,
            )

        finally:
            # Restore original scheduler
            self.sampling_scheduler = original_scheduler

        return result
