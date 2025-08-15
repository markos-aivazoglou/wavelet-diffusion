import os
import json
import argparse
from pathlib import Path
from trainer import Trainer, TrainerConfig
from dataloading import load_datasets, DatasetName
from trainer_config import ModelFactory, ModelType
from diffusion.ddpm import WaveletDiffusion
from diffusion.ddpm import SamplingMode


def parse_args():
    parser = argparse.ArgumentParser(description="Train a diffusion model")

    # Training parameters
    train_group = parser.add_argument_group("Training parameters")
    train_group.add_argument(
        "--num-epochs", type=int, default=700, help="Number of training epochs"
    )
    train_group.add_argument(
        "--train-batch-size", type=int, default=256, help="Training batch size"
    )
    train_group.add_argument(
        "--learning-rate", type=float, default=1e-4, help="Learning rate"
    )
    train_group.add_argument(
        "--lr-warmup-steps", type=int, default=1000, help="Learning rate warmup steps"
    )

    # Diffusion parameters
    diff_group = parser.add_argument_group("Diffusion parameters")
    diff_group.add_argument(
        "--num-train-timesteps",
        type=int,
        default=1_000,
        help="Number of training timesteps",
    )
    diff_group.add_argument(
        "--prediction-type",
        type=str,
        choices=["epsilon", "sample"],
        default="epsilon",
    )

    diff_group.add_argument(
        "--sampling-mode",
        type=SamplingMode,
        choices=list(SamplingMode),
        default=SamplingMode.DDPM,
        help="Sampling mode",
    )
    diff_group.add_argument(
        "--num-inference-steps", type=int, default=50, help="Number of inference steps"
    )
    diff_group.add_argument(
        "--sampling-eta", type=float, default=0.0, help="Sampling eta"
    )

    # Evaluation parameters
    eval_group = parser.add_argument_group("Evaluation parameters")
    eval_group.add_argument(
        "--eval-every-epochs", type=int, default=50, help="Evaluate every N epochs"
    )
    eval_group.add_argument(
        "--eval-every-steps", type=int, default=10_000, help="Evaluate every N steps"
    )
    eval_group.add_argument(
        "--eval-batch-size", type=int, default=200, help="Evaluation batch size"
    )
    eval_group.add_argument(
        "--eval-samples", type=int, default=200, help="Number of evaluation samples"
    )

    # Model parameters
    model_group = parser.add_argument_group("Model parameters")
    model_group.add_argument(
        "--model-type",
        type=str,
        choices=["UNET", "UKAN"],
        default="UNET",
        help="Type of diffusion model",
    )
    model_group.add_argument(
        "--dataset",
        type=str,
        choices=["CIFAR10", "CELEBAHQ", "STL10"],
        default="CIFAR10",
        help="Dataset to use",
    )
    model_group.add_argument(
        "--wavelet-level",
        type=int,
        default=1,
        help="Number of wavelet transform levels (depth of decomposition)",
    )
    model_group.add_argument(
        "--normalize-wavelet",
        action="store_false",
        help="Normalize wavelet coefficients",
    )

    # Output parameters
    output_group = parser.add_argument_group("Output parameters")
    output_group.add_argument(
        "--output-dir", type=str, default="wavelet_output", help="Output directory"
    )
    output_group.add_argument(
        "--checkpoint-every-epochs",
        type=int,
        default=50,
        help="Save checkpoint every N epochs",
    )

    return parser.parse_args()

# from torchvision.datasets import FakeData
def main():
    args = parse_args()

    results_folder = Path(f"output/{args.output_dir}")
    results_folder.mkdir(exist_ok=True, parents=True)
    with open(
        os.path.join(results_folder, "training_config.json"),
        "w",
    ) as outfile:
        json.dump(vars(args), outfile, indent=4)
    # Load datasets
    # train_dataset, eval_dataset = FakeData(), FakeData()
    train_dataset, eval_dataset = load_datasets(
        DatasetName[args.dataset], root="./data"
    )
    resolution = train_dataset.resolution
    # resolution = 32
    # import torch

    # train_dataset = torch.utils.data.Subset(train_dataset, range(100))
    config = TrainerConfig(
        output_dir=args.output_dir,
        dataset_name=DatasetName[args.dataset],
        num_train_timesteps=args.num_train_timesteps,
        num_inference_steps=args.num_inference_steps,
        train_batch_size=args.train_batch_size,
        checkpoint_every_epochs=args.checkpoint_every_epochs,
        learning_rate=args.learning_rate,
        lr_warmup_steps=args.lr_warmup_steps,
        eval_every_epochs=args.eval_every_epochs,
        eval_every_steps=args.eval_every_steps,
        eval_batch_size=args.eval_batch_size,
        eval_samples=args.eval_samples,
        num_epochs=args.num_epochs,
        wavelet_level=args.wavelet_level,
        normalize_wavelet=args.normalize_wavelet,
    )
    # Create standard diffusion model
    model = ModelFactory.create_model(
        ModelType[args.model_type],
        DatasetName[args.dataset],
        args.wavelet_level,
    )

    diffusion = WaveletDiffusion(
        model=model,
        timesteps=args.num_train_timesteps,
        prediction_type=args.prediction_type,
        image_resolution=resolution,
        wavelet_level=args.wavelet_level,
        normalize_wavelet=args.normalize_wavelet,
        sampling_mode=SamplingMode(args.sampling_mode),
        with_reconstruction_loss=True,
        sampling_steps=args.num_inference_steps,
        sampling_eta=args.sampling_eta,
    )
    # Create standard trainer
    trainer = Trainer(
        model=model,
        diffusion_model=diffusion,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=config,
    )
    trainer.train()


if __name__ == "__main__":
    main()
