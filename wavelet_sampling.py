#!/usr/bin/env python
import argparse
from accelerate import Accelerator, PartialState
import torch
from torchvision.utils import save_image
import os
from diffusers import UNet2DModel
from diffusers import DDPMPipeline
from diffusers.schedulers import (
    DDPMScheduler,
    DDIMScheduler,
    PNDMScheduler,
)
from wavelet.dwt import DiscreteWaveletTransform
from diffusion.ddpm import WaveletDiffusion, SamplingMode


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate samples from a diffusion model"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory or huggingface repository containing the pretrained model",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save generated images",
    )

    parser.add_argument(
        "--model-type",
        type=str,
        choices=["UNET", "UKAN"],
        default="UNET",
        help="Type of model to use",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=32,
        help="Resolution of the images to generate",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="ddpm",
        choices=["ddpm", "ddim"],
        help="Type of noise scheduler to use",
    )
    parser.add_argument(
        "--prediction-type",
        type=str,
        default="epsilon",
        choices=["epsilon", "sample"],
        help="Type of prediction to use",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="Eta value for DDIM scheduler (ignored for other schedulers)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=5, help="Number of samples to generate"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size for generation"
    )
    parser.add_argument(
        "--sampling-steps", type=int, default=50, help="Number of inference steps"
    )
    parser.add_argument(
        "--wavelet-level",
        type=int,
        default=1,
        help="Number of wavelet decomposition levels",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def get_model(model_type, model_dir):
    match model_type:
        case "UNET":
            return UNet2DModel.from_pretrained(model_dir)
        case "UKAN":
            from models.UKAN import UKANHybrid

            return UKANHybrid.from_pretrained(model_dir)
        case _:
            raise ValueError(f"Unknown model type: {model_type}")

import numpy as np
def main():
    args = parse_args()
    args.seed = 42
    os.makedirs(args.output_dir, exist_ok=True)
    model = get_model(args.model_type, args.model_dir).to(args.device)
    print(f"Resolution: {args.resolution}, Channels: {model.config.in_channels}")
    
    diffusion = WaveletDiffusion(
        model=model,
        prediction_type=args.prediction_type,
        wavelet_level=args.wavelet_level,
        image_resolution=args.resolution,
        sampling_mode=args.scheduler,
        sampling_steps=args.sampling_steps,
        sampling_eta=args.eta,
    )
    diffusion = diffusion.to(args.device)
    # Generate images
    batches = list(range(args.num_samples // args.batch_size))
    generator = (
        torch.Generator(args.device).manual_seed(args.seed)
        if args.seed is not None
        else None
    )
    for batch in batches:
        # Generate images
        images = diffusion.sample(
            batch_size=args.batch_size,
            generator=generator,
        )

        for j, image in enumerate(images):
            image_id = batch * args.batch_size + j
            image_filename = os.path.join(args.output_dir, f"{image_id:05}.png")
            save_image(image, image_filename)
        
    print(f"Generated {args.num_samples} images in {args.output_dir}")


if __name__ == "__main__":
    main()
