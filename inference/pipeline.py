from diffusers import DiffusionPipeline, ImagePipelineOutput, UNet2DModel, ModelMixin
from diffusers.schedulers import DDPMScheduler, DDIMScheduler, SchedulerMixin
import torch
from typing import Optional, Union, Tuple
from wavelet.dwt import DiscreteWaveletTransform

class WaveletPipeline(DiffusionPipeline):
    r"""
    WaveletPipeline is a custom pipeline that extends DiffusionPipeline for wavelet-based image processing.
    It inherits from DiffusionPipeline and can be used to create, load, and run wavelet-based diffusion models.
    """

    def __init__(self, backbone: ModelMixin, scheduler: SchedulerMixin, level: int = 1):
        super().__init__()
        dwt = DiscreteWaveletTransform(level=level)
        self.register_modules(backbone=backbone, scheduler=scheduler, decoder=dwt)
        
    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[torch.Generator] = None,
        num_inference_steps: int = 50,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        # Sample gaussian noise to begin loop
        image_batch = torch.randn(
            (
                batch_size,
                self.backbone.in_channels,
                self.backbone.sample_size,
                self.backbone.sample_size,
            ),
            generator=generator,
            device=self.device,
        )
        # set step values
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        for t in self.progress_bar(self.scheduler.timesteps):
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

        reconstructed_batch = (
            self.decoder.decode(image_batch)
            if self.wavelet_levels > 0
            else image_batch
        )
        reconstructed_batch = ((reconstructed_batch + 1) / 2).clamp(0, 1)
        images = reconstructed_batch.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            images = self.numpy_to_pil(images)

        return ImagePipelineOutput(images=images) if return_dict else (images,)

    def denoising_step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator: torch.Generator = None,
    ) -> torch.Tensor:
        match self.scheduler:
            case DDPMScheduler():
                return self.sampling_scheduler.step(
                    model_output=model_output,
                    timestep=timestep,
                    sample=sample,
                    generator=generator,
                ).prev_sample
            case DDIMScheduler():
                return self.sampling_scheduler.step(
                    model_output=model_output,
                    timestep=timestep,
                    sample=sample,
                    eta=self.sampling_eta,
                    generator=generator,
                ).prev_sample