from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Dict, Any
from dataloading import DatasetName
from diffusers import UNet2DModel
from models.ukan.Diffusion.Model_UKAN_Hybrid import UKan_Hybrid
from models.UKAN import UKANHybrid


class ModelType(Enum):
    UNET = "UNET"
    UKAN = "UKAN"


@dataclass
class Unet2DModelConfig:
    block_out_channels: Tuple[int, ...]
    down_block_types: Tuple[str, ...]
    up_block_types: Tuple[str, ...]
    add_attention: bool = True
    dropout: float = 0.0


@dataclass
class UKAN2DModelConfig:
    timesteps: int = 1000
    in_channels: int = 3
    base_channels: int = 128
    channel_multipliers: Tuple[int, ...] = (1, 2, 4, 8)
    attention_at: Tuple[int, ...] = (1, 2)
    num_res_blocks: int = 2
    dropout: float = 0.0


WAVELET_CIFAR10_CONFIGS = {
    1: Unet2DModelConfig(
        block_out_channels=(128, 256, 256),
        down_block_types=(
            "AttnDownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "AttnUpBlock2D",
        ),
        dropout=0.1,
    ),
    2: Unet2DModelConfig(
        block_out_channels=(64, 128, 128),
        down_block_types=(
            "AttnDownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "UpBlock2D",
            "AttnUpBlock2D",
        ),
        dropout=0.1,
    ),
}
WAVELET_CELEBAHQ_CONFIGS = {
    0: Unet2DModelConfig(
        block_out_channels=(128, 256, 256, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    ),
    1: Unet2DModelConfig(
        block_out_channels=(128, 128, 256, 256, 256),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    ),
    2: Unet2DModelConfig(
        block_out_channels=(64, 128, 128, 256),
        down_block_types=(
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
        ),
    ),
}
WAVELET_STL10_CONFIGS = {
    1: Unet2DModelConfig(
        block_out_channels=(128, 256, 256, 256),
        down_block_types=(
            "AttnDownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "UpBlock2D",
            "AttnUpBlock2D",
            "AttnUpBlock2D",
        ),
    ),
    2: Unet2DModelConfig(
        block_out_channels=(128, 256, 256),
        down_block_types=(
            "AttnUpBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            # "DownBlock2D",
        ),
        up_block_types=(
            # "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "AttnUpBlock2D",
        ),
    ),
}
WAVELET_CONFIGS = {
    DatasetName.CIFAR10: WAVELET_CIFAR10_CONFIGS,
    DatasetName.CELEBAHQ: WAVELET_CELEBAHQ_CONFIGS,
    DatasetName.STL10: WAVELET_STL10_CONFIGS,
}

UKAN_CIFAR10_CONFIGS = {
    0: UKAN2DModelConfig(
        timesteps=1000,
        base_channels=128,
        channel_multipliers=(1, 2, 2),
        attention_at=(1,),
        num_res_blocks=2,
        dropout=0.1,
    ),
    1: UKAN2DModelConfig(
        timesteps=1000,
        base_channels=128,
        channel_multipliers=(1, 2),
        attention_at=(0,),
        num_res_blocks=2,
        dropout=0.1,
    ),
    2: UKAN2DModelConfig(
        timesteps=1000,
        base_channels=128,
        channel_multipliers=[1],
        attention_at=(),
        num_res_blocks=2,
        dropout=0.1,
    ),
}
UKAN_CELEBAHQ_CONFIGS = {
    0: UKAN2DModelConfig(
        timesteps=1000,
        base_channels=64,
        channel_multipliers=[1, 2, 4, 4],
        attention_at=[3],
        num_res_blocks=2,
    ),
    1: UKAN2DModelConfig(
        timesteps=1000,
        base_channels=64,
        channel_multipliers=[1, 2, 2, 4, 4],
        attention_at=[3, 4],
        num_res_blocks=2,
    ),
    2: UKAN2DModelConfig(
        timesteps=1000,
        base_channels=128,
        channel_multipliers=[1, 2, 2, 2],
        attention_at=[3],
        num_res_blocks=2,
    ),
}
UKAN_STL10_CONFIGS = {
    0: UKAN2DModelConfig(
        timesteps=1000,
        base_channels=128,
        channel_multipliers=[1, 2, 2, 2],
        attention_at=[3],
        num_res_blocks=2,
    ),
    1: UKAN2DModelConfig(
        timesteps=1000,
        base_channels=128,
        channel_multipliers=[1, 2, 2],
        attention_at=[2],
        num_res_blocks=2,
    ),
}
UKAN_CONFIGS = {
    DatasetName.CIFAR10: UKAN_CIFAR10_CONFIGS,
    DatasetName.CELEBAHQ: UKAN_CELEBAHQ_CONFIGS,
    DatasetName.STL10: UKAN_STL10_CONFIGS,
}


class ModelFactory:
    """Factory for creating various models based on configuration"""

    configs = {
        ModelType.UNET: WAVELET_CONFIGS,
        ModelType.UKAN: UKAN_CONFIGS,
    }
    wavelet_dims_formula = lambda lvl, x: x // 2**lvl  ## WHAT THE FUCK?
    wavelet_channels_formula = lambda lvl, chan: (4**lvl) * chan

    @classmethod
    def determine_wavelet_image_settings(
        cls, res, chans, levels: int
    ) -> Dict[str, int]:
        return {
            "sample_size": cls.wavelet_dims_formula(levels, res),
            "in_channels": cls.wavelet_channels_formula(levels, chans),
        }

    @classmethod
    def get_model_config(
        cls, model_type: ModelType, dataset: DatasetName, wavelet_levels: int = 1
    ) -> Unet2DModelConfig | UKAN2DModelConfig:
        return cls.configs[model_type][dataset][wavelet_levels]

    @classmethod
    def create_model(
        cls,
        model_type: ModelType,
        dataset: DatasetName,
        wavelet_levels: int = 1,
    ) -> Any:
        """Create model based on diffusion type and dataset"""
        return cls.create_unet_model(model_type, dataset, wavelet_levels)

    @classmethod
    def create_unet_model(
        cls, model_type: ModelType, dataset: DatasetName, wavelet_levels: int
    ) -> UNet2DModel:
        """Create UNet2D model for regular diffusion"""
        config = cls.get_model_config(
            model_type, dataset, wavelet_levels=wavelet_levels
        )
        image_settings = cls.determine_wavelet_image_settings(
            dataset.resolution, dataset.channels, wavelet_levels
        )
        if model_type == ModelType.UKAN:
            print(
                f"sample size: {image_settings['sample_size']}, in_channels: {image_settings['in_channels']}"
            )
            return UKANHybrid(
                in_channels=image_settings["in_channels"],
                T=config.timesteps,
                ch=config.base_channels,
                ch_mult=config.channel_multipliers,
                num_res_blocks=config.num_res_blocks,
                attn=config.attention_at,
                dropout=config.dropout,
                sample_size=image_settings["sample_size"],
            )

        return UNet2DModel(
            add_attention=config.add_attention,
            sample_size=image_settings["sample_size"],
            in_channels=image_settings["in_channels"],
            out_channels=image_settings["in_channels"],
            layers_per_block=2,
            block_out_channels=config.block_out_channels,
            down_block_types=config.down_block_types,
            up_block_types=config.up_block_types,
            dropout=config.dropout,
        )


# Backward compatibility
UNet2DModelFactory = ModelFactory
