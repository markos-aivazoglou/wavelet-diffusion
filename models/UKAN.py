from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from models.ukan.Diffusion.Model_UKAN_Hybrid import UKan_Hybrid
import json
import torch
from typing import Optional


class UKANHybrid(UKan_Hybrid, ModelMixin, ConfigMixin):

    config_name = "config.json"

    @register_to_config
    def __init__(
        self,
        in_channels,
        T,
        ch,
        ch_mult,
        attn,
        num_res_blocks,
        dropout,
        sample_size: Optional[int] = None,
    ):
        super().__init__(
            in_channels=in_channels,
            sample_size=sample_size,
            T=T,
            ch=ch,
            ch_mult=ch_mult,
            attn=attn,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
        )
