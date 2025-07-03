from typing import Union
import logging

import numpy as np
import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
import pytorch3d.transforms as pytorch3d_transforms

from sdp.model.diffusion.conditional_unet1d import ConditionalUnet1D
from sdp.model.equivariant_diffusion.irreps_conv1d_components import (
    Downsample1d, Upsample1d, IrrepConv1dBlock)
from sdp.model.diffusion.positional_embedding import SinusoidalPosEmb
from sdp.model.equiformer_v2.equiformerv2_block import FeedForwardNetwork
from sdp.model.equiformer_v2.se3_transformation import rot_pcd
from sdp.model.equiformer_v2.module_list import ModuleListInfo
from sdp.model.equiformer_v2.so3 import (
    CoefficientMappingModule,
    SO3_Embedding,
    SO3_Grid,
    SO3_Rotation,
    SO3_LinearV2
)
import e3nn.o3 as o3

logger = logging.getLogger(__name__)


class NonEquIrrepConditionalUnet1D(nn.Module):
    def __init__(self,
                 input_dim,
                 local_cond_dim=None,
                 global_cond_dim=None,
                 diffusion_step_embed_dim=256,
                 down_dims=[512, 1024, 2048],
                 kernel_size=3,
                 n_groups=8,
                 cond_predict_scale=True,
                 max_lmax=2,
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.d_irrep = (max_lmax + 1) ** 2
        self.unet = ConditionalUnet1D(
            input_dim=input_dim,
            global_cond_dim=global_cond_dim * self.d_irrep,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

    def forward(self,
                sample: torch.Tensor,
                timestep: Union[torch.Tensor, float, int],
                local_cond=None, global_cond=None):
        """
        ToDo: implement in-out put of vector_sample and scalar_sample
        sample: (B, T, C)
        timestep: (B,) or int, diffusion step
        local_cond: (B, T, (C, irrep))
        global_cond: (B, (C, irrep))
        output: (B, T, (C, irrep))
        """
        bs, n_hist, c = sample.shape
        out = self.unet(sample, timestep, local_cond, global_cond.reshape(bs, -1))
        return out

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
