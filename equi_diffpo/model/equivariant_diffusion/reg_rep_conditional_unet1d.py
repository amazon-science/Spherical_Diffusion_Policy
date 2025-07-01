from typing import Union
import logging

import numpy as np
import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
import pytorch3d.transforms as pytorch3d_transforms

from equi_diffpo.model.diffusion.conditional_unet1d import ConditionalUnet1D
from equi_diffpo.model.equivariant_diffusion.irreps_conv1d_components import (
    Downsample1d, Upsample1d, IrrepConv1dBlock)
from equi_diffpo.model.diffusion.positional_embedding import SinusoidalPosEmb
from equi_diffpo.model.equiformer_v2.equiformerv2_block import FeedForwardNetwork
from equi_diffpo.model.equiformer_v2.se3_transformation import rot_pcd
from equi_diffpo.model.equiformer_v2.module_list import ModuleListInfo
from equi_diffpo.model.equiformer_v2.so3 import (
    CoefficientMappingModule,
    SO3_Embedding,
    SO3_Grid,
    SO3_Rotation,
    SO3_LinearV2
)
import e3nn.o3 as o3

logger = logging.getLogger(__name__)


class RegRepConditionalUnet1D(nn.Module):
    def __init__(self,
                 input_dim,
                 local_cond_dim=None,
                 global_cond_dim=None,
                 diffusion_step_embed_dim=256,
                 act_emb_dim=64,
                 down_dims=[512, 1024, 2048],
                 kernel_size=3,
                 n_groups=8,
                 norm=True,
                 cond_predict_scale=True,
                 max_lmax=2,
                 grid_resolution=14
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.d_irrep = (max_lmax + 1) ** 2
        self.unet = ConditionalUnet1D(
            input_dim=act_emb_dim,
            local_cond_dim=local_cond_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        # Initialize the transformations between spherical and grid representations
        self.SO3_grid = ModuleListInfo('({}, {})'.format(max_lmax, max_lmax))
        for l in range(max_lmax + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(max_lmax + 1):
                SO3_m_grid.append(
                    SO3_Grid(
                        l,
                        m,
                        resolution=grid_resolution,
                        normalization='component'
                    )
                )
            self.SO3_grid.append(SO3_m_grid)

        self.enc = IrrepConv1dBlock(input_dim, act_emb_dim, 1, self.SO3_grid,
                                    norm=False, max_lmax=max_lmax, activation=False)
        self.dec = IrrepConv1dBlock(act_emb_dim, input_dim, 1, self.SO3_grid,
                                    norm=False, max_lmax=max_lmax, activation=False)

        oct_xyz = torch.as_tensor([[1, 0, 0],
                                   [0, 1, 0],
                                   [0, 0, 1],
                                   [-1, 0, 0],
                                   [0, -1, 0],
                                   [0, 0, -1]], dtype=float)
        # spherical_coords = o3.xyz_to_angles(oct_xyz)
        spherical_basis = o3.spherical_harmonics(range(max_lmax + 1), oct_xyz, normalize=True).float()
        self.register_buffer('to_reg', spherical_basis.transpose(0, 1) / spherical_basis.shape[1] ** 0.5)
        self.register_buffer('to_irrep', spherical_basis / spherical_basis.shape[0] ** 0.5)
        self.reg_order = spherical_basis.shape[0]

    def irrep2reg(self, irrep):
        """
        irrep: [b h c i] -> [b h c f]
        """
        return torch.einsum('bhci, if -> bhcf', irrep, self.to_reg)

    def reg2irrep(self, reg):
        """
        reg: [b h c f] -> [b h c i]
        """
        return torch.einsum('bhcf, fi -> bhci', reg, self.to_irrep)

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
        s2_input = torch.zeros(bs, n_hist, self.input_dim, self.d_irrep).to(sample.device)  # [b h c i]
        s2_input[..., 1:4] = sample[:, :, :9].reshape(bs, n_hist, 3, 3)  # 3D position + 6D rotation
        s2_input[..., 0, 0] = sample[:, :, -1]
        s2_input = einops.rearrange(s2_input, 'b n c i -> b (c i) n')
        s2_enc = self.enc(s2_input)
        s2_enc = einops.rearrange(s2_enc, 'b (c i) n -> b n c i', i=self.d_irrep)
        reg_input = self.irrep2reg(s2_enc)
        reg_input = einops.rearrange(reg_input, 'b h c f -> (b f) h c')
        reg_global_cond = self.irrep2reg(global_cond.reshape(bs, 1, -1, self.d_irrep))
        reg_global_cond = einops.rearrange(reg_global_cond, 'b t c f -> (b f t) c')
        if type(timestep) == torch.Tensor and len(timestep.shape) == 1:
            timestep = timestep.repeat_interleave(self.reg_order)

        reg_out = self.unet(reg_input, timestep, local_cond, reg_global_cond)
        reg_out = einops.rearrange(reg_out, "(b f) h c -> b h c f", f=self.reg_order)
        irrep_out = self.reg2irrep(reg_out)

        irrep_out = einops.rearrange(irrep_out, 'b n c i -> b (c i) n')
        x = self.dec(irrep_out)  # (B, (C, irrep), n)
        x = einops.rearrange(x, 'b (c i) n -> b n c i', i=self.d_irrep)

        gripper_out = x[..., :1, 0]  # b h 1
        pos_rot_out = x[..., 1:4]  # b h 3 3
        pos_rot_out = einops.rearrange(pos_rot_out, 'b h n i -> b h (n i)')  # b h 9

        out = torch.cat([pos_rot_out, gripper_out], dim=-1)
        return out

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    device = 'cuda:1'
    atol = 1e-3
    torch.manual_seed(0)
    np.random.seed(0)
    bs = 1
    max_lmax = 1
    condition_dim = 128
    d_irrep = (max_lmax + 1) ** 2
    trajectory = torch.rand(bs, 16, 3, 3) - 0.5
    condition = torch.zeros(bs, condition_dim, d_irrep)
    condition[..., 1:4] = torch.rand(bs, 3) - 0.5
    gripper_open = torch.rand(bs, 16, 2) - 0.5
    condition = condition.to(device)
    trajectory = trajectory.to(device)
    gripper_open = gripper_open.to(device)
    nactions = torch.cat((trajectory.reshape(bs, 16, 9), gripper_open), dim=-1)
    model = RegRepConditionalUnet1D(3,
                                    grid_resolution=6,
                                    global_cond_dim=condition_dim,
                                    diffusion_step_embed_dim=condition_dim,
                                    kernel_size=5,
                                    max_lmax=max_lmax).to(device)
    c4_xyz_rots = torch.zeros((10, 3)).to(device)
    c4_xyz_rots[1:4, 0] = torch.arange(1, 4) * torch.pi / 2
    c4_xyz_rots[4:7, 1] = torch.arange(1, 4) * torch.pi / 2
    c4_xyz_rots[7:10, 2] = torch.arange(1, 4) * torch.pi / 2
    print('total #param: ', model.num_params)

    out = model(sample=nactions,
                timestep=0,
                global_cond=condition.reshape(-1, d_irrep))  # (B, T, (C, irrep))
    vector = out[..., :9].reshape(bs, -1, 9)

    success = True
    for i in range(c4_xyz_rots.shape[0]):
        trajectory_tfm = rot_pcd(trajectory.reshape(bs, -1, 3).clone(), c4_xyz_rots[i]).reshape(bs, 16, 9)
        condition_tfm = torch.zeros_like(condition)
        condition_tfm[..., 1:4] = rot_pcd(condition[..., 1:4], c4_xyz_rots[i])
        vector_tfm_after = rot_pcd(vector.reshape(bs, -1, 3).clone(), c4_xyz_rots[i]).reshape(bs, 16, 9)
        nactions_tfm_after = torch.cat((trajectory_tfm, gripper_open), dim=-1)
        out_tfm_before = model(sample=nactions_tfm_after,
                               timestep=0,
                               global_cond=condition_tfm.reshape(-1, d_irrep))  # (B, T, (C, irrep))
        vector_tfm_before = out_tfm_before[..., :9].reshape(bs, 16, 9)

        # equ error for vector
        eerr = torch.linalg.norm(vector_tfm_before - vector_tfm_after, dim=-1).max()
        err = torch.linalg.norm(vector_tfm_after - vector, dim=-1).max()
        if not torch.allclose(vector_tfm_before, vector_tfm_after, atol=atol):
            print(f"FAILED on {c4_xyz_rots[i]}: {eerr:.1E} > {atol}, {err}")
            success = False
        else:
            print(f"PASSED on {c4_xyz_rots[i]}: {eerr:.1E} < {atol}, {err}")
        # equ error for scalar
        eerr = torch.linalg.norm(out[..., -1:] - out_tfm_before[..., -1:], dim=-1).max()
        err = torch.linalg.norm(out[..., -1:], dim=-1).max()
        if not torch.allclose(vector_tfm_before, vector_tfm_after, atol=atol):
            print(f"FAILED on {c4_xyz_rots[i]}: {eerr:.1E} > {atol}, {err}")
            success = False
        else:
            print(f"PASSED on {c4_xyz_rots[i]}: {eerr:.1E} < {atol}, {err}")


    if success:
        print('PASSED')
    print(1)
