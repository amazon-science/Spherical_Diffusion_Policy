from typing import Union
import logging

import numpy as np
import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
import pytorch3d.transforms as pytorch3d_transforms
from sdp.model.equivariant_diffusion.irreps_conv1d_components import (
    Downsample1d, Upsample1d, IrrepConv1dBlock, IrrepLinear)
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

logger = logging.getLogger(__name__)


class IrrepConditionalResidualBlock1D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 cond_dim,
                 kernel_size=3,
                 n_groups=8,
                 cond_predict_scale=True,
                 norm=True,
                 max_lmax=3,
                 SO3_grid=None,
                 FiLM_type='SFiLM', ):
        super().__init__()
        self.max_lmax = max_lmax
        self.blocks = nn.ModuleList([
            IrrepConv1dBlock(in_channels, out_channels, kernel_size, SO3_grid, norm=norm, max_lmax=max_lmax),
            IrrepConv1dBlock(out_channels, out_channels, kernel_size, SO3_grid, norm=norm, max_lmax=max_lmax),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cond_dim = cond_dim
        assert FiLM_type in ['FiLM', 'SFiLM', 'SSFiLM']
        self.FiLM_type = FiLM_type
        if self.FiLM_type == 'SFiLM':
            self.cond_encoder = IrrepConv1dBlock(cond_dim, 2 * out_channels, 1, SO3_grid,
                                                 norm=False, max_lmax=max_lmax, activation=False)
        elif self.FiLM_type == 'SSFiLM':
            self.cond_encoder = IrrepLinear(cond_dim, 2 * out_channels, max_lmax)
        elif self.FiLM_type == 'FiLM':
            self.dim_irrep = (max_lmax + 1) ** 2
            self.square_irrep = max_lmax + 1
            self.cond_encoder = torch.nn.Linear(cond_dim * self.square_irrep, 2 * out_channels * self.square_irrep)

        # make sure dimensions compatible
        self.residual_conv = IrrepConv1dBlock(in_channels, out_channels, 1, SO3_grid, norm=False,
                                              max_lmax=max_lmax, activation=False) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : (B, (Cin1, irrep), n_pts)
            cond : (B, (Cin2, irrep), 1)

            returns:
            out : (B, (Cout, irrep), n_pts)
        '''
        bs, npts = x.shape[0], x.shape[-1]
        out = self.blocks[0](x).reshape(bs, self.out_channels, -1, npts)
        if self.FiLM_type == 'SFiLM':
            embed = self.cond_encoder(cond).reshape(bs, 2 * self.out_channels, -1, 1)  # (B, 2*Cout, irrep, n_pts)
            # SFiLM
            # FiLM conditioning for each irrep type l
            l_start, l_end = 0, 0
            outs = []
            for l in range(self.max_lmax + 1):
                l_order = 2 * l + 1
                l_end += l_order
                l_scale, l_bias = embed[:, ::2, l_start:l_end, :], embed[:, 1::2, l_start:l_end, :]
                l_out = out[:, :, l_start:l_end, :]
                length = (l_out * l_scale).sum(dim=2, keepdim=True)
                direction = l_out / (torch.norm(l_out, p=2, dim=2, keepdim=True) + 1e-12)  # add epsilon for stability
                l_out = length * direction + l_bias
                outs.append(l_out)
                l_start = l_end
            out = torch.cat(outs, dim=2)
        elif self.FiLM_type == 'SSFiLM':
            embed = self.cond_encoder(cond).reshape(bs, 2 * self.out_channels, -1, 1)  # (B, 2*Cout, irrep, n_pts)
            l_scale, l_bias = embed[:, ::2, 0:1, :], embed[:, 1::2, :, :]
            out = l_scale * out + l_bias
        else:
            cond = einops.rearrange(cond, 'b (c i) n -> (b i) (c n)', i=self.square_irrep)
            embed = self.cond_encoder(cond)
            embed = embed.reshape(bs, -1, self.dim_irrep, 1)  # (B, 2*Cout, irrep, n_pts)
            # Conventional FiLM
            out = out * embed[:, ::2, ...] + embed[:, 1::2, ...]

        out = out.reshape(bs, -1, npts)
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class IrrepConditionalUnet1D(nn.Module):
    def __init__(self,
                 input_dim,
                 local_cond_dim=None,
                 global_cond_dim=None,
                 diffusion_step_embed_dim=128,
                 # down_dims=[128, 256, 512],
                 down_dims=[200, 400, 800],
                 kernel_size=3,
                 n_groups=8,
                 norm=True,
                 FiLM_type='SFiLM',
                 cond_predict_scale=True,
                 max_lmax=2,
                 grid_resolution=14
                 ):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]
        self.input_dim = input_dim
        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed
        if global_cond_dim is not None:
            cond_dim += global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

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

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            IrrepConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim, norm=norm,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale, FiLM_type=FiLM_type,
                max_lmax=max_lmax, SO3_grid=self.SO3_grid
            ),
            IrrepConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim, norm=norm,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale, FiLM_type=FiLM_type,
                max_lmax=max_lmax, SO3_grid=self.SO3_grid
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            scale = 1 if is_last else 0.5
            down_modules.append(nn.ModuleList([
                IrrepConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, norm=norm,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale, FiLM_type=FiLM_type,
                    max_lmax=max_lmax, SO3_grid=self.SO3_grid),
                IrrepConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim, norm=norm,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale, FiLM_type=FiLM_type,
                    max_lmax=max_lmax, SO3_grid=self.SO3_grid),
                IrrepConv1dBlock(dim_out, dim_out, 3, self.SO3_grid,
                                 norm=False, max_lmax=max_lmax, scale=scale)
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            scale = 1 if is_last else 2
            up_modules.append(nn.ModuleList([
                IrrepConditionalResidualBlock1D(
                    dim_out * 2, dim_in, cond_dim=cond_dim, norm=norm,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale, FiLM_type=FiLM_type,
                    max_lmax=max_lmax, SO3_grid=self.SO3_grid),
                IrrepConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim, norm=norm,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale, FiLM_type=FiLM_type,
                    max_lmax=max_lmax, SO3_grid=self.SO3_grid),
                IrrepConv1dBlock(dim_in, dim_in, 3, self.SO3_grid,
                                 norm=False, max_lmax=max_lmax, scale=scale)
            ]))

        final_conv = IrrepConv1dBlock(start_dim, input_dim, 3, self.SO3_grid,
                                      norm=False, max_lmax=max_lmax, activation=False)

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv
        self.d_irrep = (max_lmax + 1) ** 2
        self.diffusion_step_embed_dim = diffusion_step_embed_dim


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
        s2_input = torch.zeros(bs, n_hist, self.input_dim, self.d_irrep).to(sample.device)
        s2_input[..., 1:4] = sample[:, :, :9].reshape(bs, n_hist, 3, 3)  # 3D position + 6D rotation
        s2_input[..., 0, 0] = sample[:, :, -1]
        s2_input = einops.rearrange(s2_input, 'b h c i -> b (c i) h')

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=s2_input.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(s2_input.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(s2_input.shape[0])

        global_feature = torch.zeros(bs, self.diffusion_step_embed_dim, self.d_irrep).to(s2_input.device)
        global_feature[:, :, 0] = self.diffusion_step_encoder(timesteps)  # (B, dsed, irrep)

        if global_cond is not None:
            global_feature = torch.cat((global_feature, global_cond.reshape(bs, -1, self.d_irrep)), dim=1)
            global_feature = global_feature.reshape(bs, -1, 1)
            # ToDo: check the feature

        # encode local features
        h_local = list()
        assert local_cond is None, "local cond is not implemented"

        x = s2_input
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            if idx == 0 and len(h_local) > 0:
                x = x + h_local[1]
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)  # (B, (C, irrep), n)

        x = einops.rearrange(x, 'b (n i) h -> b h n i', i=self.d_irrep)
        gripper_out = x[..., :1, 0]  # b h 1
        pos_rot_out = x[..., 1:4]  # b h 3 3
        pos_rot_out = einops.rearrange(pos_rot_out, 'b h n i -> b h (n i)')  # b h 9

        out = torch.cat([pos_rot_out, gripper_out], dim=-1)
        return out

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    device = 'cuda:0'
    atol = 1e-4
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
    model = IrrepConditionalUnet1D(3,
                                   grid_resolution=8,
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
