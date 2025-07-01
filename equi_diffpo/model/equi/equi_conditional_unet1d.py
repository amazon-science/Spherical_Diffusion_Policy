from typing import Union
import torch
from escnn import gspaces, nn
from escnn.group import CyclicGroup
from einops import rearrange, repeat
import numpy as np

import sys
sys.path.append('/home/dian/projects/diffusion_policy')

from equi_diffpo.model.diffusion.conditional_unet1d import ConditionalUnet1D

class EquiDiffusionUNet(torch.nn.Module):
    def __init__(self, act_emb_dim, local_cond_dim, global_cond_dim, diffusion_step_embed_dim, down_dims, kernel_size, n_groups, cond_predict_scale, N):
        super().__init__()
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
        self.N = N
        self.group = gspaces.no_base_space(CyclicGroup(self.N))
        self.order = self.N
        self.act_type = nn.FieldType(self.group, act_emb_dim * [self.group.regular_repr])
        self.out_layer = nn.Linear(self.act_type, 
                                   self.getOutFieldType())
        self.enc_a = nn.SequentialModule(
            nn.Linear(self.getOutFieldType(), self.act_type), 
            nn.ReLU(self.act_type)
        )

    def getOutFieldType(self):
        return nn.FieldType(
            self.group,
            4 * [self.group.irrep(1)] # 8
            + 2 * [self.group.trivial_repr], # 2
        )

    def getOutput(self, conv_out):
        xy = conv_out[:, 0:2]
        cos1 = conv_out[:, 2:3]
        sin1 = conv_out[:, 3:4]
        cos2 = conv_out[:, 4:5]
        sin2 = conv_out[:, 5:6]
        cos3 = conv_out[:, 6:7]
        sin3 = conv_out[:, 7:8]
        z = conv_out[:, 8:9]
        g = conv_out[:, 9:10]

        action = torch.cat((xy, z, cos1, cos2, cos3, sin1, sin2, sin3, g), dim=1)
        return action
    
    def getActionGeometricTensor(self, act):
        batch_size = act.shape[0]
        xy = act[:, 0:2]
        z = act[:, 2:3]
        rot = act[:, 3:9]
        g = act[:, 9:]

        cat = torch.cat(
            (
                xy.reshape(batch_size, 2),
                rot[:, 0].reshape(batch_size, 1),
                rot[:, 3].reshape(batch_size, 1),
                rot[:, 1].reshape(batch_size, 1),
                rot[:, 4].reshape(batch_size, 1),
                rot[:, 2].reshape(batch_size, 1),
                rot[:, 5].reshape(batch_size, 1),
                z.reshape(batch_size, 1),
                g.reshape(batch_size, 1),
            ),
            dim=1,
        )
        return nn.GeometricTensor(cat, self.getOutFieldType())
    
    def forward(self, 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int], 
            local_cond=None, global_cond=None, **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        local_cond: (B,T,local_cond_dim)
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        B, T = sample.shape[:2]
        sample = rearrange(sample, "b t d -> (b t) d")
        sample = self.getActionGeometricTensor(sample)
        enc_a_out = self.enc_a(sample).tensor.reshape(B, T, -1)
        enc_a_out = rearrange(enc_a_out, "b t (c f) -> (b f) t c", f=self.order)
        if type(timestep) == torch.Tensor and len(timestep.shape) == 1:
            timestep = repeat(timestep, "b -> (b f)", f=self.order)
        if local_cond is not None:
            local_cond = rearrange(local_cond, "b t (c f) -> (b f) t c", f=self.order)
        if global_cond is not None:
            global_cond = rearrange(global_cond, "b (c f) -> (b f) c", f=self.order)
        out = self.unet(enc_a_out, timestep, local_cond, global_cond, **kwargs)
        out = rearrange(out, "(b f) t c -> (b t) (c f)", f=self.order)
        out = nn.GeometricTensor(out, self.act_type)
        out = self.out_layer(out).tensor.reshape(B * T, -1)
        out = self.getOutput(out)
        out = rearrange(out, "(b t) n -> b t n", b=B)
        return out
 
class EquiDiffusionUNetSE2(torch.nn.Module):
    def __init__(self, act_emb_dim, local_cond_dim, global_cond_dim, diffusion_step_embed_dim, down_dims, kernel_size, n_groups, cond_predict_scale, N):

        super().__init__()
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
        self.N = N
        self.group = gspaces.no_base_space(CyclicGroup(self.N))
        self.order = self.N
        self.act_type = nn.FieldType(self.group, act_emb_dim * [self.group.regular_repr])
        self.out_layer = nn.Linear(self.act_type, 
                                   self.getOutFieldType())
        self.enc_a = nn.SequentialModule(
            nn.Linear(self.getOutFieldType(), self.act_type), 
            nn.ReLU(self.act_type)
        )

    def getOutFieldType(self):
        return nn.FieldType(
            self.group,
            2 * [self.group.irrep(1)] # 4
            + 2 * [self.group.trivial_repr], # 2
        )

    def getOutput(self, conv_out):
        xy = conv_out[:, 0:2]
        cos1 = conv_out[:, 2:3]
        sin1 = conv_out[:, 3:4]
        z = conv_out[:, 4:5]
        g = conv_out[:, 5:6]

        action = torch.cat((xy, z, cos1, sin1, g), dim=1)
        return action
    
    def getActionGeometricTensor(self, act):
        batch_size = act.shape[0]
        xy = act[:, 0:2]
        z = act[:, 2:3]
        cos = act[:, 3:4]
        sin = act[:, 4:5]
        g = act[:, 5:]

        cat = torch.cat(
            (
                xy.reshape(batch_size, 2),
                cos.reshape(batch_size, 1),
                sin.reshape(batch_size, 1),
                z.reshape(batch_size, 1),
                g.reshape(batch_size, 1),
            ),
            dim=1,
        )
        return nn.GeometricTensor(cat, self.getOutFieldType())
    
    def forward(self, 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int], 
            local_cond=None, global_cond=None, **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        local_cond: (B,T,local_cond_dim)
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        B, T = sample.shape[:2]
        sample = rearrange(sample, "b t d -> (b t) d")
        sample = self.getActionGeometricTensor(sample)
        enc_a_out = self.enc_a(sample).tensor.reshape(B, T, -1)
        enc_a_out = rearrange(enc_a_out, "b t (c f) -> (b f) t c", f=self.order)
        if type(timestep) == torch.Tensor and len(timestep.shape) == 1:
            timestep = repeat(timestep, "b -> (b f)", f=self.order)
        if local_cond is not None:
            local_cond = rearrange(local_cond, "b t (c f) -> (b f) t c", f=self.order)
        if global_cond is not None:
            global_cond = rearrange(global_cond, "b (c f) -> (b f) c", f=self.order)
        out = self.unet(enc_a_out, timestep, local_cond, global_cond, **kwargs)
        out = rearrange(out, "(b f) t c -> (b t) (c f)", f=self.order)
        out = nn.GeometricTensor(out, self.act_type)
        out = self.out_layer(out).tensor.reshape(B * T, -1)
        out = self.getOutput(out)
        out = rearrange(out, "(b t) n -> b t n", b=B)
        return out
    
class EquiDiffusionUNetFrameAverage(torch.nn.Module):
    def __init__(self, act_emb_dim, local_cond_dim, global_cond_dim, diffusion_step_embed_dim, down_dims, kernel_size, n_groups, cond_predict_scale, N):
        super().__init__()
        self.N = N
        self.group = gspaces.no_base_space(CyclicGroup(self.N))
        self.order = self.N
        self.unet = ConditionalUnet1D(
            input_dim=act_emb_dim,
            local_cond_dim=local_cond_dim,
            global_cond_dim=global_cond_dim * N,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.trans = torch.zeros([self.N, 2, 2])
        self.trans_inv = torch.zeros([self.N, 2, 2])
        for i in range(self.N):
            self.trans[i] = torch.tensor([[np.cos(2 * i * np.pi / self.N), -np.sin(2 * i * np.pi / self.N)], 
                                          [np.sin(2 * i * np.pi / self.N), np.cos(2 * i * np.pi / self.N)]])
            self.trans_inv[i] = torch.tensor([[np.cos(-2 * i * np.pi / self.N), -np.sin(-2 * i * np.pi / self.N)], 
                                              [np.sin(-2 * i * np.pi / self.N), np.cos(-2 * i * np.pi / self.N)]])

    def getOutFieldType(self):
        return nn.FieldType(
            self.group,
            4 * [self.group.irrep(1)] # 8
            + 2 * [self.group.trivial_repr], # 2
        )

    def getOutput(self, conv_out):
        xy = conv_out[:, 0:2]
        cos1 = conv_out[:, 2:3]
        sin1 = conv_out[:, 3:4]
        cos2 = conv_out[:, 4:5]
        sin2 = conv_out[:, 5:6]
        cos3 = conv_out[:, 6:7]
        sin3 = conv_out[:, 7:8]
        z = conv_out[:, 8:9]
        g = conv_out[:, 9:10]

        action = torch.cat((xy, z, cos1, cos2, cos3, sin1, sin2, sin3, g), dim=1)
        return action
    
    def getActionGeometricTensor(self, act):
        batch_size = act.shape[0]
        xy = act[:, 0:2]
        z = act[:, 2:3]
        rot = act[:, 3:9]
        g = act[:, 9:]

        cat = torch.cat(
            (
                xy.reshape(batch_size, 2),
                rot[:, 0].reshape(batch_size, 1),
                rot[:, 3].reshape(batch_size, 1),
                rot[:, 1].reshape(batch_size, 1),
                rot[:, 4].reshape(batch_size, 1),
                rot[:, 2].reshape(batch_size, 1),
                rot[:, 5].reshape(batch_size, 1),
                z.reshape(batch_size, 1),
                g.reshape(batch_size, 1),
            ),
            dim=1,
        )
        return nn.GeometricTensor(cat, self.getOutFieldType())
    
    def forward(self, 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int], 
            local_cond=None, global_cond=None, **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        local_cond: (B,T,local_cond_dim)
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        B, T = sample.shape[:2]
        # sample = rearrange(sample, "b t d -> (b t) d")
        expanded = repeat(sample, "b t d -> (b t) f d", f=self.order)
        trans_sample = expanded.clone()
        trans_inv = self.trans_inv.to(expanded.device).unsqueeze(0)
        trans_sample[:, :, 0:2] = (trans_inv @ expanded[:, :, 0:2].unsqueeze(-1)).squeeze(-1)
        trans_sample[:, :, [3, 6]] = (trans_inv @ expanded[:, :, [3, 6]].unsqueeze(-1)).squeeze(-1)
        trans_sample[:, :, [4, 7]] = (trans_inv @ expanded[:, :, [4, 7]].unsqueeze(-1)).squeeze(-1)
        trans_sample[:, :, [5, 8]] = (trans_inv @ expanded[:, :, [5, 8]].unsqueeze(-1)).squeeze(-1)
        trans_sample = rearrange(trans_sample, "(b t) f d -> (b f) t d", t=T)

        # sample = self.getActionGeometricTensor(sample)
        # enc_a_out = self.enc_a(sample).tensor.reshape(B, T, -1)

        # enc_a_out = rearrange(enc_a_out, "b t (c f) -> b t c f", f=self.order)
        # expanded = enc_a_out.unsqueeze(1).expand(-1, self.order, -1, -1, -1)
        # indices = (torch.arange(self.order)[:, None] - torch.arange(self.order)) % self.order
        # indices = indices.to(expanded.device)
        # indices = indices.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # Shape: [1, 1, F, 1, F]
        # indices = indices.expand(B, -1, T, expanded.shape[-2], -1)
        # gathered = torch.gather(expanded, 4, indices)
        # enc_a_out = rearrange(gathered, "b f1 t c f2 -> (b f1) t (c f2)")

        # enc_a_out = rearrange(enc_a_out, "b t (c f) -> (b f) t c", f=self.order)
        if type(timestep) == torch.Tensor and len(timestep.shape) == 1:
            timestep = repeat(timestep, "b -> (b f)", f=self.order)
        if local_cond is not None:
            local_cond = rearrange(local_cond, "b t (c f) -> (b f) t c", f=self.order)
        if global_cond is not None:
            # global_cond = rearrange(global_cond, "b (c f) -> (b f) c", f=self.order)
            global_cond = rearrange(global_cond, "b (c f) -> b c f", f=self.order)
            expanded = global_cond.unsqueeze(1).expand(-1, self.order, -1, -1)
            indices = (torch.arange(self.order)[:, None] - torch.arange(self.order)) % self.order
            indices = indices.to(expanded.device)
            indices = indices.unsqueeze(0).unsqueeze(2)  # Shape: [1, 1, F, 1, F]
            indices = indices.expand(B, -1, expanded.shape[-2], -1)
            gathered = torch.gather(expanded, 3, indices)
            global_cond = rearrange(gathered, "b f1 c f2 -> (b f1) (c f2)")
        out = self.unet(trans_sample, timestep, local_cond, global_cond, **kwargs)

        out = rearrange(out, "(b f) t d -> (b t) f d", f=self.order)
        trans = self.trans.to(out.device).unsqueeze(0)
        out[:, :, 0:2] = (trans @ out[:, :, 0:2].unsqueeze(-1)).squeeze(-1)
        out[:, :, [3, 6]] = (trans @ out[:, :, [3, 6]].unsqueeze(-1)).squeeze(-1)
        out[:, :, [4, 7]] = (trans @ out[:, :, [4, 7]].unsqueeze(-1)).squeeze(-1)
        out[:, :, [5, 8]] = (trans @ out[:, :, [5, 8]].unsqueeze(-1)).squeeze(-1)
        out = rearrange(out, "(b t) f d -> b f t d", t=T)
        return out.mean(1)

        # out = rearrange(out, "(b f1) t (c f2) -> b f1 t c f2", f1=self.order, f2=self.order) 
        # indices = (torch.arange(self.order)[None, :] - torch.arange(self.order)[:, None]) % self.order
        # indices = indices.to(out.device)
        # indices = indices.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        # indices = indices.expand(B, -1, T, out.shape[-2], -1)  
        # gathered = torch.gather(out, 4, indices)
        # out = gathered.mean(1)
        # out = rearrange(out, "b t c f -> (b t) (c f)")

        # # out = rearrange(out, "(b f) t c -> (b t) (c f)", f=self.order)
        # out = nn.GeometricTensor(out, self.act_type)
        # out = self.out_layer(out).tensor.reshape(B * T, -1)
        # out = self.getOutput(out)
        # out = rearrange(out, "(b t) n -> b t n", b=B)
        # return out
    
if __name__ == "__main__":
    act_dim = 10
    global_dim = 8
    N = 4
    group = gspaces.no_base_space(CyclicGroup(N))
    model = EquiDiffusionUNetFrameAverage(act_dim, None, global_dim, 8, [128, 256, 512], 5, 4, True, N)
    sample = torch.tensor([1, 0, 0, 
                           1, 0, 0, 
                           0, 1, 0, 
                           0]).float()
    sample = repeat(sample, "d -> 1 t d", t=16)
    global_cond = torch.rand(1, global_dim*N).float()
    timestep = torch.tensor([0]).float()
    out = model(sample=sample, timestep=timestep, global_cond=global_cond)
    out_geo = model.getActionGeometricTensor(out[0])

    g_sample = torch.tensor([0, 1, 0, 
                             0, -1, 0, 
                             1, 0, 0, 
                             0]).float()
    g_sample = repeat(g_sample, "d -> 1 t d", t=16)
    g_global_cond = nn.GeometricTensor(global_cond, nn.FieldType(group, global_dim * [group.regular_repr])).transform(list(group.testing_elements)[1]).tensor
    g_out = model(sample=g_sample, timestep=timestep, global_cond=g_global_cond)
    g_out_geo = model.getActionGeometricTensor(g_out[0])
    a = out_geo.transform(list(group.testing_elements)[1])
    if torch.allclose(out_geo.transform(list(group.testing_elements)[1]).tensor, g_out_geo.tensor):
        print("passed")
    else:
        print("failed")
    print(0)