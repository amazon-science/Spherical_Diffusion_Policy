import sys
sys.path.append('/home/dian/projects/diffusion_policy')
from typing import Union, Dict, Tuple
import torch
import torch.nn.functional as F
from escnn import gspaces, nn
from escnn.group import DihedralGroup, CyclicGroup
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from sdp.model.diffusion.positional_embedding import SinusoidalPosEmb

class EquiLinear(torch.nn.Module):
    def __init__(self, in_type: nn.FieldType, out_type: nn.FieldType, bias=True):
        super().__init__()
        self.in_type = in_type
        self.out_type = out_type
        self.layer = nn.Linear(in_type, out_type, bias=bias)

    def forward(self, x):
        return self.layer(self.in_type(x)).tensor

    
class EquiGroupNorm(torch.nn.Module):
    def __init__(self, in_type: nn.FieldType, n_groups):
        super().__init__()
        self.in_type = in_type
        self.field_size = in_type.size
        self.n_repr = len(in_type.representations)
        self.repr_size = self.field_size // self.n_repr
        self.group_norm = torch.nn.GroupNorm(n_groups, self.n_repr)

    def forward(self, x):
        x = rearrange(x, 'b (c f) d -> (b f) c d', f=self.repr_size)
        x = self.group_norm(x)
        x = rearrange(x, '(b f) c d -> b (c f) d', f=self.repr_size)
        return x

# class EquiConv1d(torch.nn.Module):
#     def __init__(self, in_type: nn.FieldType, out_type: nn.FieldType, kernel_size, stride=1):
#         super().__init__()
#         self.in_type = in_type
#         self.out_type = out_type
#         self.n_repr = len(in_type)
#         self.repr_size = in_type.size // self.n_repr

#         # self.layer = nn.Linear(self.layer_in_type, out_type)
#         self.conv = torch.nn.Conv1d(len(in_type), len(out_type), kernel_size, padding=kernel_size // 2, stride=stride)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x: b c d, 
#         x = rearrange(x, 'b (d f) c -> (b f) d c', f=self.repr_size)
#         out = self.conv(x)
#         out = rearrange(out, '(b f) d c -> b (d f) c', f=self.repr_size)
#         return out

        # B = x.shape[0]
        # padding = self.kernel_size // 2
        # x_padded = F.pad(x, (padding, padding), 'constant', 0)
        # windows = x_padded.unfold(dimension=2, size=self.kernel_size, step=self.stride)
        # # windows shape: [b, c, d, kernel_size]
        # windows = rearrange(windows, 'b c d k -> (b d) (k c)')
        # y = self.layer(self.layer_in_type(windows))
        # # (b d) cout
        # return rearrange(y.tensor, '(b d) c -> b c d', b=B, c=self.out_type.size)

class EquiConv1d(torch.nn.Module):
    def __init__(self, in_type: nn.FieldType, out_type: nn.FieldType, kernel_size, stride=1):
        super().__init__()
        self.in_type = in_type
        self.out_type = out_type
        self.kernel_size = kernel_size
        self.stride = stride
        self.layer_in_type = in_type
        for _ in range(kernel_size - 1):
            self.layer_in_type += in_type
        self.layer = nn.Linear(self.layer_in_type, out_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: b c d
        if not self.training:
            _matrix = self.layer.matrix
            _bias = self.layer.expanded_bias
        else:
            # retrieve the matrix and the bias
            _matrix, _bias = self.layer.expand_parameters()
        weights = _matrix.reshape(self.out_type.size, self.kernel_size, -1)
        weights = rearrange(weights, 'o k i -> o i k')
        out = F.conv1d(x, weights, _bias, stride=self.stride, padding=self.kernel_size//2)
        return out

class EquiConv1dBlock(torch.nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''
    def __init__(self, in_type: nn.FieldType, out_type: nn.FieldType, kernel_size, n_groups=8):
        super().__init__()

        self.block = torch.nn.Sequential(
            EquiConv1d(in_type, out_type, kernel_size),
            # Rearrange('batch channels horizon -> batch channels 1 horizon'),
            EquiGroupNorm(out_type, n_groups),
            # Rearrange('batch channels 1 horizon -> batch channels horizon'),
            torch.nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)
        
class EquiConditionalResidualBlock1D(torch.nn.Module):
    def __init__(
            self, 
            in_type: nn.FieldType, 
            out_type: nn.FieldType, 
            cond_in_type: nn.FieldType, 
            kernel_size, 
            n_groups=8, 
            cond_predict_scale=False):
        super().__init__()
        self.group = in_type.gspace
        self.out_type = out_type
        self.out_repr_size = out_type.representations[0].size

        self.blocks = torch.nn.ModuleList([
            EquiConv1dBlock(in_type, out_type, kernel_size, n_groups),
            EquiConv1dBlock(out_type, out_type, kernel_size, n_groups),
        ])

        cond_channels = len(out_type.representations)
        if cond_predict_scale:
            cond_channels = cond_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.cond_encoder = torch.nn.Sequential(
            torch.nn.Mish(),
            EquiLinear(cond_in_type, nn.FieldType(self.group, cond_channels * [self.group.regular_repr])),
            Rearrange('batch t -> batch t 1'),
        )

        self.residual_conv = EquiConv1d(in_type, out_type, 1) \
            if in_type != out_type else torch.nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(
                embed.shape[0], 2, -1, 1)
            scale = embed[:,0,...]
            bias = embed[:,1,...]
            # scale = repeat(scale, 'b c 1 -> b (c f) 1', f=self.out_repr_size)
            # bias = repeat(bias, 'b c 1 -> b (c f) 1', f=self.out_repr_size)
            out = scale * out + bias
        else:
            # embed = repeat(embed, 'b c 1 -> b (c f) 1', f=self.out_repr_size)
            out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out

        # FILM
        # x = self.conv1(x)
        # x = self.relu(x)
        # res = x
        # x = self.conv2(x)
        # x = self.group_norm(x)

        # embed = self.cond_encoder(cond)
        # if self.cond_predict_scale:
        #     embed = embed.reshape(
        #         embed.shape[0], 2, -1, 1)
        #     scale = embed[:,0,...]
        #     bias = embed[:,1,...]
        #     # scale = repeat(scale, 'b c 1 -> b (c f) 1', f=self.out_repr_size)
        #     # bias = repeat(bias, 'b c 1 -> b (c f) 1', f=self.out_repr_size)
        #     x = scale * x + bias
        # else:
        #     # embed = repeat(embed, 'b c 1 -> b (c f) 1', f=self.out_repr_size)
        #     x = x + embed

        # x = self.relu(x)
        # x = x + res
        # return x

        # Temporal 1D conv
        # res = x
        # x = self.conv1(x)

        # embed = self.cond_encoder(cond)
        # if self.cond_predict_scale:
        #     embed = embed.reshape(
        #         embed.shape[0], 2, -1, 1)
        #     scale = embed[:,0,...]
        #     bias = embed[:,1,...]
        #     # scale = repeat(scale, 'b c 1 -> b (c f) 1', f=self.out_repr_size)
        #     # bias = repeat(bias, 'b c 1 -> b (c f) 1', f=self.out_repr_size)
        #     x = scale * x + bias
        # else:
        #     # embed = repeat(embed, 'b c 1 -> b (c f) 1', f=self.out_repr_size)
        #     x = x + embed

        # x = self.group_norm(x)
        # x = self.relu(x)
        # x = self.conv2(x)
        # x = self.group_norm(x)
        # x = self.relu(x)
        # x = x + self.residual_conv(res)
        # return x
    
class Interpolate(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return F.interpolate(x, size=x.shape[-1] * 2, mode='linear', align_corners=False)

class EquiConditionalUnet1D(torch.nn.Module):
    def __init__(self, 
        act_emb_dim,
        local_cond_dim=None,
        global_cond_dim=None,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False,
        N=8
        ):
        super().__init__()
        self.N = N
        self.group = gspaces.no_base_space(CyclicGroup(self.N))
        self.act_emb_type = nn.FieldType(self.group, act_emb_dim * [self.group.regular_repr])

        self.action_type = self.getOutFieldType()

        self.enc_a = nn.SequentialModule(
            nn.Linear(self.action_type, self.act_emb_type), 
            nn.ReLU(self.act_emb_type)
        )

        all_dims = [act_emb_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = torch.nn.Sequential(
            SinusoidalPosEmb(dsed),
            torch.nn.Linear(dsed, dsed * 4),
            torch.nn.Mish(),
            torch.nn.Linear(dsed * 4, dsed),
        )
        # cond_dim = dsed
        # if global_cond_dim is not None:
        #     cond_dim += global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        local_cond_encoder = None
        if local_cond_dim is not None:
            raise NotImplementedError

        mid_dim = all_dims[-1]
        self.mid_modules = torch.nn.ModuleList([
            EquiConditionalResidualBlock1D(
                nn.FieldType(self.group, mid_dim * [self.group.regular_repr]), 
                nn.FieldType(self.group, mid_dim * [self.group.regular_repr]),
                nn.FieldType(self.group, dsed * [self.group.trivial_repr] + global_cond_dim * [self.group.regular_repr]),
                kernel_size=kernel_size, 
                n_groups=n_groups,
                cond_predict_scale=cond_predict_scale),
            EquiConditionalResidualBlock1D(
                nn.FieldType(self.group, mid_dim * [self.group.regular_repr]), 
                nn.FieldType(self.group, mid_dim * [self.group.regular_repr]),
                nn.FieldType(self.group, dsed * [self.group.trivial_repr] + global_cond_dim * [self.group.regular_repr]),
                kernel_size=kernel_size, 
                n_groups=n_groups,
                cond_predict_scale=cond_predict_scale),
        ])

        down_modules = torch.nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(torch.nn.ModuleList([
                EquiConditionalResidualBlock1D(
                    nn.FieldType(self.group, dim_in * [self.group.regular_repr]), 
                    nn.FieldType(self.group, dim_out * [self.group.regular_repr]),
                    nn.FieldType(self.group, dsed * [self.group.trivial_repr] + global_cond_dim * [self.group.regular_repr]),
                    kernel_size=kernel_size, 
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                EquiConditionalResidualBlock1D(
                    nn.FieldType(self.group, dim_out * [self.group.regular_repr]), 
                    nn.FieldType(self.group, dim_out * [self.group.regular_repr]),
                    nn.FieldType(self.group, dsed * [self.group.trivial_repr] + global_cond_dim * [self.group.regular_repr]),
                    kernel_size=kernel_size, 
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                torch.nn.MaxPool1d(2) if not is_last else torch.nn.Identity()
            ]))

        up_modules = torch.nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(torch.nn.ModuleList([
                EquiConditionalResidualBlock1D(
                    nn.FieldType(self.group, dim_out*2 * [self.group.regular_repr]), 
                    nn.FieldType(self.group, dim_in * [self.group.regular_repr]),
                    nn.FieldType(self.group, dsed * [self.group.trivial_repr] + global_cond_dim * [self.group.regular_repr]),
                    kernel_size=kernel_size, 
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                EquiConditionalResidualBlock1D(
                    nn.FieldType(self.group, dim_in * [self.group.regular_repr]), 
                    nn.FieldType(self.group, dim_in * [self.group.regular_repr]),
                    nn.FieldType(self.group, dsed * [self.group.trivial_repr] + global_cond_dim * [self.group.regular_repr]),
                    kernel_size=kernel_size, 
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Interpolate() if not is_last else torch.nn.Identity()
            ]))
        
        # self.final_conv = EquiConv1dBlock(
        #         nn.FieldType(self.group, start_dim * [self.group.regular_repr]), 
        #         nn.FieldType(self.group, start_dim * [self.group.regular_repr]), 
        #         kernel_size=kernel_size, n_groups=n_groups)
        # self.final_conv = torch.nn.Sequential(
        #     EquiConv1d(nn.FieldType(self.group, start_dim * [self.group.regular_repr]), 
        #                nn.FieldType(self.group, start_dim * [self.group.regular_repr]), 
        #                kernel_size),
        #     EquiGroupNorm(nn.FieldType(self.group, start_dim * [self.group.regular_repr]), n_groups),
        #     torch.nn.ReLU(),
        # )
        self.final_conv = EquiConv1dBlock(nn.FieldType(self.group, start_dim * [self.group.regular_repr]), 
                                    nn.FieldType(self.group, start_dim * [self.group.regular_repr]), 
                                    kernel_size)

        self.out_layer = nn.Linear(nn.FieldType(self.group, start_dim * [self.group.regular_repr]), 
                                   self.action_type)


        self.diffusion_step_encoder = diffusion_step_encoder
        self.local_cond_encoder = local_cond_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules


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
        return self.action_type(cat)

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
        sample = rearrange(sample, 'b t h -> (b t) h')
        act_emb = self.enc_a(self.getActionGeometricTensor(sample)).tensor
        sample = rearrange(act_emb, '(b t) h -> b h t', b=B)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1)
        
        # encode local features
        h_local = list()
        if local_cond is not None:
            local_cond = rearrange(local_cond, 'b t h -> b h t')
            resnet, resnet2 = self.local_cond_encoder
            x = resnet(local_cond, global_feature)
            h_local.append(x)
            x = resnet2(local_cond, global_feature)
            h_local.append(x)
        
        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            if idx == 0 and len(h_local) > 0:
                x = x + h_local[0]
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            # The correct condition should be:
            # if idx == (len(self.up_modules)-1) and len(h_local) > 0:
            # However this change will break compatibility with published checkpoints.
            # Therefore it is left as a comment.
            if idx == len(self.up_modules) and len(h_local) > 0:
                x = x + h_local[1]
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        # x = rearrange(x, 'b h t -> (b t) h')
        # x = self.getOutput(x)
        # x = rearrange(x, '(b t) h -> b t h', b=B)
        x = rearrange(x, 'b h t -> (b t) h')
        x = self.out_layer(self.out_layer.in_type(x)).tensor
        x = self.getOutput(x)
        x = rearrange(x, '(b t) h -> b t h', b=B)
        return x

if __name__ == '__main__':
    # from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
    # net = ConditionalUnet1D(10, None, 16*8, 128, [16, 32, 64], kernel_size=5, n_groups=8, cond_predict_scale=True)
    # samp = torch.randn(2, 5, 10)
    # cond = torch.randn(2, 16*8)
    # out = net(samp, 1, global_cond=cond)

    from escnn import gspaces, nn
    from escnn.group import DihedralGroup, CyclicGroup
    g = gspaces.no_base_space(CyclicGroup(4))
    in_type = nn.FieldType(g, 1*[g.regular_repr])
    out_type = nn.FieldType(g, 1*[g.regular_repr])
    cond_type = nn.FieldType(g, 1*[g.regular_repr])

    # layer = EquiConv1d(in_type, out_type, kernel_size=3, stride=1)
    # inp = torch.tensor([
    #     0.1, 0.2, 0.3, 0.4
    # ]).float().reshape(1, 4, 1)
    # inp = repeat(inp, 'b c 1 -> b c 3')

    # ginp = torch.tensor([
    #     0.4, 0.1, 0.2, 0.3
    # ]).float().reshape(1, 4, 1)
    # ginp = repeat(ginp, 'b c 1 -> b c 3')

    # out = layer(inp)
    # gout = layer(ginp)

    # layer = EquiConditionalResidualBlock1D(in_type, out_type, cond_type, kernel_size=5, n_groups=1, cond_predict_scale=True)
    # # inp = torch.randn([2, 16*4, 6])
    # # cond = torch.randn([2, 16*4])
    # inp = torch.tensor([
    #     0.1, 0.2, 0.3, 0.4
    # ]).float().reshape(1, 4, 1)
    # cond = torch.tensor([
    #     0.5, 0.6, 0.7, 0.8
    # ]).float().reshape(1, 4)
    # out = layer(inp, cond)

    # ginp = torch.tensor([
    #     0.4, 0.1, 0.2, 0.3
    # ]).float().reshape(1, 4, 1)
    # gcond = torch.tensor([
    #     0.8, 0.5, 0.6, 0.7
    # ]).float().reshape(1, 4)
    # gout = layer(ginp, gcond)

    # from diffusion_policy.model.equi.equi_conditional_unet1d import EquiDiffusionUNet
    # net1 = EquiDiffusionUNet(4, None, 1, 8, [8, 16, 32], kernel_size=5, n_groups=1, cond_predict_scale=True, N=4)
    
    net = EquiConditionalUnet1D(4, None, 1, 8, [8, 16, 32], kernel_size=5, n_groups=1, cond_predict_scale=False, N=4)
    samp = torch.tensor([[[
        0, 0, 1, 
        1, 0, 0, 0, 1, 0, 
        0
    ]]]).float()
    samp = repeat(samp, 'b 1 d -> b 8 d')
    cond = torch.tensor([[1, 2, 3, 4]]).float()

    gsamp = torch.tensor([[[
        0, 0, 1,
        0, -1, 0, 1, 0, 0,
        0
    ]]]).float()
    gsamp = repeat(gsamp, 'b 1 d -> b 8 d')
    gcond = torch.tensor([[4, 1, 2, 3]]).float()

    # samp = torch.randn(1, 1, 10)
    # cond = torch.randn(1, 1*8)
    out = net(samp, 1, global_cond=cond)
    gout = net(gsamp, 1, global_cond=gcond)

    print(out[0, 0])
    print(gout[0, 0])
    print(1)