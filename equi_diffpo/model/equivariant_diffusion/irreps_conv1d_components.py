import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from equi_diffpo.model.equiformer_v2.activation import SeparableS2Activation
from equi_diffpo.model.equiformer_v2.layer_norm import get_normalization_layer


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class IrrepConv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, SO3_grid,
                 norm=True, max_lmax=3, scale=1, activation=True):
        super().__init__()
        self.d_irrep = (max_lmax + 1) ** 2
        self.max_lmax = max_lmax
        # construct irrep conv weight based on direct sum
        # weight = torch.zeros(out_channels, d_irrep, inp_channels, d_irrep, kernel_size)
        dense_weight = nn.Conv1d(inp_channels, out_channels * (max_lmax + 1), kernel_size).weight
        dense_weight = dense_weight.reshape((max_lmax+1), out_channels, inp_channels, kernel_size)
        bias = nn.Conv1d(1, out_channels, kernel_size).bias.unsqueeze(0).unsqueeze(1)
        self.weight = nn.Parameter(dense_weight)
        self.bias = nn.Parameter(bias)
        self.kernel_size, self.padding = kernel_size, kernel_size // 2
        self.out_channels = out_channels
        self.activation = activation
        if norm:
            self.norm = get_normalization_layer('rms_norm_sh', lmax=max_lmax, num_channels=out_channels)
        else:
            self.norm = nn.Identity()
        if activation:
            self.SO3_grid = SO3_grid
            self.gating_linear = torch.nn.Linear(self.out_channels, self.out_channels)
            self.s2_act = SeparableS2Activation(max_lmax, max_lmax)
        self.scale = scale

    def forward(self, x):
        # x in shape (B, (C, irrep), n_pts)
        assert x.dim() == 3
        # h = nn.functional.conv1d(x, self.weight, self.bias, padding=self.padding)
        l_start, l_end = 0, 0
        h = []
        x = einops.rearrange(x, 'b (c i) n -> b c i n', i=self.d_irrep)
        for l in range(self.max_lmax + 1):
            l_order = 2 * l + 1
            l_end += l_order
            x_l = einops.rearrange(x[:, :, l_start:l_end, :], 'b c i n -> (b i) c n')
            h_l = nn.functional.conv1d(x_l, self.weight[l, :, :, :], padding=self.padding)
            h_l = einops.rearrange(h_l, '(b i) c n -> (b n) i c', i=l_order)
            if l == 0:
                h_l = h_l + self.bias
            h.append(h_l)
            l_start = l_end
        h = torch.cat(h, dim=1)
        h = self.norm(h)
        if self.activation:
            gating_scalars = self.gating_linear(h.narrow(1, 0, 1))  # This is different from Equiformer
            h = self.s2_act(gating_scalars, h, self.SO3_grid)
        h = einops.rearrange(h, '(b n) i c -> b (c i) n', n=x.shape[-1])
        if self.scale != 1:
            h = nn.functional.interpolate(h, scale_factor=self.scale)
        return h


class IrrepLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, lmax, bias=True):
        '''
            1. Use `torch.einsum` to prevent slicing and concatenation
            2. Need to specify some behaviors in `no_weight_decay` and weight initialization.
        '''
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lmax = lmax

        self.weight = torch.nn.Parameter(torch.randn((self.lmax + 1), out_features, in_features))
        bound = 1 / math.sqrt(self.in_features)
        torch.nn.init.uniform_(self.weight, -bound, bound)
        self.bias = torch.nn.Parameter(torch.zeros(out_features))

        expand_index = torch.zeros([(lmax + 1) ** 2]).long()
        for l in range(lmax + 1):
            start_idx = l ** 2
            length = 2 * l + 1
            expand_index[start_idx: (start_idx + length)] = l
        self.register_buffer('expand_index', expand_index)

    def forward(self, input):
        # input_embedding (B, (Cin, irrep), 1)
        # out (B, (Cout, irrep), 1)

        weight = torch.index_select(self.weight, dim=0, index=self.expand_index)  # [(L_max + 1) ** 2, C_out, C_in]
        input = einops.rearrange(input, 'b (i m) n -> b m (i n)', i=self.in_features)
        out = torch.einsum('bmi, moi -> bmo', input, weight)  # [N, (L_max + 1) ** 2, C_out]
        bias = self.bias.view(1, 1, self.out_features)
        out[:, 0:1, :] = out.narrow(1, 0, 1) + bias
        out = einops.rearrange(out, 'b (m n) o -> b (o m) n', n=1)
        return out
