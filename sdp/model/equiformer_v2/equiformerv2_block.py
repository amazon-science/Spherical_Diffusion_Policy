import torch
import torch.nn as nn
import math
import torch_geometric
import copy

from sdp.model.equiformer_v2.radial_function import RadialFunction
from sdp.model.equiformer_v2.so2_ops import SO2_Convolution
from sdp.model.equiformer_v2.so3 import SO3_Embedding, SO3_LinearV2
from sdp.model.equiformer_v2.drop import GraphDropPath, EquivariantDropoutArraySphericalHarmonics
from sdp.model.equiformer_v2.activation import (
    SmoothLeakyReLU,
    SeparableS2Activation,
    S2Activation
)
from sdp.model.equiformer_v2.layer_norm import (
    EquivariantLayerNormArray,
    EquivariantLayerNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonics,
    get_normalization_layer
)


class SO2EquivariantGraphAttention(torch.nn.Module):
    """
    SO2EquivariantGraphAttention: Perform MLP attention + non-linear message passing
        SO(2) Convolution with radial function -> S2 Activation -> SO(2) Convolution -> attention weights and non-linear messages
        attention weights * non-linear messages -> Linear

    Args:
        sphere_channels (int):      Number of spherical channels
        hidden_channels (int):      Number of hidden channels used during the SO(2) conv
        num_heads (int):            Number of attention heads
        attn_alpha_head (int):      Number of channels for alpha vector in each attention head
        attn_value_head (int):      Number of channels for value vector in each attention head
        output_channels (int):      Number of output channels
        lmax_list (list:int):       List of degrees (l) for each resolution
        mmax_list (list:int):       List of orders (m) for each resolution

        SO3_rotation (ModuleList,list:SO3_Rotation): Class to calculate Wigner-D matrices and rotate embeddings
        mappingReduced (CoefficientMappingModule): Class to convert l and m indices once node embedding is rotated
        SO3_grid (SO3_grid):        Class used to convert from grid the spherical harmonic representations

        edge_channels_list (list:int):  List of sizes of invariant edge embedding. For example, [input_channels, hidden_channels, hidden_channels].

        use_m_share_rad (bool):     Whether all m components within a type-L vector of one channel share radial function weights

        activation (str):           Type of activation function
        use_attn_renorm (bool):     Whether to re-normalize attention weights
        use_gate_act (bool):        If `True`, use gate activation. Otherwise, use S2 activation.
        use_sep_s2_act (bool):      If `True`, use separable S2 activation when `use_gate_act` is False.

        alpha_drop (float):         Dropout rate for attention weights
    """

    def __init__(
            self,
            sphere_channels,  # 128
            hidden_channels,  # 64
            num_heads,  # 8
            attn_alpha_channels,  # 64
            attn_value_channels,  # 16
            output_channels,  # 128
            lmax_list,  # [6]
            mmax_list,  # [3]
            SO3_rotation,
            mappingReduced,
            SO3_grid,
            edge_channels_list,
            use_m_share_rad=False,
            use_attn_renorm=True,
            use_sep_s2_act=True,
            alpha_drop=0.0,
    ):
        super(SO2EquivariantGraphAttention, self).__init__()

        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.attn_alpha_channels = attn_alpha_channels
        self.attn_value_channels = attn_value_channels
        self.output_channels = output_channels
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.num_resolutions = len(self.lmax_list)

        self.SO3_rotation = SO3_rotation
        self.mappingReduced = mappingReduced
        self.SO3_grid = SO3_grid

        # Create edge scalar (invariant to rotations) features

        self.edge_channels_list = copy.deepcopy(edge_channels_list)
        self.use_m_share_rad = use_m_share_rad

        self.source_embedding, self.target_embedding = None, None
        self.use_attn_renorm = use_attn_renorm
        self.use_sep_s2_act = use_sep_s2_act

        # Create SO(2) convolution blocks
        extra_m0_output_channels = self.num_heads * self.attn_alpha_channels  # 8 * 64
        if self.use_sep_s2_act:
            extra_m0_output_channels = extra_m0_output_channels + self.hidden_channels

        if self.use_m_share_rad:
            self.edge_channels_list = self.edge_channels_list + [2 * self.sphere_channels * (max(self.lmax_list) + 1)]
            self.rad_func = RadialFunction(self.edge_channels_list)
            expand_index = torch.zeros([(max(self.lmax_list) + 1) ** 2]).long()
            for l in range(max(self.lmax_list) + 1):
                start_idx = l ** 2
                length = 2 * l + 1
                expand_index[start_idx: (start_idx + length)] = l
            self.register_buffer('expand_index', expand_index)

        self.so2_conv_1 = SO2_Convolution(
            2 * self.sphere_channels,
            self.hidden_channels,
            self.lmax_list,
            self.mmax_list,
            self.mappingReduced,
            internal_weights=(
                False if not self.use_m_share_rad
                else True
            ),
            edge_channels_list=(
                self.edge_channels_list if not self.use_m_share_rad
                else None
            ),
            extra_m0_output_channels=extra_m0_output_channels  # for attention weights and/or gate activation
        )

        if self.use_attn_renorm:
            self.alpha_norm = torch.nn.LayerNorm(self.attn_alpha_channels)
        else:
            self.alpha_norm = torch.nn.Identity()
        self.alpha_act = SmoothLeakyReLU()
        self.alpha_dot = torch.nn.Parameter(torch.randn(self.num_heads, self.attn_alpha_channels))
        # torch_geometric.nn.inits.glorot(self.alpha_dot) # Following GATv2
        std = 1.0 / math.sqrt(self.attn_alpha_channels)
        torch.nn.init.uniform_(self.alpha_dot, -std, std)

        self.alpha_dropout = None
        if alpha_drop != 0.0:
            self.alpha_dropout = torch.nn.Dropout(alpha_drop)

        if self.use_sep_s2_act:
            # separable S2 activation
            self.s2_act = SeparableS2Activation(
                lmax=max(self.lmax_list),
                mmax=max(self.mmax_list)
            )
        else:
            # S2 activation
            self.s2_act = S2Activation(
                lmax=max(self.lmax_list),
                mmax=max(self.mmax_list)
            )

        self.so2_conv_2 = SO2_Convolution(
            self.hidden_channels,
            self.num_heads * self.attn_value_channels,
            self.lmax_list,
            self.mmax_list,
            self.mappingReduced,
            internal_weights=True,
            edge_channels_list=None,
            extra_m0_output_channels=None  # for attention weights
        )

        self.proj = SO3_LinearV2(self.num_heads * self.attn_value_channels, self.output_channels,
                                 lmax=self.lmax_list[0])

    def forward(
            self,
            x_src,
            x_dst,
            edge_distance,
            edge_src,
            edge_dst,
    ):

        # Compute edge scalar features (invariant to rotations)
        x_edge = edge_distance
        length = len(x_dst.embedding)
        output_src = x_src.clone()
        output_dst = x_dst.clone()

        output_src._expand_edge(edge_src)
        output_dst._expand_edge(edge_dst)

        x_message_data = torch.cat((output_src.embedding, output_dst.embedding), dim=2)

        x_message = SO3_Embedding(
            0,
            output_dst.lmax_list.copy(),
            output_dst.num_channels * 2,
            device=output_dst.device,
            dtype=output_dst.dtype
        )
        x_message.set_embedding(x_message_data)
        x_message.set_lmax_mmax(self.lmax_list.copy(), self.mmax_list.copy())

        # radial function (scale all m components within a type-L vector of one channel with the same weight)
        if self.use_m_share_rad:
            x_edge_weight = self.rad_func(x_edge)
            x_edge_weight = x_edge_weight.reshape(-1, (max(self.lmax_list) + 1), 2 * self.sphere_channels)
            x_edge_weight = torch.index_select(x_edge_weight, dim=1,
                                               index=self.expand_index)  # [E, (L_max + 1) ** 2, C]
            x_message.embedding = x_message.embedding * x_edge_weight

        # Rotate the irreps to align with the edge
        x_message._rotate(self.SO3_rotation, self.lmax_list, self.mmax_list)

        # First SO(2)-convolution
        x_message, x_0_extra = self.so2_conv_1(x_message, x_edge)

        # Activation
        x_alpha_num_channels = self.num_heads * self.attn_alpha_channels

        if self.use_sep_s2_act:
            x_0_gating = x_0_extra.narrow(1, x_alpha_num_channels,
                                          x_0_extra.shape[1] - x_alpha_num_channels)  # for activation
            x_0_alpha = x_0_extra.narrow(1, 0, x_alpha_num_channels)  # for attention weights
            x_message.embedding = self.s2_act(x_0_gating, x_message.embedding, self.SO3_grid)
        else:
            x_0_alpha = x_0_extra
            x_message.embedding = self.s2_act(x_message.embedding, self.SO3_grid)
            ##x_message._grid_act(self.SO3_grid, self.value_act, self.mappingReduced)

        # Second SO(2)-convolution
        x_message = self.so2_conv_2(x_message, x_edge)

        # Attention weights
        x_0_alpha = x_0_alpha.reshape(-1, self.num_heads, self.attn_alpha_channels)
        x_0_alpha = self.alpha_norm(x_0_alpha)
        x_0_alpha = self.alpha_act(x_0_alpha)
        alpha = torch.einsum('bik, ik -> bi', x_0_alpha, self.alpha_dot)
        alpha = torch_geometric.utils.softmax(alpha, edge_dst)
        alpha = alpha.reshape(alpha.shape[0], 1, self.num_heads, 1)
        if self.alpha_dropout is not None:
            alpha = self.alpha_dropout(alpha)

        # Attention weights * non-linear messages
        attn = x_message.embedding
        attn = attn.reshape(attn.shape[0], attn.shape[1], self.num_heads, self.attn_value_channels)
        attn = attn * alpha
        attn = attn.reshape(attn.shape[0], attn.shape[1], self.num_heads * self.attn_value_channels)
        x_message.embedding = attn

        # Rotate back the irreps
        x_message._rotate_inv(self.SO3_rotation, self.mappingReduced)

        # Compute the sum of the incoming neighboring messages for each target node
        x_message._reduce_edge(edge_dst, length)

        # Project
        out_embedding = self.proj(x_message)

        return out_embedding


class FeedForwardNetwork(torch.nn.Module):
    """
    FeedForwardNetwork: Perform feedforward network with S2 activation or gate activation

    Args:
        sphere_channels (int):      Number of spherical channels
        hidden_channels (int):      Number of hidden channels used during feedforward network
        output_channels (int):      Number of output channels

        lmax_list (list:int):       List of degrees (l) for each resolution
        mmax_list (list:int):       List of orders (m) for each resolution

        SO3_grid (SO3_grid):        Class used to convert from grid the spherical harmonic representations

        activation (str):           Type of activation function
        use_gate_act (bool):        If `True`, use gate activation. Otherwise, use S2 activation
        use_grid_mlp (bool):        If `True`, use projecting to grids and performing MLPs.
        use_sep_s2_act (bool):      If `True`, use separable grid MLP when `use_grid_mlp` is True.
    """

    def __init__(
            self,
            sphere_channels,
            hidden_channels,
            output_channels,
            lmax_list,
            mmax_list,
            SO3_grid,
            use_grid_mlp=False,
            use_sep_s2_act=True
    ):
        super(FeedForwardNetwork, self).__init__()
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.num_resolutions = len(lmax_list)
        self.sphere_channels_all = self.num_resolutions * self.sphere_channels
        self.SO3_grid = SO3_grid
        self.use_grid_mlp = use_grid_mlp
        self.use_sep_s2_act = use_sep_s2_act

        self.max_lmax = max(self.lmax_list)

        self.so3_linear_1 = SO3_LinearV2(self.sphere_channels_all, self.hidden_channels, lmax=self.max_lmax)
        if self.use_grid_mlp:
            if self.use_sep_s2_act:
                self.scalar_mlp = nn.Sequential(
                    nn.Linear(self.sphere_channels_all, self.hidden_channels, bias=True),
                    nn.SiLU(),
                )
            else:
                self.scalar_mlp = None
            self.grid_mlp = nn.Sequential(
                nn.Linear(self.hidden_channels, self.hidden_channels, bias=False),
                nn.SiLU(),
                nn.Linear(self.hidden_channels, self.hidden_channels, bias=False),
                nn.SiLU(),
                nn.Linear(self.hidden_channels, self.hidden_channels, bias=False)
            )
        else:
            if self.use_sep_s2_act:
                self.gating_linear = torch.nn.Linear(self.sphere_channels_all, self.hidden_channels)
                self.s2_act = SeparableS2Activation(self.max_lmax, self.max_lmax)
            else:
                self.gating_linear = None
                self.s2_act = S2Activation(self.max_lmax, self.max_lmax)
        self.so3_linear_2 = SO3_LinearV2(self.hidden_channels, self.output_channels, lmax=self.max_lmax)

    def forward(self, input_embedding):

        gating_scalars = None
        if self.use_grid_mlp:
            if self.use_sep_s2_act:
                gating_scalars = self.scalar_mlp(input_embedding.embedding.narrow(1, 0, 1))
        else:
            if self.gating_linear is not None:
                gating_scalars = self.gating_linear(input_embedding.embedding.narrow(1, 0, 1))

        input_embedding = self.so3_linear_1(input_embedding)

        if self.use_grid_mlp:
            # Project to grid
            input_embedding_grid = input_embedding.to_grid(self.SO3_grid, lmax=self.max_lmax)
            # Perform point-wise operations
            input_embedding_grid = self.grid_mlp(input_embedding_grid)
            # Project back to spherical harmonic coefficients
            input_embedding._from_grid(input_embedding_grid, self.SO3_grid, lmax=self.max_lmax)

            if self.use_sep_s2_act:
                input_embedding.embedding = torch.cat(
                    (gating_scalars, input_embedding.embedding.narrow(1, 1, input_embedding.embedding.shape[1] - 1)),
                    dim=1
                )
        else:
            if self.use_sep_s2_act:
                input_embedding.embedding = self.s2_act(gating_scalars, input_embedding.embedding, self.SO3_grid)
            else:
                input_embedding.embedding = self.s2_act(input_embedding.embedding, self.SO3_grid)

        input_embedding = self.so3_linear_2(input_embedding)

        return input_embedding


class TransBlock(torch.nn.Module):
    """

    Args:
        sphere_channels (int):      Number of spherical channels
        attn_hidden_channels (int): Number of hidden channels used during SO(2) graph attention
        num_heads (int):            Number of attention heads
        attn_alpha_head (int):      Number of channels for alpha vector in each attention head
        attn_value_head (int):      Number of channels for value vector in each attention head
        ffn_hidden_channels (int):  Number of hidden channels used during feedforward network
        output_channels (int):      Number of output channels

        lmax_list (list:int):       List of degrees (l) for each resolution
        mmax_list (list:int):       List of orders (m) for each resolution

        SO3_rotation (list,ModuleList): Class to calculate Wigner-D matrices and rotate embeddings
        mappingReduced (CoefficientMappingModule): Class to convert l and m indices once node embedding is rotated
        SO3_grid (SO3_grid):        Class used to convert from grid the spherical harmonic representations

        edge_channels_list (list:int):  List of sizes of invariant edge embedding. For example, [input_channels, hidden_channels, hidden_channels].

        use_m_share_rad (bool):     Whether all m components within a type-L vector of one channel share radial function weights

        attn_activation (str):      Type of activation function for SO(2) graph attention
         use_attn_renorm (bool):     Whether to re-normalize attention weights
        ffn_activation (str):       Type of activation function for feedforward network
        use_gate_act (bool):        If `True`, use gate activation. Otherwise, use S2 activation
        use_grid_mlp (bool):        If `True`, use projecting to grids and performing MLPs for FFN.
        use_sep_s2_act (bool):      If `True`, use separable S2 activation when `use_gate_act` is False.

        norm_type (str):            Type of normalization layer (['layer_norm', 'layer_norm_sh'])

        alpha_drop (float):         Dropout rate for attention weights
        drop_path_rate (float):     Drop path rate
        proj_drop (float):          Dropout rate for outputs of attention and FFN
    """

    def __init__(
            self,
            sphere_channels,
            attn_hidden_channels,
            num_heads,
            attn_alpha_channels,
            attn_value_channels,
            ffn_hidden_channels,
            output_channels,

            lmax_list,
            mmax_list,

            SO3_rotation,
            mappingReduced,
            SO3_grid,

            edge_channels_list,
            use_m_share_rad=False,

            use_attn_renorm=True,

            use_grid_mlp=False,
            use_sep_s2_act=True,

            norm_type='rms_norm_sh',

            alpha_drop=0.0,
            drop_path_rate=0.0,
            proj_drop=0.0
    ):
        super(TransBlock, self).__init__()

        max_lmax = max(lmax_list)
        self.norm_1_src = get_normalization_layer(norm_type, lmax=max_lmax, num_channels=sphere_channels)
        self.norm_1_dst = get_normalization_layer(norm_type, lmax=max_lmax, num_channels=sphere_channels)

        self.ga = SO2EquivariantGraphAttention(
            sphere_channels=sphere_channels,
            hidden_channels=attn_hidden_channels,
            num_heads=num_heads,
            attn_alpha_channels=attn_alpha_channels,
            attn_value_channels=attn_value_channels,
            output_channels=sphere_channels,
            lmax_list=lmax_list,
            mmax_list=mmax_list,
            SO3_rotation=SO3_rotation,
            mappingReduced=mappingReduced,
            SO3_grid=SO3_grid,
            edge_channels_list=edge_channels_list,
            use_m_share_rad=use_m_share_rad,
            use_attn_renorm=use_attn_renorm,
            use_sep_s2_act=use_sep_s2_act,
            alpha_drop=alpha_drop,
        )

        self.drop_path = GraphDropPath(drop_path_rate) if drop_path_rate > 0. else None
        self.proj_drop = EquivariantDropoutArraySphericalHarmonics(proj_drop,
                                                                   drop_graph=False) if proj_drop > 0.0 else None

        self.norm_2 = get_normalization_layer(norm_type, lmax=max_lmax, num_channels=sphere_channels)

        self.ffn = FeedForwardNetwork(
            sphere_channels=sphere_channels,
            hidden_channels=ffn_hidden_channels,
            output_channels=output_channels,
            lmax_list=lmax_list,
            mmax_list=mmax_list,
            SO3_grid=SO3_grid,
            use_grid_mlp=use_grid_mlp,
            use_sep_s2_act=use_sep_s2_act
        )

        if sphere_channels != output_channels:
            self.ffn_shortcut = SO3_LinearV2(sphere_channels, output_channels, lmax=max_lmax)
        else:
            self.ffn_shortcut = None

    def forward(
            self,
            x_src,  # SO3_Embedding
            x_dst,  # SO3_Embedding
            edge_distance,
            edge_src,
            edge_dst,
            batch=None  # for GraphDropPath
    ):
        output_src = x_src.clone()
        output_dst = x_dst.clone()
        x_res = output_dst.embedding
        output_src.embedding = self.norm_1_src(output_src.embedding)
        output_dst.embedding = self.norm_1_dst(output_dst.embedding)

        output_embedding = self.ga(output_src,
                                   output_dst,
                                   edge_distance,
                                   edge_src,
                                   edge_dst)

        if self.drop_path is not None:
            output_embedding.embedding = self.drop_path(output_embedding.embedding, batch)
        if self.proj_drop is not None:
            output_embedding.embedding = self.proj_drop(output_embedding.embedding, batch)

        output_embedding.embedding = output_embedding.embedding + x_res

        x_res = output_embedding.embedding
        output_embedding.embedding = self.norm_2(output_embedding.embedding)
        output_embedding = self.ffn(output_embedding)

        if self.drop_path is not None:
            output_embedding.embedding = self.drop_path(output_embedding.embedding, batch)
        if self.proj_drop is not None:
            output_embedding.embedding = self.proj_drop(output_embedding.embedding, batch)

        if self.ffn_shortcut is not None:
            shortcut_embedding = SO3_Embedding(
                0,
                output_embedding.lmax_list.copy(),
                self.ffn_shortcut.in_features,
                device=output_embedding.device,
                dtype=output_embedding.dtype
            )
            shortcut_embedding.set_embedding(x_res)
            shortcut_embedding.set_lmax_mmax(output_embedding.lmax_list.copy(), output_embedding.lmax_list.copy())
            shortcut_embedding = self.ffn_shortcut(shortcut_embedding)
            x_res = shortcut_embedding.embedding

        output_embedding.embedding = output_embedding.embedding + x_res

        return output_embedding
