import math
import torch.nn as nn
import pytorch3d.transforms as pytorch3d_transforms
import dgl.geometry as dgl_geo
from einops import rearrange

from equi_diffpo.model.equiformer_v2.gaussian_rbf import GaussianRadialBasisLayer, \
    GaussianRadialBasisLayerFiniteCutoff
from equi_diffpo.model.equiformer_v2.edge_rot_mat import init_edge_rot_mat2
from equi_diffpo.model.equiformer_v2.layer_norm import get_normalization_layer
from equi_diffpo.model.equiformer_v2.so3 import (
    CoefficientMappingModule,
    SO3_Embedding,
    SO3_Grid,
    SO3_Rotation,
    SO3_LinearV2
)
from equi_diffpo.model.equiformer_v2.module_list import ModuleListInfo
from equi_diffpo.model.equiformer_v2.radial_function import RadialFunction
from equi_diffpo.model.equiformer_v2.equiformerv2_block import (
    TransBlock, FeedForwardNetwork,
)
from equi_diffpo.model.equiformer_v2.se3_transformation import rot_pcd
from equi_diffpo.model.equiformer_v2.connectivity import RadiusGraph, FpsPool, AdaptiveOriginPool, FpsKnnPool
import torch
import numpy as np
import einops

debug = False


class EquiFormerEnc(nn.Module):

    def __init__(
            self,
            c_dim=64,
            max_neighbors=(16, 16, 16, 16),
            max_radius=(0.05, 0.2, 0.8, 3),
            # n_pts: 1024, 256, 64, 16, 1
            pool_ratio=(0.25, 0.25, 0.25),  # the last pool_ratio is adaptive
            sphere_channels=(32, 64, 128),  # the last sphere_channels will be defined by c_dim
            attn_hidden_channels=(32, 64, 128, 256),
            attn_alpha_channels=(8, 16, 32, 64),
            attn_value_channels=(4, 8, 16, 32),
            ffn_hidden_channels=(32, 64, 128, 256),
            edge_channels=(16, 32, 64, 128),
            num_distance_basis=(64, 64, 64, 64),
            num_heads=4,
            pcd_noise=0,
            norm_type='rms_norm_sh',
            deterministic=False,
            lmax=2,
            mmax=2,
            norm=True,
            grid_resolution=12,

            use_m_share_rad=False,
            distance_function="gaussian_soft",

            use_attn_renorm=True,
            use_grid_mlp=False,
            use_sep_s2_act=True,

            alpha_drop=0.1,
            drop_path_rate=0.,
            proj_drop=0.1,

            weight_init='normal',
            pool_method='fpsknn',  # in fps, fpsknn
            n_cam=1,
            n_proprio=1
    ):
        super().__init__()
        # -----------------------------------EquiformerV2 GNN Enc--------------------------------
        assert len(max_neighbors) == len(sphere_channels)
        self.max_neighbors = max_neighbors
        self.pool_ratio = pool_ratio
        self.sphere_channels = sphere_channels
        self.sphere_channels += (c_dim,)
        self.c_dim = c_dim
        self.pcd_noise = pcd_noise
        self.attn_hidden_channels = attn_hidden_channels
        self.num_heads = num_heads
        self.attn_alpha_channels = attn_alpha_channels
        self.attn_value_channels = attn_value_channels
        self.ffn_hidden_channels = ffn_hidden_channels
        self.norm_type = norm_type
        self.lmax_list = [lmax, ]
        self.mmax_list = [mmax, ]
        self.grid_resolution = grid_resolution
        self.edge_channels = edge_channels
        self.use_m_share_rad = use_m_share_rad
        self.distance_function = distance_function
        self.num_distance_basis = num_distance_basis
        self.use_attn_renorm = use_attn_renorm
        self.use_grid_mlp = use_grid_mlp
        self.use_sep_s2_act = use_sep_s2_act
        self.alpha_drop = alpha_drop
        self.drop_path_rate = drop_path_rate
        self.proj_drop = proj_drop
        self.weight_init = weight_init
        assert self.weight_init in ['normal', 'uniform']
        self.max_radius = max_radius
        print('GNN graph radius', self.max_radius)

        self.deterministic = deterministic
        self.device = torch.cuda.current_device()
        self.n_scales = len(self.max_radius)
        self.num_resolutions = len(self.lmax_list)
        self.pcd_channels = 3
        self.sphere_channels_all = self.num_resolutions * self.sphere_channels[0]

        self.n_cam = n_cam
        self.n_proprio = n_proprio

        assert self.distance_function in [
            'gaussian', 'gaussian_soft'
        ]

        # Initialize the module that compute WignerD matrices and other values for spherical harmonic calculations
        self.SO3_rotation = nn.ModuleList()
        for i in range(self.num_resolutions):
            self.SO3_rotation.append(SO3_Rotation(self.lmax_list[i]))

        # Initialize conversion between degree l and order m layouts
        self.mappingReduced = CoefficientMappingModule(self.lmax_list, self.mmax_list)

        # Initialize the transformations between spherical and grid representations
        self.SO3_grid = ModuleListInfo('({}, {})'.format(max(self.lmax_list), max(self.lmax_list)))
        for l in range(max(self.lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(max(self.lmax_list) + 1):
                SO3_m_grid.append(
                    SO3_Grid(
                        l,
                        m,
                        resolution=self.grid_resolution,
                        normalization='component'
                    )
                )
            self.SO3_grid.append(SO3_m_grid)

        ## Down Blocks
        self.down_blocks = torch.nn.ModuleList()
        for n in range(len(self.max_neighbors)):
            # Initialize the sizes of radial functions (input channels and 2 hidden channels)
            edge_channels_list = [int(self.num_distance_basis[n])] + [self.edge_channels[n]] * 2

            block = torch.nn.ModuleDict()
            if n != len(self.max_neighbors) - 1:
                if pool_method == 'fps':
                    block['pool'] = FpsPool(ratio=self.pool_ratio[n], random_start=not self.deterministic,
                                            r=self.max_radius[n], max_num_neighbors=self.max_neighbors[n])
                elif pool_method == 'fpsknn':
                    block['pool'] = FpsKnnPool(ratio=self.pool_ratio[n], random_start=not self.deterministic,
                                               r=3, max_num_neighbors=self.max_neighbors[n])
            else:
                block['pool'] = AdaptiveOriginPool(random_start=not self.deterministic,
                                                   r=self.max_radius[n], max_num_neighbors=self.max_neighbors[n])

            # Initialize the function used to measure the distances between atoms
            if self.distance_function == 'gaussian':
                block['distance_expansion'] = GaussianRadialBasisLayer(num_basis=self.num_distance_basis[n],
                                                                       cutoff=self.max_radius[n])
            elif self.distance_function == 'gaussian_soft':
                block['distance_expansion'] = GaussianRadialBasisLayerFiniteCutoff(num_basis=self.num_distance_basis[n],
                                                                                   cutoff=self.max_radius[n] * 0.99)
            else:
                raise ValueError

            # scale_out_sphere_channels = self.sphere_channels[min(n + 1, self.n_scales - 1)]
            if debug:
                print('down block {}, {}->{} channels'.format(n, self.sphere_channels[n],
                                                              self.sphere_channels[n+1]))
            block['transblock'] = TransBlock(
                sphere_channels=self.sphere_channels[n],
                attn_hidden_channels=self.attn_hidden_channels[n],
                num_heads=self.num_heads,
                attn_alpha_channels=self.attn_alpha_channels[n],
                attn_value_channels=self.attn_value_channels[n],
                ffn_hidden_channels=self.ffn_hidden_channels[n],
                output_channels=self.sphere_channels[n+1],
                lmax_list=self.lmax_list,
                mmax_list=self.mmax_list,
                SO3_rotation=self.SO3_rotation,
                mappingReduced=self.mappingReduced,
                SO3_grid=self.SO3_grid,
                edge_channels_list=edge_channels_list,
                use_m_share_rad=self.use_m_share_rad,
                use_attn_renorm=self.use_attn_renorm,
                use_grid_mlp=self.use_grid_mlp,
                use_sep_s2_act=self.use_sep_s2_act,
                norm_type=self.norm_type,
                alpha_drop=self.alpha_drop,
                drop_path_rate=self.drop_path_rate,
                proj_drop=self.proj_drop
            )

            self.down_blocks.append(block)

        if norm:
            self.norm = get_normalization_layer(self.norm_type, lmax=max(self.lmax_list),
                                                num_channels=self.sphere_channels[-1])
        else:
            self.norm = None

        self.apply(self._init_weights)
        self.apply(self._uniform_init_rad_func_linear_weights)

        # Weights for message initialization
        self.type0_linear = nn.Linear(self.pcd_channels, self.sphere_channels_all, bias=True)

    def forward(self, nobs, sanitycheck=False):
        """
        Arguments:
        -> embedded SH graph, where features are in shape (b (lmax+1)**2 c), signal over S2 irreps
        """
        pc = nobs["point_cloud"][self.n_proprio-1::self.n_proprio, ...]  # [b n_pts xyzrgb]

        # add noise to PCD to avoid grid effect of camera
        if self.pcd_noise > 0:
            for tries in range(100):
                trans_noise = torch.rand_like(pc[..., :3].repeat(4, 1, 1))
                trans_noise -= 0.5
                trans_noise *= 2 * self.pcd_noise
                insider = trans_noise.norm(dim=-1, p=2) < self.pcd_noise
                trans_noise = trans_noise[insider[:, 0]]
                if trans_noise.shape[0] > pc.shape[0]:
                    trans_noise = trans_noise[:pc.shape[0]]
                    pc[..., :3] += trans_noise
                    break
            if tries == 99:
                raise 'Can not find valid PCD noise, tried={} times'.format(tries)

        ee_quat_xyzw = nobs["robot0_eef_quat"]  # [(b t) 4]
        ee_q = nobs["robot0_gripper_qpos"]  # [(b t) 2]
        ee_q = rearrange(ee_q, '(b t) n -> b (n t)', t=self.n_proprio)
        ee_pose = nobs["robot0_eef_pos"]  # [(b t) 3]
        ee_pose = rearrange(ee_pose, '(b t) d -> b d t', t=self.n_proprio).clone()
        batch_size, num_points, _ = pc.shape  # (b t) n d
        xyzrgb = rearrange(pc, "b npts d -> (b npts) d")
        # ee_quat_xyzw = rearrange(ee_quat, "b t d -> (b t) d")
        ee_rot_vec = pytorch3d_transforms.quaternion_to_matrix(ee_quat_xyzw[:, (3, 0, 1, 2)])  # [(b t) 3 3]
        ee_rot_vec = rearrange(ee_rot_vec, "(b t) c d -> b c (t d)", t=self.n_proprio)

        # removing up to 24 duplicate points
        # xyz = pcd.reshape(B * T, -1, 3).clone()
        # fps_idx = dgl_geo.farthest_point_sampler(xyz, 1000, start_idx=0)
        # xyz = xyz[torch.arange(B * T).unsqueeze(1), fps_idx, :]

        # import open3d as o3d
        # idx = 0
        # vis_pcd = o3d.geometry.PointCloud()
        # # Assign points and colors to the PointCloud
        # vis_pcd.points = o3d.utility.Vector3dVector(pc[idx, :, :3].detach().cpu().numpy())
        # vis_pcd.colors = o3d.utility.Vector3dVector(pc[idx, :, 3:].detach().cpu().numpy())
        # # Visualize the expert pose
        # expert_pose = ee_pose[idx:idx + 1].clone().repeat(6, 1)
        # r = 0.1
        # expert_pose[:3] += r * ee_rot_mat[idx].permute(1, 0)
        # expert_pose[3:] -= r * ee_rot_mat[idx].permute(1, 0)
        # expert_edge = torch.arange(6).reshape((2, 3)).permute(1, 0)
        # expert_color = torch.zeros((3, 3))
        # expert_color[torch.arange(3), torch.arange(3)] = 1
        # expert_coord = o3d.geometry.LineSet()
        # expert_coord.points = o3d.utility.Vector3dVector(expert_pose.detach().cpu().numpy())
        # expert_coord.lines = o3d.utility.Vector2iVector(expert_edge.detach().cpu().numpy())
        # expert_coord.colors = o3d.utility.Vector3dVector(expert_color.detach().cpu().numpy())
        # # Visualize the point cloud
        # o3d.visualization.draw_geometries([vis_pcd, expert_coord])

        self.dtype = xyzrgb.dtype
        self.device = xyzrgb.device
        node_coord = xyzrgb[..., :3]  # (b npts) 3
        total_points = node_coord.shape[0]
        node_feature = xyzrgb[..., 3:]  # (b npts) 3
        batch = torch.arange(0, batch_size).repeat_interleave(num_points).to(self.device)  # (b npts)

        node_src = None
        node_dst = None
        ########### Downstream Block #############
        for n, block in enumerate(self.down_blocks):
            #### Downsampling ####
            pool_graph = block['pool'](node_coord_src=node_coord, batch_src=batch)
            node_coord_dst, edge_src, edge_dst, degree, batch_dst, node_idx = pool_graph

            edge_vec = node_coord.index_select(0, edge_src) - node_coord_dst.index_select(0, edge_dst)
            if not self.deterministic:
                edge_vec += (torch.rand_like(edge_vec) - 0.5) * 1e-6  # To avoid edge_vec=0
            edge_vec = edge_vec.detach()
            edge_length = torch.norm(edge_vec, dim=-1).detach()
            try:
                edge_length.min()
            except:
                print('Error: no valid edge, consider increase max_radius at level ', n)
                print(edge_length)

            # Compute 3x3 rotation matrix per edge
            edge_rot_mat = self._init_edge_rot_mat(edge_vec)

            # Initialize the WignerD matrices and other values for spherical harmonic calculations
            for i in range(self.num_resolutions):
                self.SO3_rotation[i].set_wigner(edge_rot_mat)

            if node_src is None:
                node_src = SO3_Embedding(
                    total_points,
                    self.lmax_list,
                    self.sphere_channels[n],
                    self.device,
                    self.dtype,
                )

                offset_res = 0
                offset = 0
                # Initialize the l = 0, m = 0 coefficients for each resolution
                for i in range(self.num_resolutions):
                    if self.num_resolutions == 1:
                        node_src.embedding[:, offset_res, :] = self.type0_linear(node_feature)
                    else:
                        node_src.embedding[:, offset_res, :] = self.type0_linear(node_feature)[:,
                                                               offset: offset + self.sphere_channels[0]]
                    offset = offset + self.sphere_channels[0]
                    offset_res = offset_res + int((self.lmax_list[i] + 1) ** 2)

            # Edge encoding (distance and atom edge)
            edge_attr = block['distance_expansion'](edge_length)
            node_dst = SO3_Embedding(
                batch_size,
                self.lmax_list,
                self.sphere_channels[n],
                self.device,
                self.dtype,
            )
            if n != len(self.max_neighbors) - 1:
                node_dst.set_embedding(node_src.embedding[node_idx])
            node_dst.set_lmax_mmax(self.lmax_list.copy(), self.mmax_list.copy())
            node_dst = block['transblock'](node_src,
                                           node_dst,
                                           edge_attr,
                                           edge_src,
                                           edge_dst,
                                           batch=batch)
            node_src = node_dst
            node_coord_src = node_coord.clone()
            node_coord = node_coord_dst
            batch = batch_dst
            if debug:
                print('down embedding', node_src.embedding.shape)
        if self.norm is not None:
            node_dst.embedding = self.norm(node_dst.embedding)

        s2_feat = node_dst.embedding
        proprio = torch.zeros_like(s2_feat[..., :4 * self.n_proprio])
        proprio[:, 1:4, :3 * self.n_proprio] = ee_rot_vec  # 3x type1 irrep
        proprio[:, 1:4, -self.n_proprio:] = ee_pose  # 1x type1 irrep
        proprio[:, 0, :2 * self.n_proprio] = ee_q  # 2x type2 irrep
        s2_feat = torch.cat([s2_feat, proprio], dim=-1)

        if not sanitycheck:
            s2_feat = einops.rearrange(s2_feat, 'b irrep c -> b (c irrep)', b=batch_size)
            return s2_feat
        else:
            ########## If debug, outputs irrp1 (vector) ###########
            vector = s2_feat.narrow(1, 1, 3)
            vector = einops.rearrange(vector, 'b d c -> b c d', b=batch_size)
            return vector

    def output_shape(self):
        return self.c_dim * self.n_cam + 4 * self.n_proprio  # plus 3 channel rot_mat and 1 channel position

    def _init_edge_rot_mat(self, edge_length_vec):
        # return init_edge_rot_mat(edge_length_vec)
        return init_edge_rot_mat2(edge_length_vec)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, m):
        if (isinstance(m, torch.nn.Linear)
                or isinstance(m, SO3_LinearV2)
        ):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            if self.weight_init == 'normal':
                std = 1 / math.sqrt(m.in_features)
                torch.nn.init.normal_(m.weight, 0, std)

        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    def _uniform_init_rad_func_linear_weights(self, m):
        if (isinstance(m, RadialFunction)):
            m.apply(self._uniform_init_linear_weights)

    def _uniform_init_linear_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            std = 1 / math.sqrt(m.in_features)
            torch.nn.init.uniform_(m.weight, -std, std)


if __name__ == "__main__":
    device = 'cuda:0'
    atol = 1e-3
    torch.manual_seed(0)
    np.random.seed(0)
    bs = 32
    T = 2
    pc = torch.rand(bs * T, 1024, 6) - 0.5
    pc = pc.to(device)
    euler = torch.rand(bs * T, 3) * 2 * torch.pi
    rot_mat = pytorch3d_transforms.euler_angles_to_matrix(euler, 'XYZ').to(device)
    ee_quat_xyzw = pytorch3d_transforms.matrix_to_quaternion(rot_mat)[:, [1, 2, 3, 0]]
    ee_pose = torch.rand(bs * T, 3) - 0.5
    ee_pose = ee_pose.to(device)
    ee_q = torch.ones(bs * T, 2).to(device)
    nobs = {"point_cloud": pc,  # [(b t) n_pts xyzrgb]
            "robot0_eef_quat": ee_quat_xyzw,  # [(b t) 4]
            "robot0_gripper_qpos": ee_q,  # [(b t) 2]
            "robot0_eef_pos": ee_pose}  # [(b t) 3]
    model = EquiFormerEnc(
        c_dim=64,
        lmax=1,
        mmax=1,
        deterministic=True,
        alpha_drop=0.,
        drop_path_rate=0.,
        proj_drop=0.,
        grid_resolution=12,
    ).to(device)
    c4_rots = torch.zeros((4, 3)).to(device)
    c4_rots[:, -1] = torch.arange(4) * np.pi / 2
    print("Vision params: %e" % model.num_params)

    out = model(nobs, sanitycheck=True)

    success = True
    for i in range(c4_rots.shape[0]):
        pc_tfm = pc.clone()
        pc_tfm[..., :3] = rot_pcd(pc[..., :3], c4_rots[i]).reshape(bs * T, -1, 3)
        rot_mat_tfm = rot_pcd(rot_mat.transpose(1, 2), c4_rots[i]).transpose(1, 2)
        ee_quat_xyzw_tfm = pytorch3d_transforms.matrix_to_quaternion(rot_mat_tfm)[:, [1, 2, 3, 0]]
        ee_pose_tfm = rot_pcd(ee_pose.unsqueeze(1), c4_rots[i]).reshape(bs * T, 3)
        out_feats_tfm_after = rot_pcd(out, c4_rots[i])

        nobs_tfm = {"point_cloud": pc_tfm,  # [(b t) n_pts xyzrgb]
                    "robot0_eef_quat": ee_quat_xyzw_tfm,  # [(b t) 4]
                    "robot0_gripper_qpos": ee_q,  # [(b t) 2]
                    "robot0_eef_pos": ee_pose_tfm}  # [(b t) 3]
        out_feats_tfm_before = model(nobs_tfm, sanitycheck=True)

        eerr = torch.linalg.norm(out_feats_tfm_before - out_feats_tfm_after, dim=1).max()
        err = torch.linalg.norm(out_feats_tfm_after - out, dim=1).max()
        if not torch.allclose(out_feats_tfm_before, out_feats_tfm_after, atol=atol):
            print(f"FAILED on {c4_rots[i]}: {eerr:.1E} > {atol}, {err}")
            # print(out_feats_tfm_after - out_feats_tfm_after)
            success = False
        else:
            print(f"PASSED on {c4_rots[i]}: {eerr:.1E} < {atol}, {err}")

    # import matplotlib.pyplot as plt
    # f = plt.figure(figsize=(16, 4))
    # ax = [f.add_subplot(1, 4, i+1, projection='3d') for i in range(4)]
    # plt.show()

    if success:
        print('PASSED')
    print(1)
