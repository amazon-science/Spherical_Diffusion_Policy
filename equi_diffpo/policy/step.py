from typing import Dict
import torch
import torch.nn.functional as F
from einops import reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from termcolor import cprint
import copy
import pytorch3d.transforms as pytorch3d_transforms

from equi_diffpo.model.common.module_attr_mixin import ModuleAttrMixin
from equi_diffpo.model.common.normalizer import LinearNormalizer
from equi_diffpo.model.equivariant_diffusion.non_equ_irrep_conditional_unet1d import NonEquIrrepConditionalUnet1D
from equi_diffpo.model.equivariant_diffusion.reg_rep_conditional_unet1d import RegRepConditionalUnet1D
from equi_diffpo.model.vision.pc_rot_randomizer import PointCloudRotRandomizer
from equi_diffpo.model.diffusion.mask_generator import LowdimMaskGenerator
from equi_diffpo.common.pytorch_util import dict_apply
from equi_diffpo.model.equivariant_vision.equiformer_enc import EquiFormerEnc
from equi_diffpo.model.equivariant_diffusion.irreps_conditional_unet1d import IrrepConditionalUnet1D
from equi_diffpo.model.vision.pc_so3_randomizer import PointCloudSO3RotRandomizer


class BasePolicy(ModuleAttrMixin):
    # init accepts keyword argument shape_meta, see config/task/*_image.yaml

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict:
            str: B,To,*
        return: B,Ta,Da
        """
        raise NotImplementedError()

    # reset state for stateful policies
    def reset(self):
        pass

    # ========== training ===========
    # no standard training interface except setting normalizer
    def set_normalizer(self, normalizer: LinearNormalizer):
        raise NotImplementedError()


class STEP(BasePolicy):
    def __init__(self,
                 shape_meta: dict,
                 noise_scheduler: DDPMScheduler,
                 horizon,
                 n_action_steps,
                 n_obs_steps,
                 num_inference_steps=None,
                 obs_as_global_cond=True,
                 diffusion_step_embed_dim=128,
                 down_dims=(256, 512, 1024),
                 kernel_size=5,
                 n_groups=8,
                 condition_type="film",
                 lmax=2,
                 mmax=2,
                 encoder_output_dim=128,
                 canonicalize=True,
                 use_pc_color=True,
                 pointnet_type="encoder",
                 v_grid_resolution=12,
                 d_grid_resolution=14,
                 max_neighbors=(16, 16, 16, 16),
                 max_radius=(0.05, 0.2, 0.8, 3),
                 pool_ratio=(0.25, 0.25, 0.25),
                 sphere_channels=(32, 64, 128),
                 attn_hidden_channels=(32, 64, 128, 256),
                 attn_alpha_channels=(8, 16, 32, 64),
                 attn_value_channels=(4, 8, 16, 32),
                 ffn_hidden_channels=(32, 64, 128, 128),
                 edge_channels=(16, 32, 64, 128),
                 num_distance_basis=(64, 64, 64, 64),
                 denoise_nn='irrep',
                 rot_aug=(0, 0, 180),
                 rad_aug=0,
                 pcd_noise=0,
                 norm=True,
                 FiLM_type='SFiLM',
                 pool_method='fpsknn',
                 # parameters passed to step
                 **kwargs):
        super().__init__()

        self.condition_type = condition_type

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2:  # use multiple hands
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")

        self.canonicalize = canonicalize
        obs_encoder = EquiFormerEnc(c_dim=encoder_output_dim,
                                    lmax=lmax,
                                    mmax=mmax,
                                    norm=norm,
                                    pcd_noise=pcd_noise,
                                    max_neighbors=max_neighbors,
                                    max_radius=max_radius,
                                    pool_ratio=pool_ratio,
                                    sphere_channels=sphere_channels,
                                    attn_hidden_channels=attn_hidden_channels,
                                    attn_alpha_channels=attn_alpha_channels,
                                    attn_value_channels=attn_value_channels,
                                    ffn_hidden_channels=ffn_hidden_channels,
                                    edge_channels=edge_channels,
                                    num_distance_basis=num_distance_basis,
                                    grid_resolution=v_grid_resolution,
                                    pool_method=pool_method,
                                    n_cam=1,  # number of time step of camera info
                                    n_proprio=n_obs_steps)  # number of proprio info

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        cprint(f"[STEPPolicy] use_pc_color: {self.use_pc_color}", "yellow")
        cprint(f"[STEPPolicy] pointnet_type: {self.pointnet_type}", "yellow")

        if denoise_nn == 'irrep':
            model = IrrepConditionalUnet1D(
                input_dim=input_dim // 3,
                max_lmax=lmax,
                norm=norm,
                local_cond_dim=None,
                global_cond_dim=global_cond_dim,
                diffusion_step_embed_dim=diffusion_step_embed_dim,
                down_dims=down_dims,
                kernel_size=kernel_size,
                n_groups=n_groups,
                FiLM_type=FiLM_type,
                grid_resolution=d_grid_resolution
            )
        elif denoise_nn == 'reg':
            model = RegRepConditionalUnet1D(
                input_dim=input_dim // 3,
                max_lmax=lmax,
                norm=norm,
                local_cond_dim=None,
                global_cond_dim=global_cond_dim,
                diffusion_step_embed_dim=diffusion_step_embed_dim,
                down_dims=down_dims,
                kernel_size=kernel_size,
                n_groups=n_groups,
                grid_resolution=d_grid_resolution
            )
        elif denoise_nn == 'nn':
            model = NonEquIrrepConditionalUnet1D(
                input_dim=input_dim,
                max_lmax=lmax,
                local_cond_dim=None,
                global_cond_dim=global_cond_dim,
                diffusion_step_embed_dim=diffusion_step_embed_dim,
                down_dims=down_dims,
                kernel_size=kernel_size,
                n_groups=n_groups
            )
        else:
            raise NotImplementedError(denoise_nn)

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler

        self.noise_scheduler_pc = copy.deepcopy(noise_scheduler)
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )

        self.normalizer = LinearNormalizer()
        self.rot_aug = rot_aug
        if self.rot_aug is not None:
            self.rot_randomizer = PointCloudSO3RotRandomizer(self.rot_aug)
        self.rad_aug = rad_aug
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
        print("Vision params: %e" % sum(p.numel() for p in self.obs_encoder.parameters()))

    # ========= inference  ============
    def conditional_sample(self,
                           condition_data, condition_mask,
                           condition_data_pc=None, condition_mask_pc=None,
                           local_cond=None, global_cond=None,
                           generator=None,
                           # keyword arguments to scheduler.step
                           **kwargs
                           ):
        model = self.model
        scheduler = self.noise_scheduler
        trajectories = []
        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device)
        trajectories.append(trajectory.clone())
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            model_output = model(sample=trajectory,
                                 timestep=t,
                                 local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, ).prev_sample

            trajectories.append(trajectory.clone())

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory, trajectories

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        if 'robot0_eye_in_hand_image' in obs_dict:
            del obs_dict['robot0_eye_in_hand_image']
        if 'agentview_image' in obs_dict:
            del obs_dict['agentview_image']
        obs_dict = copy.deepcopy(obs_dict)
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        # this_n_point_cloud = nobs['imagin_robot'][..., :3] # only use coordinate
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        # canonicalize
        if self.canonicalize:
            ee_pos_in_ws = nobs['robot0_eef_pos'][:, -1:].clone()
            nobs['point_cloud'][:, :, :, :3] -= nobs['robot0_eef_pos'][:, None, -1:]
            nobs['robot0_eef_pos'][:, -1:] -= nobs['robot0_eef_pos'][:, -1:]

        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            if "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(B, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da + Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs_features
            cond_mask[:, :To, Da:] = True

        # run sampling
        nsample, trajectories = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)

        naction_pred = nsample[..., :Da]
        # uncanonicalize
        if self.canonicalize:
            naction_pred[..., :3] += ee_pos_in_ws

        # idx = 0
        # for i in range(len(trajectories)):
        #     traj_color = torch.ones_like(trajectories[i][idx, :, :3]).cpu()
        #     traj_color[:, 1] = 1 - torch.arange(traj_color.shape[0]) / traj_color.shape[0]
        #     traj_color[:, 2] = 1 - torch.arange(traj_color.shape[0]) / traj_color.shape[0]
        #     trajectories[i] = torch.cat([trajectories[i][idx, :, :3].cpu(), traj_color], dim=1)
        # import open3d as o3d
        # pc = nobs['point_cloud'][idx, 0]
        # vis_pcd = o3d.geometry.PointCloud()
        # # Assign points and colors to the PointCloud
        # vis_pcd.points = o3d.utility.Vector3dVector(pc[:, :3].detach().cpu().numpy())
        # vis_pcd.colors = o3d.utility.Vector3dVector(pc[:, 3:].detach().cpu().numpy())
        # from equi_diffpo.common.animate_pcd import animate_point_clouds
        # animate_point_clouds(trajectories, vis_pcd, t=0.1, view=[1, 1, 1], zoom=0.5)

        # #--------------------------------------------------
        # import open3d as o3d
        # import numpy as np
        #
        # idx = 8
        # k = 99
        # traj_color = torch.ones_like(trajectories[k][idx, :, :3]).cpu()
        # # traj_color[:, 0] = torch.arange(traj_color.shape[0]) / traj_color.shape[0]
        # traj_color[:, 0] = 0.6 * (1 - torch.arange(traj_color.shape[0]) / traj_color.shape[0]) + 0.1
        # traj_color[:, 1] = 0.6 * (1 - torch.arange(traj_color.shape[0]) / traj_color.shape[0]) + 0.1
        # traj = torch.cat([trajectories[k][idx, :, :3].cpu(), traj_color], dim=1)
        #
        # pc = nobs['point_cloud'][idx, 0]
        # vis_pcd = o3d.geometry.PointCloud()
        # # Assign points and colors to the PointCloud
        # vis_pcd.points = o3d.utility.Vector3dVector(pc[:, :3].detach().cpu().numpy())
        # # vis_pcd.colors = o3d.utility.Vector3dVector((pc[:, 3:].detach().cpu().numpy() + 1) / 8 + 0.75)
        # vis_pcd.colors = o3d.utility.Vector3dVector((pc[:, 3:].detach().cpu().numpy() + 1) / 2)
        #
        # vis_traj = o3d.geometry.PointCloud()
        # # Assign points and colors to the PointCloud
        # vis_traj.points = o3d.utility.Vector3dVector(traj[:, :3].detach().cpu().numpy())
        # vis_traj.colors = o3d.utility.Vector3dVector(traj[:, 3:].detach().cpu().numpy())
        #
        # rot1 = np.asarray([-1.57, 0., 0]).reshape(-1, 1)
        # rot1 = vis_traj.get_rotation_matrix_from_axis_angle(rot1)
        # rot2 = np.asarray([0., -2.3, 0]).reshape(-1, 1)
        # # rot2 = np.asarray([0., -2, 0]).reshape(-1, 1)
        # rot2 = vis_traj.get_rotation_matrix_from_axis_angle(rot2)
        # rot3 = np.asarray([0.1, 0, 0]).reshape(-1, 1)
        # # rot3 = np.asarray([0.2, 0, 0]).reshape(-1, 1)
        # rot3 = vis_traj.get_rotation_matrix_from_axis_angle(rot3)
        # rot = rot3 @ rot2 @ rot1
        # vis_traj.rotate(rot, center=(0, 0, 0))
        # vis_pcd.rotate(rot, center=(0, 0, 0))
        #
        # o3d.visualization.draw_geometries([vis_traj, vis_pcd])
        # o3d.visualization.draw_geometries([vis_pcd])

        # Converting XY- axes of the end-effector to 6D representation
        rot_mat = pytorch3d_transforms.rotation_6d_to_matrix(naction_pred[..., 3:9].reshape(-1, 6))
        rot_xy = rot_mat.transpose(2, 1)[:, :2, :]
        rot_xy = rot_xy.reshape(B, T, 6)
        naction_pred = torch.cat((naction_pred[..., :3], rot_xy, naction_pred[..., 9:]), dim=-1)

        # unnormalize prediction
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        # get prediction

        result = {
            'action': action,
            'action_pred': action_pred,
        }

        # import pytorch3d.transforms as pytorch3d_transforms
        # import open3d as o3d
        # idx = 9
        # obs = obs_dict
        # pc = obs["point_cloud"][idx, 0]
        # ee_quat_xyzw = obs["robot0_eef_quat"][idx, 0]
        # ee_q = obs["robot0_gripper_qpos"][idx, 0]
        # ee_pose = obs["robot0_eef_pos"][idx, 0]
        # ee_rot_mat = pytorch3d_transforms.quaternion_to_matrix(ee_quat_xyzw[[3, 0, 1, 2]])  # [(b t) 3 3]
        # vis_pcd = o3d.geometry.PointCloud()
        # # Assign points and colors to the PointCloud
        # vis_pcd.points = o3d.utility.Vector3dVector(pc[:, :3].detach().cpu().numpy())
        # vis_pcd.colors = o3d.utility.Vector3dVector(pc[:, 3:].detach().cpu().numpy())
        # # Visualize the expert pose
        # expert_pose = ee_pose.unsqueeze(0).clone().repeat(6, 1)
        # r = 0.1
        # expert_pose[:3] += r * ee_rot_mat.permute(1, 0)
        # expert_pose[3:] -= r * ee_rot_mat.permute(1, 0)
        # expert_edge = torch.arange(6).reshape((2, 3)).permute(1, 0)
        # expert_color = torch.zeros((3, 3))
        # expert_color[torch.arange(3), torch.arange(3)] = 1
        # expert_coord = o3d.geometry.LineSet()
        # expert_coord.points = o3d.utility.Vector3dVector(expert_pose.detach().cpu().numpy())
        # expert_coord.lines = o3d.utility.Vector2iVector(expert_edge.detach().cpu().numpy())
        # expert_coord.colors = o3d.utility.Vector3dVector(expert_color.detach().cpu().numpy())
        # # Visualize the expert action
        # a_pose = action_pred[idx, :, :3].clone()  # 16 3
        # c_pose = torch.zeros_like(a_pose).cpu()
        # c_pose[:, 0] = 1 - torch.arange(a_pose.shape[0]) / a_pose.shape[0]
        # vis_a = o3d.geometry.PointCloud()
        # vis_a.points = o3d.utility.Vector3dVector(a_pose.detach().cpu().numpy())
        # vis_a.colors = o3d.utility.Vector3dVector(c_pose.detach().cpu().numpy())
        # # Visualize the point cloud
        # o3d.visualization.draw_geometries([vis_pcd, expert_coord, vis_a])

        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        if 'robot0_eye_in_hand_image' in batch['obs']:
            del batch['obs']['robot0_eye_in_hand_image']

        # normalize input
        batch = copy.deepcopy(batch)
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        bs, na = nactions.shape[0], nactions.shape[1]

        # canonicalize
        if self.canonicalize:
            nactions[:, :, :3] -= nobs['robot0_eef_pos'][:, -1:]
            nobs['point_cloud'][:, :, :, :3] -= nobs['robot0_eef_pos'][:, None, -1:]
            nobs['robot0_eef_pos'][:, -1:] -= nobs['robot0_eef_pos'][:, -1:]

        if self.rot_aug:
            nobs, nactions = self.rot_randomizer(nobs, nactions)

        if self.rad_aug > 0:
            for tries in range(100):
                trans_noise = torch.rand_like(nobs['robot0_eef_pos'][:, None, -1:].repeat(4, 1, 1, 1))
                trans_noise -= 0.5
                trans_noise *= 2 * self.rad_aug
                insider = trans_noise.norm(dim=-1, p=2) < self.rad_aug
                trans_noise = trans_noise[insider[:, 0, 0]]
                if trans_noise.shape[0] > nobs['robot0_eef_pos'].shape[0]:
                    trans_noise = trans_noise[:nobs['robot0_eef_pos'].shape[0]]
                    break
            if tries == 99:
                raise 'Can not find valid RAD aug noise, tried={} times'.format(tries)
            nobs['point_cloud'][:, :, :, :3] += trans_noise

        # Converting 6D representation to XY- axes of the end-effector so that it is compatible to SO(3) equ
        rot_mat = pytorch3d_transforms.rotation_6d_to_matrix(nactions[:, :, 3:9].reshape(-1, 6))
        rot_xy = rot_mat.transpose(2, 1)[:, :2, :].reshape(bs, na, 6)
        nactions = torch.cat((nactions[:, :, :3], rot_xy, nactions[:, :, 9:]), dim=-1)

        # # strangely, action are not aligned with ee_pose, fixing it manually
        # d = 2 * nactions[:, 0:1, :3] - nactions[:, 1:2, :3] - nobs["robot0_eef_pos"][:, 1:2]
        # nactions = nactions.clone()
        # nactions[..., :3] = nactions[..., :3] - d

        # import pytorch3d.transforms as pytorch3d_transforms
        # import open3d as o3d
        # idx = 1
        # obs = nobs
        # pc = obs["point_cloud"][idx, 1]
        # ee_quat_xyzw = obs["robot0_eef_quat"][idx, 1]
        # ee_q = obs["robot0_gripper_qpos"][idx, 1]
        # ee_pose = obs["robot0_eef_pos"][idx, 1]
        # ee_rot_mat = pytorch3d_transforms.quaternion_to_matrix(ee_quat_xyzw[[3, 0, 1, 2]])  # [(b t) 3 3]
        # vis_pcd = o3d.geometry.PointCloud()
        # # Assign points and colors to the PointCloud
        # vis_pcd.points = o3d.utility.Vector3dVector(pc[:, :3].detach().cpu().numpy())
        # vis_pcd.colors = o3d.utility.Vector3dVector((pc[:, 3:].detach().cpu().numpy() + 1) / 2)
        # # Visualize the proprio pose
        # expert_pose = ee_pose.unsqueeze(0).clone().repeat(6, 1)
        # r = 0.1
        # expert_pose[:3] += r * ee_rot_mat.permute(1, 0)
        # expert_pose[3:] -= r * ee_rot_mat.permute(1, 0)
        # expert_edge = torch.arange(6).reshape((2, 3)).permute(1, 0)
        # expert_color = torch.zeros((3, 3))
        # expert_color[torch.arange(3), torch.arange(3)] = 1
        # expert_coord = o3d.geometry.LineSet()
        # expert_coord.points = o3d.utility.Vector3dVector(expert_pose.detach().cpu().numpy())
        # expert_coord.lines = o3d.utility.Vector2iVector(expert_edge.detach().cpu().numpy())
        # expert_coord.colors = o3d.utility.Vector3dVector(expert_color.detach().cpu().numpy())
        # # Visualize the expert action
        # ee_a = nactions[idx, :, :-1]
        # a_pose = ee_a[:, :3].clone()  # 16 3
        # c_pose = torch.zeros_like(a_pose)
        # c_pose[:, 1] = 0.7 - 0.5 * torch.arange(a_pose.shape[0]) / a_pose.shape[0]
        # c_pose[:, 2] = 0.7 - 0.5 * torch.arange(a_pose.shape[0]) / a_pose.shape[0]
        # # c_pose[:, 0] = 0.7 - 0.5 * torch.arange(a_pose.shape[0]) / a_pose.shape[0]
        # c_pose[:, 0] = 1
        # vis_a = o3d.geometry.PointCloud()
        # vis_a.points = o3d.utility.Vector3dVector(a_pose.detach().cpu().numpy())
        # vis_a.colors = o3d.utility.Vector3dVector(c_pose.detach().cpu().numpy())
        #
        # # expert action rot
        # a_rot = ee_a[0, 3:9].clone()  # 6
        # ee_rot_mat = pytorch3d_transforms.rotation_6d_to_matrix(a_rot)  # [(b t) 3 3]
        # r = 0.1
        # a_rot_pose = ee_a[0:1, :3].clone().repeat(6, 1)  # 6 3
        # a_rot_pose[:3] += r * ee_rot_mat.permute(1, 0)
        # a_rot_pose[3:] -= r * ee_rot_mat.permute(1, 0)
        # a_rot_edge = torch.arange(6).reshape((2, 3)).permute(1, 0)
        # a_rot_color = torch.zeros((3, 3))
        # a_rot_color[torch.arange(3), torch.arange(3)] = 1
        # a_rot_coord = o3d.geometry.LineSet()
        # a_rot_coord.points = o3d.utility.Vector3dVector(a_rot_pose.detach().cpu().numpy())
        # a_rot_coord.lines = o3d.utility.Vector2iVector(a_rot_edge.detach().cpu().numpy())
        # a_rot_coord.colors = o3d.utility.Vector3dVector(a_rot_color.detach().cpu().numpy())
        #
        # # Visualize the predicted action
        # pred_a = pred[idx, :, :-1]
        # a_pose = pred_a[:, :3].clone()  # 16 3
        # c_pose = torch.zeros_like(a_pose)
        # c_pose[:, 1] = 0.7 - 0.5 * torch.arange(a_pose.shape[0]) / a_pose.shape[0]
        # c_pose[:, 0] = 0.7 - 0.5 * torch.arange(a_pose.shape[0]) / a_pose.shape[0]
        # # c_pose[:, 2] = 0.7 - 0.5 * torch.arange(a_pose.shape[0]) / a_pose.shape[0]
        # c_pose[:, 2] = 1
        # vis_pred = o3d.geometry.PointCloud()
        # vis_pred.points = o3d.utility.Vector3dVector(a_pose.detach().cpu().numpy())
        # vis_pred.colors = o3d.utility.Vector3dVector(c_pose.detach().cpu().numpy())
        # # Visualize the point cloud
        # o3d.visualization.draw_geometries([vis_pcd, expert_coord, vis_a, vis_pred, a_rot_coord])

        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]

        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory

        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs,
                                   lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)

            if "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(batch_size, -1)
            # this_n_point_cloud = this_nobs['point_cloud'].reshape(batch_size, -1, *this_nobs['point_cloud'].shape[1:])
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)

        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (bsz,), device=trajectory.device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)

        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        # Predict the noise residual
        pred = self.model(sample=noisy_trajectory,
                          timestep=timesteps,
                          local_cond=local_cond,
                          global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        elif pred_type == 'v_prediction':
            # https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
            # https://github.com/huggingface/diffusers/blob/v0.11.1-patch/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
            # sigma = self.noise_scheduler.sigmas[timesteps]
            # alpha_t, sigma_t = self.noise_scheduler._sigma_to_alpha_sigma_t(sigma)
            self.noise_scheduler.alpha_t = self.noise_scheduler.alpha_t.to(self.device)
            self.noise_scheduler.sigma_t = self.noise_scheduler.sigma_t.to(self.device)
            alpha_t, sigma_t = self.noise_scheduler.alpha_t[timesteps], self.noise_scheduler.sigma_t[timesteps]
            alpha_t = alpha_t.unsqueeze(-1).unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1).unsqueeze(-1)
            v_t = alpha_t * noise - sigma_t * trajectory
            target = v_t
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()

        loss_dict = {
            'bc_loss': loss.item(),
        }

        return loss, loss_dict
