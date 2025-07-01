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
from equi_diffpo.model.vision.pc_rot_randomizer import PointCloudRotRandomizer
from equi_diffpo.model.diffusion.mask_generator import LowdimMaskGenerator
from equi_diffpo.common.pytorch_util import dict_apply

from equi_diffpo.model.equivariant_vision.sim3_encoder import SIM3Vec4Latent
from equi_diffpo.model.equivariant_diffusion.vn_conditional_unet1d import VecConditionalUnet1D


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


class EquiBot(BasePolicy):
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
                 condition_type="film",
                 canonicalize=True,
                 use_pc_color=False,
                 pointnet_type="encoder",
                 rot_aug=True,
                 rad_aug=0,
                 model_cfg=None,
                 **kwargs):
        super().__init__()
        model_cfg.scale_cano = model_cfg.mean_cano = not canonicalize
        self.condition_type = condition_type

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        self.num_eef = len(action_shape)
        self.canonicalize = canonicalize
        obs_encoder = SIM3Vec4Latent(**model_cfg)
        self.encoder_out_dim = model_cfg.c_dim
        self.action_dim = 3 * self.num_eef  # action_dim consists of 4 3D vectors, xyz + x-y-z- axes of each eef

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        cprint(f"[EquibotPolicy] use_pc_color: {self.use_pc_color}", "yellow")
        cprint(f"[EquibotPolicy] pointnet_type: VN-Pointnet", "yellow")

        self.obs_dim = self.encoder_out_dim + self.action_dim
        model = VecConditionalUnet1D(
            input_dim=self.action_dim,
            cond_dim=self.obs_dim * n_obs_steps,
            scalar_cond_dim=self.num_eef * n_obs_steps,
            scalar_input_dim=self.num_eef,
            down_dims=down_dims,
            kernel_size=kernel_size,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            cond_predict_scale=True,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler

        self.noise_scheduler_pc = copy.deepcopy(noise_scheduler)
        # self.mask_generator = LowdimMaskGenerator(
        #     action_dim=action_dim,
        #     obs_dim=0,
        #     max_n_obs_steps=n_obs_steps,
        #     fix_obs_steps=True,
        #     action_visible=False
        # )

        self.normalizer = LinearNormalizer()
        self.rot_aug = rot_aug
        if self.rot_aug:
            self.rot_randomizer = PointCloudRotRandomizer()
        self.rad_aug = rad_aug
        self.horizon = horizon
        # self.obs_feature_dim = obs_feature_dim
        # self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
        print("Vision params: %e" % sum(p.numel() for p in self.obs_encoder.parameters()))

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

        # canonicalize
        if self.canonicalize:
            ee_pos_in_ws = nobs['robot0_eef_pos'][:, -1:].clone()
            nobs['point_cloud'][:, :, :, :3] -= nobs['robot0_eef_pos'][:, None, -1:]
            nobs['robot0_eef_pos'][:, -1:] -= nobs['robot0_eef_pos'][:, -1:]

        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        To = self.n_obs_steps

        # condition through global feature
        this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
        this_nobs['point_cloud'], rgb = this_nobs['point_cloud'][..., :3], this_nobs['point_cloud'][..., 3:]
        if not self.use_pc_color:
            rgb = None
        feat_dict = self.obs_encoder(this_nobs['point_cloud'], rgb=rgb)
        center = (
            feat_dict["center"].reshape(B, To, 1, 3)[:, [-1]].repeat(1, To, 1, 1)
        )
        scale = feat_dict["scale"].reshape(B, To, 1, 1)[:, [-1]].repeat(1, To, 1, 1)

        # z_pos: (B, H, ?, 3) [need norm] + z_dir: (B, H, ?, 3) [does not need norm] + z_scalar (B, H, E)
        ee_quat_xyzw = nobs["robot0_eef_quat"].reshape(-1, 4)
        z_scalar = nobs["robot0_gripper_qpos"].clone()  # [b t 2]
        z_scalar = z_scalar[:, 0] - z_scalar[:, 1]
        z_pos = nobs["robot0_eef_pos"].unsqueeze(2)  # [b t 1 3]
        # batch_size, num_points, _ = pc.shape  # (b t) n d
        # xyzrgb = rearrange(pc, "b npts d -> (b npts) d")
        ee_rot_mat = pytorch3d_transforms.quaternion_to_matrix(ee_quat_xyzw[:, (3, 0, 1, 2)])  # [(b t) 3 3]
        z_dir = ee_rot_mat.permute(0, 2, 1)[..., :2, :].reshape(B, To, 2, 3)
        z_pos = (z_pos - center) / scale
        z = feat_dict["so3"]
        z = z.reshape(B, To, -1, 3)
        z = torch.cat([z, z_pos, z_dir], dim=-2)  # Bs To condition_dim 3
        obs_cond_vec, obs_cond_scalar = z.reshape(B, -1, 3), (
            z_scalar.reshape(B, -1) if z_scalar is not None else None
        )

        initial_noise_scale = 1.0
        noisy_action = (
            torch.randn((B, T, self.action_dim, 3)).to(self.device)
            * initial_noise_scale,
            torch.randn((B, T, self.num_eef)).to(self.device)
            * initial_noise_scale,
        )

        # set step values
        curr_action = noisy_action
        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        for k in self.noise_scheduler.timesteps:
            # load from existing data statistics
            # predict noise
            noise_pred = self.model(
                sample=curr_action[0],
                timestep=k,
                scalar_sample=curr_action[1],
                cond=obs_cond_vec,
                scalar_cond=obs_cond_scalar,
            )

            # inverse diffusion step
            new_action = [None, None]
            new_action[0] = self.noise_scheduler.step(
                model_output=noise_pred[0], timestep=k, sample=curr_action[0]
            ).prev_sample
            if noise_pred[1] is not None:
                new_action[1] = self.noise_scheduler.step(
                    model_output=noise_pred[1], timestep=k, sample=curr_action[1]
                ).prev_sample
            curr_action = tuple(new_action)

        center = (
            feat_dict["center"]
            .reshape(B, To, 3)[:, [-1], None]
            .repeat(1, T, 1, 1)
        )
        scale = (
            feat_dict["scale"].reshape(B, To, 1)[:, [-1], None].repeat(1, T, 1, 1)
        )
        a_pos = new_action[0][..., 0, :] * scale.squeeze(2) + center.squeeze(2)  # B T 3
        a_XY_rot = new_action[0][..., 1:, :].reshape(B, T, 6)  # B T 6
        a_grip = new_action[1]

        # Converting XY- axes of the end-effector to 6D representation
        rot_mat = pytorch3d_transforms.rotation_6d_to_matrix(a_XY_rot.reshape(-1, 6))
        rot_xy = rot_mat.transpose(2, 1)[:, :2, :]
        rot_xy = rot_xy.reshape(B, T, 6)
        naction_pred = torch.cat((a_pos, rot_xy, a_grip), dim=-1)

        if self.canonicalize:
            naction_pred[..., :3] += ee_pos_in_ws

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
        To = self.n_obs_steps

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
                trans_noise *= 2*self.rad_aug
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
        # ee_a = target[idx, :, :-1]
        # a_pose = ee_a[:, :3].clone()  # 16 3
        # c_pose = torch.zeros_like(a_pose)
        # c_pose[:, 1] = 0.7 - 0.5 * torch.arange(a_pose.shape[0]) / a_pose.shape[0]
        # c_pose[:, 2] = 0.7 - 0.5 * torch.arange(a_pose.shape[0]) / a_pose.shape[0]
        # # c_pose[:, 0] = 0.7 - 0.5 * torch.arange(a_pose.shape[0]) / a_pose.shape[0]
        # c_pose[:, 0] = 1
        # vis_a = o3d.geometry.PointCloud()
        # vis_a.points = o3d.utility.Vector3dVector(a_pose.detach().cpu().numpy())
        # vis_a.colors = o3d.utility.Vector3dVector(c_pose.detach().cpu().numpy())
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
        # o3d.visualization.draw_geometries([vis_pcd, expert_coord, vis_a, vis_pred])

        # condition through global feature
        this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
        this_nobs['point_cloud'], rgb = this_nobs['point_cloud'][..., :3], this_nobs['point_cloud'][..., 3:]
        if not self.use_pc_color:
            rgb = None
        feat_dict = self.obs_encoder(this_nobs['point_cloud'], rgb=rgb)
        center = (
            feat_dict["center"].reshape(bs, To, 1, 3)[:, [-1]].repeat(1, To, 1, 1)
        )
        scale = feat_dict["scale"].reshape(bs, To, 1, 1)[:, [-1]].repeat(1, To, 1, 1)

        # z_pos: (bs, H, ?, 3) [need norm] + z_dir: (bs, H, ?, 3) [does not need norm] + z_scalar (bs, H, E)
        ee_quat_xyzw = nobs["robot0_eef_quat"].reshape(-1, 4)
        z_scalar = nobs["robot0_gripper_qpos"].clone()  # [b t 2]
        z_scalar = z_scalar[:, 0] - z_scalar[:, 1]
        z_pos = nobs["robot0_eef_pos"].unsqueeze(2)  # [b t 1 3]
        # batch_size, num_points, _ = pc.shape  # (b t) n d
        # xyzrgb = rearrange(pc, "b npts d -> (b npts) d")
        ee_rot_mat = pytorch3d_transforms.quaternion_to_matrix(ee_quat_xyzw[:, (3, 0, 1, 2)])  # [(b t) 3 3]
        z_dir = ee_rot_mat.permute(0, 2, 1)[..., :2, :].reshape(bs, To, 2, 3)
        z_pos = (z_pos - center) / scale
        z = feat_dict["so3"]
        z = z.reshape(bs, To, -1, 3)
        z = torch.cat([z, z_pos, z_dir], dim=-2)  # Bs To condition_dim 3
        obs_cond_vec, obs_cond_scalar = z.reshape(bs, -1, 3), (
            z_scalar.reshape(bs, -1) if z_scalar is not None else None
        )

        # handle different ways of passing observation
        center = (
            feat_dict["center"]
            .reshape(bs, To, 3)[:, [-1]]
            .repeat(1, na, 1)
        )
        scale = feat_dict["scale"].reshape(bs, To, 1)[:, [-1]].repeat(1, na, 1)
        trajectory = torch.cat([(nactions[..., :3] - center) / scale, nactions[..., 3:]], dim=-1)

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

        # Predict the noise residual
        noisy_eef_actions = noisy_trajectory[..., :9].reshape(bs, na, self.action_dim, 3)
        noisy_gripper_actions = noisy_trajectory[..., -1].reshape(bs, na, self.num_eef)
        vec_eef_noise_pred, vec_gripper_noise_pred = self.model(
            sample=noisy_eef_actions,
            timestep=timesteps,
            scalar_sample=noisy_gripper_actions,
            cond=obs_cond_vec,
            scalar_cond=obs_cond_scalar,
        )
        pred = torch.cat([vec_eef_noise_pred.reshape(bs, na, -1), vec_gripper_noise_pred], dim=-1)
        # pred = self.model(sample=noisy_trajectory,
        #                   timestep=timesteps,
        #                   local_cond=local_cond,
        #                   global_cond=global_cond)

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
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()

        loss_dict = {
            'bc_loss': loss.item(),
        }

        return loss, loss_dict
