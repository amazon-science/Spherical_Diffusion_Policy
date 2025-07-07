import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import random
import numpy as np
from einops import rearrange, repeat
import math
from copy import deepcopy

from sdp.model.common.rotation_transformer import RotationTransformer

class PointCloudSO3RotRandomizer(nn.Module):
    """
    Continuously and randomly rotate the input tensor during training.
    Does not rotate the tensor during evaluation.
    """
    
    def __init__(self, aug_angles=[0, 0, 180]):
        """
        Args:
            min_angle (float): Minimum rotation angle.
            max_angle (float): Maximum rotation angle.
        """
        super().__init__()

        aug_angles = torch.tensor(aug_angles) / 180 * math.pi
        self.register_buffer('aug_angles', aug_angles.unsqueeze(0))
        self.quat2mat = RotationTransformer('quaternion', 'matrix')
        self.euler2mat = RotationTransformer('euler_angles', 'matrix', from_convention='XYZ')
        self.mat26D = RotationTransformer('matrix', 'rotation_6d')
    
    def forward(self, nobs, naction: torch.Tensor):
        """
        Randomly rotates the inputs if in training mode.
        Keeps inputs unchanged if in evaluation mode.

        Args:
            inputs (torch.Tensor): input tensors

        Returns:
            torch.Tensor: rotated or unrotated tensors based on the mode
        """
        if self.training:
            pc = nobs["point_cloud"]
            pos = nobs["robot0_eef_pos"]
            # x, y, z, w -> w, x, y, z
            quat = nobs["robot0_eef_quat"][:, :, [3, 0, 1, 2]]
            batch_size = pc.shape[0]
            T = pc.shape[1]
            C = pc.shape[2]
            Ta = naction.shape[1]
            max_tries = 99

            for i in range(max_tries):
                angles = torch.rand(batch_size, 3).to(pc.device) - 0.5
                angles[torch.rand(batch_size) < 1/64, :] = 0
                angles *= self.aug_angles
                rotation_matrix = self.euler2mat.forward(angles)

                rotated_apos = naction[:, :, 0:3].clone()
                rotated_apos = (rotation_matrix @ rotated_apos.permute(0, 2, 1)).permute(0, 2, 1)
                rotated_arot_mat = self.mat26D.inverse(naction[:, :, 3:9].reshape(-1, 6).clone())
                rotated_arot_mat = torch.bmm(rotation_matrix.repeat_interleave(Ta, 0), rotated_arot_mat)
                rotated_naction = naction.clone()
                rotated_naction[:, :, 0:3] = rotated_apos
                rotated_naction[:, :, 3:9] = self.mat26D.forward(rotated_arot_mat).reshape(batch_size, Ta, 6)

                # rotated_naction[:, :, [3, 6]] = (rotation_matrix[:, :2, :2] @ naction[:, :, [3, 6]].permute(0, 2, 1)).permute(0, 2, 1)
                # rotated_naction[:, :, [4, 7]] = (rotation_matrix[:, :2, :2] @ naction[:, :, [4, 7]].permute(0, 2, 1)).permute(0, 2, 1)
                # rotated_naction[:, :, [5, 8]] = (rotation_matrix[:, :2, :2] @ naction[:, :, [5, 8]].permute(0, 2, 1)).permute(0, 2, 1)

                rotated_pos = (rotation_matrix @ pos.permute(0, 2, 1)).permute(0, 2, 1)
                rot = self.quat2mat.forward(quat)
                rotated_rot = rotation_matrix.unsqueeze(1) @ rot
                rotated_quat = self.quat2mat.inverse(rotated_rot)

                if rotated_pos.min() >= -1 and rotated_pos.max() <= 1 and rotated_naction[:, :, :2].min() >= -1 and rotated_naction[:, :, :2].max() <= 1:
                    break
            if i == max_tries:
                print('warning: rotz augmentation max tries reached {}'.format(max_tries))
                return nobs, naction

            pc = rearrange(pc, "b t n d -> b (t n) d")
            if pc.shape[2] == 3:
                rotated_pc = (rotation_matrix @ pc.permute(0, 2, 1)).permute(0, 2, 1)
            else:
                xyz = pc[:, :, :3]
                rgb = pc[:, :, 3:]
                rotated_xyz = (rotation_matrix @ xyz.permute(0, 2, 1)).permute(0, 2, 1)
                rotated_pc = torch.cat([rotated_xyz, rgb], dim=2)
            rotated_pc = rearrange(rotated_pc, "b (t n) d -> b t n d", t=T)

            nobs["point_cloud"] = rotated_pc
            nobs["robot0_eef_pos"] = rotated_pos
            # w, x, y, z -> x, y, z, w
            nobs["robot0_eef_quat"] = rotated_quat[:, :, [1, 2, 3, 0]]
            naction = rotated_naction

        return nobs, naction

    def __repr__(self):
        """Pretty print the network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = header + "(min_angle={}, max_angle={})".format(self.min_angle, self.max_angle)
        return msg

