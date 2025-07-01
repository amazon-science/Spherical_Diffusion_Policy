# using the backbone, different encoders

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from equi_diffpo.model.equivariant_vision.vec_layers import VecLinear
from equi_diffpo.model.equivariant_vision.vec_pointnet import VecPointNet


BACKBONE_DICT = {"vn_pointnet": VecPointNet}

NORMALIZATION_METHOD = {"bn": nn.BatchNorm1d, "in": nn.InstanceNorm1d}


class SIM3Vec4Latent(nn.Module):
    """
    This encoder encode the input point cloud to 4 latents
    Now only support so3 mode
    TODO: se3 and hybrid
    """

    def __init__(
        self,
        c_dim,
        backbone_type,
        backbone_args,
        mode="so3",
        normalization_method=None,
        scale_cano=True,
        mean_cano=True,
    ):
        super().__init__()
        assert mode == "so3", NotImplementedError("TODO, add se3")
        if normalization_method is not None:
            backbone_args["normalization_method"] = NORMALIZATION_METHOD[
                normalization_method
            ]

        self.backbone = BACKBONE_DICT[backbone_type](**backbone_args)
        self.fc_inv = VecLinear(c_dim, c_dim, mode=mode)
        self.scale_cano = scale_cano
        self.mean_cano = mean_cano

    def forward(self, pcl, ret_perpoint_feat=False, target_norm=1.0, rgb=None):
        # B, T, N, _ = pcl.shape
        pcl = pcl.transpose(1, 2)  # [BT, 3, N]
        if rgb is not None:
            rgb = rgb.transpose(1, 2)  # [BT, 3, N]

        input_pcl = pcl.clone()
        centroid = pcl.mean(-1, keepdim=True)  # B,3,1
        z_center = centroid.permute(0, 2, 1)
        if self.mean_cano:
            input_pcl = pcl - centroid
        else:
            z_center = torch.zeros_like(z_center)
        z_scale = input_pcl.norm(dim=1).mean(-1) / target_norm  # B
        if self.scale_cano:
            input_pcl = input_pcl / z_scale[:, None, None]
        else:
            z_scale = torch.ones_like(z_scale)

        x, x_perpoint = self.backbone(input_pcl, rgb)  # B,C,3

        z_so3 = x
        z_inv_dual, _ = self.fc_inv(x[..., None])
        z_inv_dual = z_inv_dual.squeeze(-1)
        z_inv = (z_inv_dual * z_so3).sum(-1)

        ret = {
            "inv": z_inv,
            "so3": z_so3,
            "scale": z_scale,
            "center": z_center,
        }

        if ret_perpoint_feat:
            ret["per_point_so3"] = x_perpoint  # [B, C, 3, N]

        return ret
