from einops import einops
import torch
import pytorch3d.transforms as torch3d_tf


def rot_pcd(pcd, euler_XYZ):
    pcd = einops.rearrange(pcd, 'b n d -> b d n')
    rot_shift_3x3 = torch3d_tf.euler_angles_to_matrix(euler_XYZ, "XYZ").unsqueeze(0).repeat(pcd.shape[0], 1, 1)
    rotated_pcd = torch.bmm(rot_shift_3x3, pcd)
    rotated_pcd = einops.rearrange(rotated_pcd, 'b d n -> b n d')
    return rotated_pcd