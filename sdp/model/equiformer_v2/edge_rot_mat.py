import torch


def init_edge_rot_mat(edge_distance_vec):
    edge_vec_0 = edge_distance_vec
    edge_vec_0_distance = torch.norm(edge_vec_0, dim=1)

    # Ensure edge vectors are not too small to avoid division by zero
    if torch.min(edge_vec_0_distance) < 1e-10:
        print("Error edge_vec_0_distance: {}".format(torch.min(edge_vec_0_distance)))

    norm_x = edge_vec_0 / (edge_vec_0_distance.view(-1, 1))

    # Generate a random vector and ensure it is not aligned with norm_x
    edge_vec_2 = torch.rand_like(edge_vec_0) - 0.5
    edge_vec_2 = edge_vec_2 / torch.norm(edge_vec_2, dim=1).view(-1, 1)

    # Create two rotated copys of the random vectors in case the random vector is aligned with norm_x
    # With two 90 degree rotated vectors, at least one should not be aligned with norm_x
    edge_vec_2b = edge_vec_2.clone()
    edge_vec_2b[:, 0] = -edge_vec_2[:, 1]
    edge_vec_2b[:, 1] = edge_vec_2[:, 0]
    edge_vec_2c = edge_vec_2.clone()
    edge_vec_2c[:, 1] = -edge_vec_2[:, 2]
    edge_vec_2c[:, 2] = edge_vec_2[:, 1]

    vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
    vec_dot_b = torch.abs(torch.sum(edge_vec_2b * norm_x, dim=1)).view(-1, 1)
    vec_dot_c = torch.abs(torch.sum(edge_vec_2c * norm_x, dim=1)).view(-1, 1)

    edge_vec_2 = torch.where(torch.gt(vec_dot, vec_dot_b), edge_vec_2b, edge_vec_2)
    vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)

    edge_vec_2 = torch.where(torch.gt(vec_dot, vec_dot_c), edge_vec_2c, edge_vec_2)
    vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1))

    # Check the vectors aren't aligned
    assert torch.max(vec_dot) < 0.99

    norm_z = torch.cross(norm_x, edge_vec_2, dim=1)
    norm_z = norm_z / torch.norm(norm_z, dim=1, keepdim=True)
    norm_y = torch.cross(norm_x, norm_z, dim=1)
    norm_y = norm_y / torch.norm(norm_y, dim=1, keepdim=True)

    # Construct the 3D rotation matrix
    norm_x = norm_x.view(-1, 3, 1)
    norm_y = -norm_y.view(-1, 3, 1)
    norm_z = norm_z.view(-1, 3, 1)

    edge_rot_mat_inv = torch.cat([norm_z, norm_x, norm_y], dim=2)
    edge_rot_mat = torch.transpose(edge_rot_mat_inv, 1, 2)

    return edge_rot_mat.detach()


def init_edge_rot_mat2(edge_distance_vec):
    edge_vec_0 = edge_distance_vec
    edge_vec_0_distance = torch.norm(edge_vec_0, dim=1)

    # Ensure edge vectors are not too small to avoid division by zero
    if torch.min(edge_vec_0_distance) < 1e-10:
        print("Error edge_vec_0_distance: {}".format(torch.min(edge_vec_0_distance)))

    norm_x = edge_vec_0 / (edge_vec_0_distance.view(-1, 1))

    # Use multiple fixed vectors and choose one that is least aligned with norm_x
    fixed_vecs = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], device=edge_vec_0.device, dtype=edge_vec_0.dtype)
    dot_products = torch.abs(torch.matmul(norm_x, fixed_vecs.t()))
    # Find index of the minimum dot product for each norm_x
    min_indices = torch.argmin(dot_products, dim=1)
    selected_fixed_vec = fixed_vecs[min_indices]

    # Check the vectors aren't aligned before using them to compute norm_z
    vec_dot = torch.abs(torch.sum(norm_x * selected_fixed_vec, dim=1))
    assert torch.max(vec_dot) < 0.99, "Vectors are aligned within tolerance."

    norm_z = torch.cross(norm_x, selected_fixed_vec, dim=1)
    norm_z = norm_z / torch.norm(norm_z, dim=1, keepdim=True)
    norm_y = torch.cross(norm_x, norm_z, dim=1)
    norm_y = norm_y / torch.norm(norm_y, dim=1, keepdim=True)

    # Construct the 3D rotation matrix
    norm_x = norm_x.view(-1, 3, 1)
    norm_y = -norm_y.view(-1, 3, 1)
    norm_z = norm_z.view(-1, 3, 1)

    edge_rot_mat_inv = torch.cat([norm_z, norm_x, norm_y], dim=2)
    edge_rot_mat = torch.transpose(edge_rot_mat_inv, 1, 2)

    return edge_rot_mat.detach()
