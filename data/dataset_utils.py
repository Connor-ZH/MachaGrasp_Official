import numpy as np
import torch
import transforms3d
from scipy.spatial.transform import Rotation as R


def euler2rot6d(euler):
    rot = np.array(transforms3d.euler.euler2mat(*euler))
    ortho6d = rot[:, :2].T.ravel()
    return torch.from_numpy(ortho6d.astype(np.float32))


def rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    if rot6d.shape[-1] != 6:
        raise ValueError(f"rot6d last dim must be 6, got {rot6d.shape}")

    a1 = rot6d[..., 0:3]
    a2 = rot6d[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1, eps=1e-8)
    dot = (b1 * a2).sum(dim=-1, keepdim=True)
    b2 = torch.nn.functional.normalize(a2 - dot * b1, dim=-1, eps=1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def matrix_to_rot6d(rotation_matrix: torch.Tensor) -> torch.Tensor:
    c0 = rotation_matrix[..., :, 0]
    c1 = rotation_matrix[..., :, 1]
    return torch.cat([c0, c1], dim=-1)


def rotate_pointcloud_and_hand_pose(
    pcl: torch.Tensor,
    wrist_translation: torch.Tensor,
    wrist_euler_sxyz: torch.Tensor,
    rotation_deg: np.ndarray = None,
):
    if rotation_deg is None:
        rotation_deg = np.random.uniform(-20, 20, size=3)

    delta_rot = R.from_euler("xyz", rotation_deg, degrees=True)
    rotation_matrix_np = delta_rot.as_matrix().astype(pcl.cpu().numpy().dtype)
    rotation_matrix = torch.tensor(rotation_matrix_np, device=pcl.device, dtype=pcl.dtype)

    pcl_rotated = rotation_matrix @ pcl
    wrist_translation_rotated = rotation_matrix @ wrist_translation.to(dtype=pcl.dtype)

    wrist_rot = R.from_euler("xyz", wrist_euler_sxyz.cpu().numpy())
    wrist_new_rot = delta_rot * wrist_rot
    wrist_euler_rotated = torch.tensor(
        wrist_new_rot.as_euler("xyz"),
        dtype=pcl.dtype,
        device=pcl.device,
    )
    return pcl_rotated, wrist_translation_rotated, wrist_euler_rotated


def rotate_pointcloud_and_hand_pose_rot6d(
    pcl: torch.Tensor,
    wrist_translation: torch.Tensor,
    wrist_rot6d: torch.Tensor,
    rotation_deg: np.ndarray = None,
):
    if rotation_deg is None:
        rotation_deg = np.random.uniform(-90, 90, size=3)

    rotation_np = R.from_euler("xyz", rotation_deg, degrees=True).as_matrix().astype(np.float32)
    rotation_matrix = torch.tensor(rotation_np, device=pcl.device, dtype=pcl.dtype)

    pcl_rotated = rotation_matrix @ pcl
    wrist_translation_rotated = rotation_matrix @ wrist_translation.to(
        device=pcl.device, dtype=pcl.dtype
    )
    wrist_rot6d = wrist_rot6d.to(device=pcl.device, dtype=pcl.dtype)
    wrist_rot6d_rotated = matrix_to_rot6d(rotation_matrix @ rot6d_to_matrix(wrist_rot6d))
    return pcl_rotated, wrist_translation_rotated, wrist_rot6d_rotated
