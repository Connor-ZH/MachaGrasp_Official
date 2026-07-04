import numpy as np
from scipy.spatial.transform import Rotation as R


def gripper_to_hand_pose(hand_type, gripper_pose):
    converters = {
        "shadow": gripper_to_shadow_pose,
        "shadow_hand": gripper_to_shadow_pose,
        "allegro": gripper_to_allegro_pose,
        "allegro_hand": gripper_to_allegro_pose,
        "barrett": gripper_to_barrett_pose,
        "robotiq_3f": gripper_to_robotiq_3f_pose,
    }
    if hand_type not in converters:
        raise ValueError(f"Unsupported hand type: {hand_type}")
    return converters[hand_type](np.asarray(gripper_pose, dtype=np.float32))


def _apply_pose_offset(gripper_pose, rotation_correction, pre_offset, post_offset, extra_offsets):
    if gripper_pose.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 pose, got {gripper_pose.shape}.")

    rotation = gripper_pose[:3, :3]
    translation = gripper_pose[:3, 3].astype(np.float32).copy()
    translation += pre_offset * rotation[:, 2]

    hand_rotation = rotation @ rotation_correction
    translation += post_offset * hand_rotation[:, 2]
    for axis, offset in extra_offsets:
        translation += offset * hand_rotation[:, axis]

    hand_pose = np.eye(4, dtype=np.float32)
    hand_pose[:3, :3] = hand_rotation
    hand_pose[:3, 3] = translation
    return hand_pose


def gripper_to_shadow_pose(gripper_pose):
    correction = R.from_euler("x", -np.pi / 2).as_matrix() @ R.from_euler("y", -np.pi / 2).as_matrix()
    return _apply_pose_offset(
        gripper_pose,
        correction,
        pre_offset=0.065,
        post_offset=-0.1,
        extra_offsets=[(0, -0.02)],
    )


def gripper_to_allegro_pose(gripper_pose):
    correction = R.from_euler("y", -np.pi / 2).as_matrix()
    return _apply_pose_offset(
        gripper_pose,
        correction,
        pre_offset=0.04,
        post_offset=-0.01,
        extra_offsets=[(1, -0.05), (0, 0.0)],
    )


def gripper_to_barrett_pose(gripper_pose):
    correction = R.from_euler("z", np.pi / 2).as_matrix()
    return _apply_pose_offset(
        gripper_pose,
        correction,
        pre_offset=0.04,
        post_offset=-0.01,
        extra_offsets=[(1, -0.005)],
    )


def gripper_to_robotiq_3f_pose(gripper_pose):
    correction = R.from_euler("x", np.pi / 2).as_matrix()
    return _apply_pose_offset(
        gripper_pose,
        correction,
        pre_offset=0.02,
        post_offset=0.0,
        extra_offsets=[(0, 0.0)],
    )
