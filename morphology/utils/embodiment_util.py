import pickle
from collections import deque

import numpy as np
import torch


class EmbodimentProperty:
    def __init__(self, asset_file_path, limit_joint=-1):
        asset_file_path = str(asset_file_path)
        if asset_file_path.endswith(".pt"):
            asset_file_contents = torch.load(asset_file_path, weights_only=False)
        else:
            with open(asset_file_path, "rb") as file:
                asset_file_contents = pickle.load(file)

        self.adjacency_matrix = torch.from_numpy(asset_file_contents["adjacency_matrix"]).long()
        self.spatial_distance_matrix = torch.from_numpy(
            asset_file_contents["spatial_distance_matrix"]
        ).long()

        parent_distance_matrix = asset_file_contents["parent_distance_matrix"].copy()
        parent_distance_matrix[np.isinf(parent_distance_matrix)] = 0
        self.parent_distance_matrix = torch.from_numpy(parent_distance_matrix).long()

        child_distance_matrix = asset_file_contents["child_distance_matrix"].copy()
        child_distance_matrix[np.isinf(child_distance_matrix)] = 0
        self.child_distance_matrix = torch.from_numpy(child_distance_matrix).long()

        self.joint_names = asset_file_contents["joint_names"]
        self.dof_count = len(self.adjacency_matrix)
        self.joint_tokens = torch.as_tensor(
            asset_file_contents["joint_encodings"], dtype=torch.float32
        )
        self.joint_revolute_property = torch.as_tensor(
            asset_file_contents["joint_revolute_property"], dtype=torch.bool
        )

        if limit_joint != -1:
            self.adjacency_matrix = self.adjacency_matrix[:limit_joint, :limit_joint]
            self.spatial_distance_matrix = self.spatial_distance_matrix[:limit_joint, :limit_joint]
            self.parent_distance_matrix = self.parent_distance_matrix[:limit_joint, :limit_joint]
            self.child_distance_matrix = self.child_distance_matrix[:limit_joint, :limit_joint]
            self.joint_tokens = self.joint_tokens[:limit_joint]
            self.joint_revolute_property = self.joint_revolute_property[:limit_joint]
            self.dof_count = limit_joint

    def get_spatial_distance_matrix(self, refine=False):
        return self.spatial_distance_matrix

    def get_parent_distance_matrix(self, refine=False):
        return self.parent_distance_matrix

    def get_child_distance_matrix(self, refine=False):
        return self.child_distance_matrix

    def get_joint_tokens(self, domain_randomization=False):
        joint_tokens = self.joint_tokens.clone().float().detach()
        if not domain_randomization:
            return joint_tokens

        revolute_idx = torch.nonzero(self.joint_revolute_property, as_tuple=False).squeeze(1)
        joint_tokens[revolute_idx, 0:11] += (
            torch.rand((revolute_idx.shape[0], 11), device=joint_tokens.device) - 0.5
        ) * 0.002
        joint_tokens[revolute_idx, 12:18] += (
            torch.rand((revolute_idx.shape[0], 6), device=joint_tokens.device) - 0.5
        ) * 0.002
        joint_tokens[revolute_idx, 22:28] += (
            torch.rand((revolute_idx.shape[0], 6), device=joint_tokens.device) - 0.5
        ) * 0.002

        parent_link_type = joint_tokens[revolute_idx, 11].int()
        for link_type, start_idx, width in [(1, 18, 3), (2, 18, 2), (3, 18, 1)]:
            mask = parent_link_type == link_type
            if mask.any():
                joint_tokens[revolute_idx[mask], start_idx : start_idx + width] += (
                    torch.rand((mask.sum(), width), device=joint_tokens.device) - 0.5
                ) * 0.002

        child_link_type = joint_tokens[revolute_idx, 21].int()
        for link_type, start_idx, width in [(1, 28, 3), (2, 28, 2), (3, 28, 1)]:
            mask = child_link_type == link_type
            if mask.any():
                joint_tokens[revolute_idx[mask], start_idx : start_idx + width] += (
                    torch.rand((mask.sum(), width), device=joint_tokens.device) - 0.5
                ) * 0.002

        return joint_tokens

    def get_adjacency_matrix(self):
        return self.adjacency_matrix

    def get_joint_revolute_property(self):
        return self.joint_revolute_property


def compute_spd_matrix(adjacency_matrix: np.ndarray, node_count: int):
    matrix = np.asarray(adjacency_matrix)
    size = matrix.shape[0]
    distances = np.full((size, size), -1, dtype=np.int64)

    for start in range(min(node_count, size)):
        distances[start, start] = 0
        queue = deque([start])
        while queue:
            node = queue.popleft()
            neighbors = np.nonzero(matrix[node])[0]
            for neighbor in neighbors:
                if neighbor >= node_count or distances[start, neighbor] != -1:
                    continue
                distances[start, neighbor] = distances[start, node] + 1
                queue.append(neighbor)

    return distances
