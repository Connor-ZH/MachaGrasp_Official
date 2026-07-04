import os

import numpy as np
import torch
from torch.utils.data import Dataset


class PointCloudDataset(Dataset):
    def __init__(self, pcl_dirs, augment=False):
        self.augment = augment
        self.pcls = []

        for pcl_dir in pcl_dirs:
            if not os.path.isdir(pcl_dir):
                continue
            for file_name in sorted(os.listdir(pcl_dir)):
                if not file_name.endswith(".pt"):
                    continue
                data = torch.load(os.path.join(pcl_dir, file_name), weights_only=False)
                self.pcls.append(data["pcl"].float())

        if len(self.pcls) == 0:
            raise ValueError(f"No .pt point clouds with key 'pcl' found in: {pcl_dirs}")

    def __len__(self):
        return len(self.pcls)

    def __getitem__(self, idx):
        pcl = self.pcls[idx].clone()
        if self.augment:
            pcl = self.apply_augmentation(pcl)
        return pcl

    def apply_augmentation(self, pcl):
        angle = np.random.uniform(0, 2 * np.pi)
        cosval, sinval = np.cos(angle), np.sin(angle)
        rotation_matrix = torch.tensor(
            [[cosval, -sinval, 0.0], [sinval, cosval, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        pcl = pcl @ rotation_matrix.T
        pcl = pcl * np.random.uniform(0.8, 1.2)
        return pcl + torch.randn_like(pcl) * 0.005
