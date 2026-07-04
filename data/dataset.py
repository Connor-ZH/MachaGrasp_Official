import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

from .dataset_utils import euler2rot6d, rotate_pointcloud_and_hand_pose


DEFAULT_PROJECT_ROOT = os.environ.get("SYNERGY_GRASP_PROJECT_ROOT", str(Path(__file__).resolve().parents[1]))
PCL_NOISE_FACTOR = 0.002
ARTICULATION_NOISE_FACTOR = 0.002
TRANS_NOISE_FACTOR = 0.001
ROT_NOISE_FACTOR = 0.01


class SynergyDataset:
    def __init__(
        self,
        split,
        eigengrasp_head_cnt,
        max_dof,
        augment=False,
        data_root=DEFAULT_PROJECT_ROOT,
    ):
        self.allegro_dataset = SynergyDatasetAllegroHand(
            split,
            eigengrasp_head_cnt,
            max_dof,
            augment=augment,
            data_root=data_root,
        )
        self.shadow_dataset = SynergyDatasetShadowHand(
            split,
            eigengrasp_head_cnt,
            max_dof,
            augment=augment,
            data_root=data_root,
        )
        self.barrett_dataset = SynergyDatasetBarrett(
            split,
            eigengrasp_head_cnt,
            max_dof,
            augment=augment,
            data_root=data_root,
        )
        self.dataset = ConcatDataset(
            [self.allegro_dataset, self.shadow_dataset, self.barrett_dataset]
        )
        self.split = split
        self.length = len(self.allegro_dataset) + len(self.shadow_dataset) + len(self.barrett_dataset)
        print(f"total {self.length} samples")

        if split == "train":
            weights = np.concatenate(
                [
                    np.full(len(self.allegro_dataset), 1 / len(self.allegro_dataset)),
                    np.full(len(self.shadow_dataset), 1 / len(self.shadow_dataset)),
                    np.full(len(self.barrett_dataset), 1 / len(self.barrett_dataset)),
                ]
            )
            self.sampler = WeightedRandomSampler(weights, num_samples=len(self.dataset), replacement=True)
        else:
            self.sampler = None

    def get_loader(self, batch_size, num_workers):
        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            sampler=self.sampler if self.split == "train" else None,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    def __len__(self):
        return self.length


class _HandDataset(Dataset):
    hand_name = None
    dataset_dir_name = None
    pointcloud_dir_name = None

    def __init__(
        self,
        split,
        eigengrasp_head_cnt,
        max_dof,
        pcl_dir=None,
        augment=False,
        data_root=DEFAULT_PROJECT_ROOT,
    ):
        print(f"loading {self.hand_name} {split} dataset")
        self.split = split
        self.augment = augment
        self.max_dof = max_dof

        data_dir = os.path.join(data_root, "data")
        dataset_dir = os.path.join(data_dir, self.dataset_dir_name)
        if pcl_dir is None:
            pcl_dir = os.path.join(data_dir, self.pointcloud_dir_name)

        start_time = time.time()
        self.pcl_dict = self._load_pointclouds(pcl_dir)
        self.dataset_path = os.path.join(dataset_dir, f"{split}_2.pt")
        self.data = torch.load(self.dataset_path, weights_only=False)

        eigengrasp_path = os.path.join(dataset_dir, "eigengrasps_2_whitened.pt")
        print(f"loading eigengrasps from {eigengrasp_path}")
        raw_eigengrasp = torch.load(eigengrasp_path, weights_only=False)[:eigengrasp_head_cnt]
        self.eigengrasp = F.pad(
            raw_eigengrasp,
            (0, max_dof - raw_eigengrasp.shape[1]),
            mode="constant",
            value=0,
        ).to(torch.float32)

        self._prepare_metadata()
        print(f"loaded {self.hand_name} {split} dataset in {time.time() - start_time:.2f}s")

    def _load_pointclouds(self, pcl_dir):
        pcl_dict = {}
        for file_name in os.listdir(pcl_dir):
            if not file_name.endswith(".pt"):
                continue
            pcl_data = torch.load(os.path.join(pcl_dir, file_name), weights_only=False)
            pcl_dict[file_name[:-3]] = torch.as_tensor(pcl_data["pcl"]).permute(1, 0).float()
        return pcl_dict

    def _prepare_metadata(self):
        pass

    def _pointcloud_key(self, grasp_code, scale):
        return grasp_code

    def __len__(self):
        return len(self.data["pose"])

    def __getitem__(self, idx):
        pose = self.data["pose"][idx]
        trans = pose[0:3].float()
        rot = pose[3:6].float()
        raw_articulation = pose[6:].float()
        articulation = F.pad(
            raw_articulation,
            (0, self.max_dof - raw_articulation.shape[0]),
            mode="constant",
            value=0,
        )

        grasp_code = str(self.data["grasp_code"][idx])
        scale = self.data["scale"][idx]
        pcl = self.pcl_dict[self._pointcloud_key(grasp_code, scale)].float()
        embodiment_id = self.data["embodiment_id"][idx].item()
        translation_shift = self.data["translation_shift"][idx].float()

        if self.augment and self.split == "train":
            pcl, trans, rot = rotate_pointcloud_and_hand_pose(pcl, trans, rot)

        if self.split == "train":
            pcl = pcl + torch.randn_like(pcl) * PCL_NOISE_FACTOR
            trans = trans + torch.randn_like(trans) * TRANS_NOISE_FACTOR
            rot = rot + torch.randn_like(rot) * ROT_NOISE_FACTOR
            articulation = articulation + torch.randn_like(articulation) * ARTICULATION_NOISE_FACTOR

        rot = euler2rot6d(rot)

        if "jacobian_weight" in self.data:
            jacobian_weight = self.data["jacobian_weight"][idx]
            jacobian_weight = F.pad(
                jacobian_weight,
                (0, self.max_dof - jacobian_weight.shape[0]),
                mode="constant",
                value=0,
            )
        elif self.split == "train":
            raise KeyError(
                f"{self.dataset_path} is missing "
                "'jacobian_weight'. Use the released weighted split files before training."
            )
        else:
            jacobian_weight = torch.ones_like(articulation)

        return (
            trans,
            rot,
            pcl,
            articulation,
            embodiment_id,
            self.eigengrasp,
            translation_shift,
            grasp_code,
            float(scale),
            0,
            0,
            jacobian_weight,
        )


class SynergyDatasetAllegroHand(_HandDataset):
    hand_name = "allegro hand"
    dataset_dir_name = "allegro"
    pointcloud_dir_name = "pointcloud_allegro"


class SynergyDatasetBarrett(_HandDataset):
    hand_name = "barrett"
    dataset_dir_name = "barrett"
    pointcloud_dir_name = "pointcloud_barrett"


class SynergyDatasetShadowHand(_HandDataset):
    hand_name = "shadow hand"
    dataset_dir_name = "shadow"
    pointcloud_dir_name = "pointcloud_shadow"

    def _load_pointclouds(self, pcl_dir):
        cache_path = os.path.join(pcl_dir, "pcl_dict_shadow.pt")
        if os.path.exists(cache_path):
            print(f"found cached point clouds at {cache_path}")
            return torch.load(cache_path)

        pcl_dict = {}
        for file_name in os.listdir(pcl_dir):
            if not file_name.endswith(".npy"):
                continue
            pcl = torch.load(os.path.join(pcl_dir, file_name))["pcl"]
            pcl_dict[file_name[:-4]] = torch.as_tensor(pcl).permute(1, 0).float()
        torch.save(pcl_dict, cache_path)
        print(f"cached {len(pcl_dict)} shadow point clouds at {cache_path}")
        return pcl_dict

    def _prepare_metadata(self):
        scale_values = self.data["scale"].view(-1).tolist()
        self.data["scale"] = [f"{scale:.2f}" for scale in scale_values]

    def _pointcloud_key(self, grasp_code, scale):
        return f"{grasp_code}_{scale}"
