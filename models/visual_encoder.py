from pathlib import Path

import torch.nn as nn
import yaml

from models.pointnet_utils import PointNetSetAbstraction


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PointCloudDecoder(nn.Module):
    def __init__(self, input_dim=1024, num_points=2048):
        super().__init__()
        self.num_points = num_points
        self.output_dim = num_points * 3
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, self.output_dim),
        )

    def forward(self, visual_feature):
        x = self.decoder(visual_feature)
        return x.view(-1, self.num_points, 3)


class PointnetEncoder(nn.Module):
    def __init__(self, path_to_cfg=None, no_feats=False):
        super().__init__()

        if path_to_cfg is None:
            path_to_cfg = PROJECT_ROOT / "configs" / "visual_encoder.yaml"
        with open(path_to_cfg, "r") as f:
            pointnet_params = yaml.safe_load(f)

        if no_feats:
            pointnet_params[0]["in_channel"] = 0

        self.pointnet_modules = nn.ModuleList()
        for _, params in pointnet_params.items():
            in_channel = params["in_channel"] + 3
            sa_module = PointNetSetAbstraction(
                npoint=params["npoint"],
                radius=params["radius"],
                nsample=params["nsample"],
                in_channel=in_channel,
                mlp=params["mlp"],
                group_all=params["group_all"],
                bias=params.get("bias", True),
            )
            self.pointnet_modules.append(sa_module)

    def forward(self, points, point_features=None):
        for pointnet_layer in self.pointnet_modules:
            points, point_features = pointnet_layer(points, point_features)
        return point_features.squeeze(-1)

    def freeze_params(self):
        for param in self.parameters():
            param.requires_grad = False
