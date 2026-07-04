from pathlib import Path

import torch
import torch.nn as nn
import yaml

from models.embodiment_transformer import EmbodimentTransformer
from models.policy import PolicyModelMultipath
from models.visual_encoder import PointnetEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SynergyGrasper(nn.Module):
    def __init__(
        self,
        embodiment_properties,
        morphology_encoder_config=None,
        visual_encoder_ckpt_path=None,
    ):
        super().__init__()
        self.embodiment_properties = embodiment_properties

        if morphology_encoder_config is None:
            morphology_encoder_config = PROJECT_ROOT / "configs" / "model.yaml"
        if visual_encoder_ckpt_path is None:
            visual_encoder_ckpt_path = PROJECT_ROOT / "checkpoints" / "pretrained_pointnet_encoder.pth"
        self.visual_encoder_ckpt_path = visual_encoder_ckpt_path

        with open(morphology_encoder_config, "r") as file:
            config = yaml.safe_load(file)

        if config.get("visual_encoder_type") != 0:
            raise ValueError("The released training recipe uses the PointNet visual encoder.")
        if config.get("use_pretrain_visual_encoder") is not True:
            raise ValueError("The released training recipe uses the pretrained visual encoder.")
        if config.get("visual_encoder_trainable") is not True:
            raise ValueError("The released training recipe keeps the visual encoder trainable.")
        if config.get("use_rot6d") is not True:
            raise ValueError("The released training recipe uses rot6d wrist rotations.")
        if config["policy_transformer"].get("amplitude_model") != 2:
            raise ValueError("The released training recipe uses amplitude_model: 2.")
        morphology_pooling_cfg = config.get("morphology_pooling", {})
        if (
            morphology_pooling_cfg.get("enable") is not True
            or morphology_pooling_cfg.get("method") != "attention"
        ):
            raise ValueError("The released training recipe uses attention morphology pooling.")

        self.embodiment_transformer = EmbodimentTransformer(
            config, embodiment_properties.get_all_embodiment_properties()
        )

        self.pcl_feature_extractor = PointnetEncoder()

        head_cfg = config["heads"]
        eigengrasp_cnt = head_cfg["eigengrasp_head"]["count"]
        morphology_head_output_dim = head_cfg["morphology_head"]["output_dim"]
        max_eigengrasp_feature_dim = config["max_degree_count"]
        d_model = (
            morphology_head_output_dim
            + max_eigengrasp_feature_dim
            + config["visual_feature_dim"]
            + 3
            + 6
        )

        self.policy = PolicyModelMultipath(
            d_model,
            config["policy_transformer"]["n_head"],
            config["policy_transformer"]["encoder_layers"],
            config["policy_transformer"]["dim_forward"],
            max_eigengrasp_feature_dim,
            config["policy_transformer"]["dropout"],
            eigengrasp_cnt,
        )

    def load_pretrained_pcl_extractor_if_needed(self):
        state_dict = torch.load(self.visual_encoder_ckpt_path)
        self.pcl_feature_extractor.load_state_dict(state_dict)
        print(f"loaded pretrained visual encoder from {self.visual_encoder_ckpt_path}")

    def forward(self, pcl, translation, rotation, embodiment_ids, domain_randomization=False):
        embodiment_ids = torch.flatten(embodiment_ids)
        visual_feature = self.pcl_feature_extractor(pcl)
        tokens = self.embodiment_properties.get_joint_tokens(embodiment_ids, domain_randomization)
        morphology_output = self.embodiment_transformer(tokens, embodiment_ids)

        eigengrasps = morphology_output["eigengrasps"]
        morphology_head = morphology_output["morphology_head"]
        amplitudes = self.policy(eigengrasps, morphology_head, visual_feature, translation, rotation)

        revolute_counts = self.embodiment_properties.get_revolute_tokens(embodiment_ids).sum(dim=1).long()
        range_tensor = torch.arange(amplitudes.shape[1], device=amplitudes.device).unsqueeze(0)
        valid_mask = range_tensor < revolute_counts.unsqueeze(1)
        amplitudes = amplitudes * valid_mask.unsqueeze(-1).float()

        joint_val = (amplitudes * eigengrasps).sum(dim=1)
        return eigengrasps, joint_val, amplitudes

    def freeze_params_except_embodiment_transformer(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.embodiment_transformer.parameters():
            param.requires_grad = True

    def freeze_embodiment_transformer_params(self):
        for param in self.embodiment_transformer.parameters():
            param.requires_grad = False
