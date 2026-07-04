from pathlib import Path

import torch
import torch.nn as nn

from morphology.utils.embodiment_util import EmbodimentProperty


META_DIR = Path(__file__).resolve().parent / "meta"


class EmbodimentProperties(nn.Module):
    def __init__(self, embodiment_properties):
        super().__init__()
        self.embodiment_properties = embodiment_properties
        all_joint_tokens, self.max_dof = self.prepare_all_joint_tokens()
        all_joint_revolute_properties = self.prepare_all_revolute_properties()
        self.register_buffer("all_joint_tokens", all_joint_tokens)
        self.register_buffer("all_joint_revolute_properties", all_joint_revolute_properties)

    def get_all_embodiment_properties(self):
        return self.embodiment_properties

    def prepare_all_revolute_properties(self):
        revolute_properties = []
        for embodiment_property in self.embodiment_properties:
            padded = torch.nn.functional.pad(
                embodiment_property.joint_revolute_property,
                (0, self.max_dof - embodiment_property.joint_revolute_property.shape[0]),
            )
            revolute_properties.append(padded)
        return torch.stack(revolute_properties)

    def prepare_all_joint_tokens(self):
        joint_tokens_list = []
        max_dof = 0
        for embodiment_property in self.embodiment_properties:
            joint_tokens = embodiment_property.get_joint_tokens(domain_randomization=False)
            joint_tokens_list.append(joint_tokens)
            max_dof = max(max_dof, joint_tokens.shape[0])

        padded_tokens = [
            torch.nn.functional.pad(tokens, (0, 0, 0, max_dof - tokens.shape[0]))
            for tokens in joint_tokens_list
        ]
        return torch.stack(padded_tokens), max_dof

    def get_spatial_distance_matrix(self, embodiment_id):
        return self.embodiment_properties[embodiment_id].get_spatial_distance_matrix()

    def get_parent_distance_matrix(self, embodiment_id):
        return self.embodiment_properties[embodiment_id].get_parent_distance_matrix()

    def get_child_distance_matrix(self, embodiment_id):
        return self.embodiment_properties[embodiment_id].get_child_distance_matrix()

    def get_joint_tokens(self, embodiment_ids, domain_randomization=False):
        joint_tokens = self.all_joint_tokens[embodiment_ids].clone().float().detach()
        if not domain_randomization:
            return joint_tokens

        revolute_property = self.all_joint_revolute_properties[embodiment_ids]
        revolute_selected = joint_tokens[revolute_property]
        revolute_selected[:, 0:11] += (
            torch.rand((revolute_property.sum(), 11), device=joint_tokens.device) - 0.5
        ) * 0.002
        revolute_selected[:, 12:18] += (
            torch.rand((revolute_property.sum(), 6), device=joint_tokens.device) - 0.5
        ) * 0.002
        revolute_selected[:, 22:28] += (
            torch.rand((revolute_property.sum(), 6), device=joint_tokens.device) - 0.5
        ) * 0.002
        joint_tokens[revolute_property] = revolute_selected

        parent_link_type = joint_tokens[..., 11].long()
        child_link_type = joint_tokens[..., 21].long()

        for link_type, start_idx, width in [(1, 18, 3), (2, 18, 2), (3, 18, 1)]:
            parent_mask = (parent_link_type == link_type) & revolute_property
            if parent_mask.any():
                parent_selected = joint_tokens[parent_mask]
                parent_selected[:, start_idx : start_idx + width] += (
                    torch.rand((parent_mask.sum(), width), device=joint_tokens.device) - 0.5
                ) * 0.002
                joint_tokens[parent_mask] = parent_selected

        for link_type, start_idx, width in [(1, 28, 3), (2, 28, 2), (3, 28, 1)]:
            child_mask = (child_link_type == link_type) & revolute_property
            if child_mask.any():
                child_selected = joint_tokens[child_mask]
                child_selected[:, start_idx : start_idx + width] += (
                    torch.rand((child_mask.sum(), width), device=joint_tokens.device) - 0.5
                ) * 0.002
                joint_tokens[child_mask] = child_selected

        return joint_tokens

    def get_adjacency_matrix(self, embodiment_id):
        return self.embodiment_properties[embodiment_id].get_adjacency_matrix()

    def get_revolute_tokens(self, embodiment_ids):
        return self.all_joint_revolute_properties[embodiment_ids]


def get_embodiment_property(use_cmap=True):
    meta_files = [
        "shadow_joint_encodings_with_link.pt",
        "allegro_joint_encodings_with_link.pt",
        "robotiq_3f_joint_encodings_with_link.pt",
        "barrett_joint_encodings_with_link.pt",
        "human_joint_encodings_with_link.pt",
    ]
    if use_cmap:
        meta_files.extend(
            [
                "shadow_cmap_joint_encodings_with_link.pt",
                "allegro_cmap_joint_encodings_with_link.pt",
                "barrett_cmap_joint_encodings_with_link.pt",
            ]
        )

    return EmbodimentProperties(
        [EmbodimentProperty(META_DIR / meta_file) for meta_file in meta_files]
    )
