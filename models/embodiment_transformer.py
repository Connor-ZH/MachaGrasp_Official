"""Embodiment transformer modules adapted from GET-Zero."""

import torch
from torch import nn
from typing import Dict, List
from torch.nn.utils.rnn import pad_sequence

from models.attention_pooling import AttentionPooling
from morphology.utils.embodiment_util import EmbodimentProperty
from torch.nn import functional as F
from models.embodiment_attention import EmbodimentTransformerEncoder, SpatialEncodingTransformerEncoderLayer


class EmbodimentTransformer(nn.Module):
    def __init__(self, model_cfg: Dict, embodiment_properties_by_id: List[EmbodimentProperty]):
        super().__init__()

        if len(embodiment_properties_by_id) == 0:
            print('WARNING: no embodiments provided when initializing EmbodimentTransformer. You will be able to initalize the model, but the forward pass may error depending on whether the model config includes components that rely on embodiment information')

        self.num_embodiment_types = len(embodiment_properties_by_id)
        dof_counts_by_id = []

        self.provided_embodiment_max_dof_count = 0
        for embodiment_properties in embodiment_properties_by_id:
            dof_count = embodiment_properties.dof_count
            dof_counts_by_id.append(dof_count)
            self.provided_embodiment_max_dof_count = max(self.provided_embodiment_max_dof_count, dof_count)
        dof_counts_by_id = torch.tensor(dof_counts_by_id, dtype=torch.long)
        self.register_buffer('dof_counts_by_id', dof_counts_by_id, persistent=False)

        self.max_dof_count = self.provided_embodiment_max_dof_count
        self.max_degree_count = model_cfg['max_degree_count']
        print(f"Using max_dof_count={self.max_dof_count}")
        print(f"Using max_degree_count={self.max_degree_count}")

        revolute_joint_property = torch.zeros(self.num_embodiment_types, self.max_dof_count, dtype=torch.bool)
        for embodiment_id in range(len(embodiment_properties_by_id)):
            revolute_props = embodiment_properties_by_id[embodiment_id].get_joint_revolute_property()
            revolute_joint_property[embodiment_id, :len(revolute_props)] = revolute_props.clone().detach().bool()
        self.register_buffer('revolute_joint_property', revolute_joint_property, persistent=False)

        tokenization_cfg = model_cfg['tokenization']
        self.joint_feature_dim = tokenization_cfg['jointFeatureDim']
        self.parent_link_feature_dim = tokenization_cfg['parentLinkFeatureDim']
        self.child_link_feature_dim = tokenization_cfg['childLinkFeatureDim']
        self.total_feature_dim = self.joint_feature_dim + self.parent_link_feature_dim + self.child_link_feature_dim

        self.joint_embedding_dim = model_cfg['joint_embedding_dim']
        self.parent_link_embedding_dim = model_cfg['parent_link_embedding_dim']
        self.child_link_embedding_dim = model_cfg['child_link_embedding_dim']
        self.total_embedding_dim = self.joint_embedding_dim + self.parent_link_embedding_dim + self.child_link_embedding_dim

        self.token_embeddings = nn.ModuleDict()
        self.token_embeddings["share_embedding"] = nn.Sequential(
            nn.Linear(self.total_feature_dim, self.total_embedding_dim),
            nn.ReLU(),
            nn.Linear(self.total_embedding_dim, self.total_embedding_dim),
        )

        self.input_layer_norm = nn.LayerNorm([self.total_embedding_dim])

        transformer_params = model_cfg['transformer']
        self.feedforward_dim = transformer_params['feedforward_dim']
        self.num_attention_heads = transformer_params['num_attention_heads']
        self.num_layers = transformer_params['num_layers']

        encoder_layer_kwargs = {
            'd_model': self.total_embedding_dim,
            'nhead': self.num_attention_heads,
            'dim_feedforward': self.feedforward_dim,
            'norm_first': True, # found to have much more stable training performance in recent literature,
            'dropout': model_cfg['transformer']['dropout'],
            'batch_first': True
        }
        encoder_kwargs = {
            'num_layers': self.num_layers
        }
        print("using spatial encoding transformer layer")
        encoder_layer = SpatialEncodingTransformerEncoderLayer(
            embodiment_properties_by_id,
            self.max_dof_count,
            model_cfg['graphormer']['attention'],
            **encoder_layer_kwargs,
        )
        self.encoder = EmbodimentTransformerEncoder(encoder_layer, **encoder_kwargs)

        self.head_configs = {}
        self.head_modules = nn.ModuleDict()
        head_cfg = model_cfg['heads']

        self.head_configs["morphology_head"] =  morphology_cfg = {**head_cfg["morphology_head"]}
        self.head_modules["morphology_head"] = self._build_mlp(self.total_embedding_dim, morphology_cfg['units'], morphology_cfg['output_dim'], self._activation_name_to_module(morphology_cfg['activation']))

        eigengrasp_cfg = head_cfg["eigengrasp_head"]
        for i in range(eigengrasp_cfg["count"]):
            self.head_configs[f"eigengrasp_{i}"] =eigengrasp_cfg
            self.head_modules[f"eigengrasp_{i}"] = self._build_mlp(self.total_embedding_dim, eigengrasp_cfg['units'], eigengrasp_cfg['output_dim'], self._activation_name_to_module(eigengrasp_cfg['activation']))

        self.morphology_pooling_head = model_cfg['morphology_pooling']['pooling_head']
        self.morphology_pooling = AttentionPooling(model_cfg['heads']['morphology_head']['output_dim'])


    def _activation_name_to_module(self, name):
        if name == 'relu':
            return nn.ReLU
        elif name == 'elu':
            return nn.ELU
        else:
            raise NotImplementedError

    def _build_mlp(self, in_size, intermediate_sizes, out_size, activation):
        modules = []
        if len(intermediate_sizes) > 0:
            sizes = [in_size] + intermediate_sizes
            for i in range(len(sizes) - 1):
                modules.append(nn.Linear(sizes[i], sizes[i+1]))
                modules.append(activation())
            modules.append(nn.Linear(sizes[-1], out_size))
        else:
            modules.append(nn.Linear(in_size, out_size))

        return nn.Sequential(*modules)

    def get_head_names(self):
        return list(self.head_modules.keys())

    def forward(self, tokens, embodiment_ids, head_names=None):
        batch_size = tokens.shape[0]
        largest_dof_count_this_batch = self.max_dof_count
        dof_counts = self.dof_counts_by_id[embodiment_ids]

        src_key_padding_mask = (torch.arange(largest_dof_count_this_batch, device=tokens.device)+1).unsqueeze(0).repeat(batch_size, 1) > dof_counts.unsqueeze(1)

        token_embeddings = self.token_embeddings["share_embedding"](tokens)
        token_embeddings = self.input_layer_norm(token_embeddings)

        encoder_kwargs = {
            'src': token_embeddings,
            'src_key_padding_mask': src_key_padding_mask,
            'embodiment_ids': embodiment_ids,
        }

        tokens = self.encoder(**encoder_kwargs)

        revolute_mask = self.revolute_joint_property[embodiment_ids]
        revolute_tokens_list = [tokens[i][revolute_mask[i]] for i in range(batch_size)]
        padded_revolute_tokens = pad_sequence(revolute_tokens_list, batch_first=True)
        if padded_revolute_tokens.size(1) < self.max_degree_count:
            pad_width = self.max_degree_count - padded_revolute_tokens.size(1)
            padded_revolute_tokens = F.pad(padded_revolute_tokens, (0, 0, 0, pad_width))
        elif padded_revolute_tokens.size(1) > self.max_degree_count:
            padded_revolute_tokens = padded_revolute_tokens[:, :self.max_degree_count, :]

        revolute_lengths = torch.tensor(
            [t.shape[0] for t in revolute_tokens_list],
            device=tokens.device
        )

        revolute_padding_mask = ~(
                    torch.arange(self.max_degree_count, device=tokens.device)[None, :] < revolute_lengths[:, None])


        results_by_head = {}
        head_names_to_use = head_names if head_names is not None else self.head_configs.keys()
        eigengrasp_outputs = []

        for head_name in head_names_to_use:
            head_config = self.head_configs[head_name]
            head_module = self.head_modules[head_name]

            head_output = head_module(padded_revolute_tokens)

            if head_config['squeeze_output_dim'] and head_output.size(2) == 1:
                head_output = head_output.squeeze(2)

            if head_output.dim() == 2:
                head_output = head_output.masked_fill(revolute_padding_mask, 0.0)
            elif head_output.dim() == 3:
                head_output = head_output.masked_fill(revolute_padding_mask.unsqueeze(-1), 0.0)
            else:
                raise ValueError(f"Unexpected head_output shape: {head_output.shape}")

            if head_name == self.morphology_pooling_head:
                head_output = self.morphology_pooling(head_output)


            if head_name == "morphology_head":
                results_by_head[head_name] = head_output
            else:
                if "eigengrasps" in results_by_head:
                    results_by_head["eigengrasps"] = torch.cat((results_by_head["eigengrasps"], head_output.unsqueeze(1)), dim=1)
                else:
                    eigengrasp_outputs.append(head_output.unsqueeze(1))
        results_by_head["eigengrasps"] = torch.cat(eigengrasp_outputs, dim=1)

        return results_by_head
