import torch
import torch.nn as nn


class AmplitudeModel(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.feature_reduction = nn.Linear(d_model, 256)
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(32)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        batch_size, eigengrasp_count, feature_dim = x.shape
        x = self.feature_reduction(x.view(-1, feature_dim))
        x = x.view(batch_size * eigengrasp_count, 1, -1)
        x = self.relu(self.batch_norm(self.conv1d(x)))
        x = torch.mean(x, dim=2)
        x = self.fc(x)
        return x.view(batch_size, eigengrasp_count, 1)


class PolicyModelMultipath(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        num_encoder_layers,
        dim_feedforward,
        max_eigengrasp_feature_dim,
        dropout,
        eigengrasp_cnt,
    ):
        super().__init__()
        self.eigengrasp_cnt = eigengrasp_cnt
        self.max_eigengrasp_feature_dim = max_eigengrasp_feature_dim
        self.d_model_in = d_model
        self.d_model_out = 1566

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model_out,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            norm_first=True,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        self.input_norm = nn.LayerNorm(self.d_model_in)
        self.projection_layer = nn.Linear(self.d_model_in, self.d_model_out, bias=False)
        self.amplitude_heads = nn.ModuleList(
            [AmplitudeModel(self.d_model_out) for _ in range(eigengrasp_cnt)]
        )

    def forward(self, eigengrasps, morphology_encoding, visual_feature, translation, rotation):
        batch_size, eigengrasp_cnt, _ = eigengrasps.shape

        morphology_encoding = morphology_encoding.unsqueeze(1).expand(batch_size, eigengrasp_cnt, -1)

        visual_feature = visual_feature.unsqueeze(1).expand(batch_size, eigengrasp_cnt, -1)
        translation = translation.unsqueeze(1).expand(batch_size, eigengrasp_cnt, -1)
        rotation = rotation.unsqueeze(1).expand(batch_size, eigengrasp_cnt, -1)

        x = torch.cat(
            [eigengrasps, morphology_encoding, visual_feature, translation, rotation],
            dim=-1,
        )
        x = self.projection_layer(self.input_norm(x))
        transformer_output = self.transformer_encoder(x)

        amplitudes = [
            head(transformer_output[:, idx, :].unsqueeze(1)).squeeze(1)
            for idx, head in enumerate(self.amplitude_heads)
        ]
        return torch.stack(amplitudes, dim=1)
