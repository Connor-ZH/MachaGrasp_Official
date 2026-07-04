import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attention_fc = nn.Linear(d_model, 1)

    def forward(self, morphology_encoding):
        attention_scores = self.attention_fc(morphology_encoding)
        attention_weights = F.softmax(attention_scores, dim=1)
        return torch.sum(attention_weights * morphology_encoding, dim=1)
