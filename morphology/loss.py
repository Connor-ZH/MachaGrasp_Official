import torch.nn as nn


class JacobianMSELoss(nn.Module):
    def __init__(self, lambda_j=1.0, lambda_mse=0):
        super().__init__()
        self.lambda_j = lambda_j
        self.lambda_mse = lambda_mse

    def forward(self, q_pred, q_gt, jacobian_weight):
        diff2 = (q_pred - q_gt) ** 2
        loss_j = (jacobian_weight.detach() * diff2).mean()
        loss_mse = diff2.mean()
        return self.lambda_j * loss_j + self.lambda_mse * loss_mse
