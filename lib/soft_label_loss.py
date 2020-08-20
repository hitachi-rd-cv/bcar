import torch
from torch import nn


class SoftLabelLoss(nn.Module):

    def __init__(self):
        super(SoftLabelLoss, self).__init__()

    def forward(self, y_pred, y, reduction=True):
        assert len(y_pred.shape) == 2, 'Only support two dimensional input'

        y_pred = y_pred.log_softmax(dim=1)

        if reduction:
            return torch.mean(torch.sum(-y * y_pred, dim=1))
        else:
            return torch.sum(-y * y_pred, dim=1)
