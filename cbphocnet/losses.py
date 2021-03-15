import torch
from torch import nn


class CosineLoss(nn.Module):
    def __init__(self, size_average=True, use_sigmoid=False):
        super(CosineLoss, self).__init__()
        self.averaging = size_average
        self.use_sigmoid = use_sigmoid

    def forward(self, input_var, target_var):
        r"""Torch layer with the cosine loss

            Cosine loss:
            1.0 - (y.x / |y|*|x|)

        :param input_var: a torch tensor with the predictions.
        :param target_var: a torch tensor with the groundtruch.
        :return: a tensor with the values to be optimised.
        """
        if self.use_sigmoid:
            loss_val = sum(1.0 - nn.functional.cosine_similarity(torch.sigmoid(input_var), target_var))
        else:
            loss_val = sum(1.0 - nn.functional.cosine_similarity(input_var, target_var))
        if self.averaging:
            loss_val = loss_val/input_var.data.shape[0]
        return loss_val