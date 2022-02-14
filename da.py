from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function, Variable


class GRLayer(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.alpha = 0.1

        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_outputs):
        output = grad_outputs.neg() * ctx.alpha
        return output


def grad_reverse(x):
    return GRLayer.apply(x)


class _DA(nn.Module):
    def __init__(self, dim):
        super(_DA, self).__init__()
        self.dim = dim  # feat layer          256*H*W for vgg16
        self.Conv1 = nn.Conv2d(self.dim, 512, kernel_size=1, stride=1, bias=True)
        self.Conv2 = nn.Conv2d(512, 2, kernel_size=1, stride=1, bias=True)
        self.reLu = nn.ReLU(inplace=False)

    def forward(self, x, need_backprop):
        x = grad_reverse(x)
        x = self.reLu(self.Conv1(x))
        x = self.Conv2(x)
        label = self.LabelResizeLayer(x, need_backprop)
        return x, label
