# -*- coding: utf-8 -*-
# Author: X.Yang
# Concat: pistonyang@gmail.com
# Date  : 3/26/19

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.autograd.function import Function


class ShakeShake(Function):
    @staticmethod
    def forward(ctx, x1, x2, training=True):
        if training:
            alpha = torch.cuda.FloatTensor(x1.size(0)).uniform_()
            alpha = alpha.view(alpha.size(0), 1, 1, 1).expand_as(x1)
        else:
            alpha = 0.5
        return alpha * x1 + (1 - alpha) * x2

    @staticmethod
    def backward(ctx, grad_outputs):
        beta = torch.cuda.FloatTensor(grad_outputs.size(0)).uniform_()
        beta = beta.view(beta.size(0), 1, 1, 1).expand_as(grad_outputs)
        beta = Variable(beta)
        return beta * grad_outputs, (1 - beta) * grad_outputs, None