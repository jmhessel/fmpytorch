from __future__ import print_function

import torch
from torch.autograd import Variable
from torch import nn

from .second_order_fast_inner import fast_forward, fast_backward


class SecondOrderInteraction(torch.nn.Module):
    def __init__(self, n_feats, n_factors):
        super(SecondOrderInteraction, self).__init__()
        self.n_feats = n_feats
        self.n_factors = n_factors
        self.v = nn.Parameter(torch.Tensor(self.n_feats, self.n_factors),
                              requires_grad=True)
        self.v.data.uniform_(-0.01, 0.01)

    def forward(self, x):
        return SecondOrderFunction()(x, self.v)

    
class SecondOrderFunction(torch.autograd.Function):
    def forward(self, x, v):
        return fast_forward(self, x, v)
        
    def backward(self, grad_output):
        return fast_backward(self, grad_output)

