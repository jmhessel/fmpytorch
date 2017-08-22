import torch
from torch.autograd import Variable

import time

from second_order_fast_inner import fast_forward


class SecondOrderInteraction(torch.autograd.Function):

    def __init__(self, v):
        self.v = v
        self.n_factors = v.size()[-1]
        self.n_feats = v.size()[0]

    
    def forward(self, x):
        return fast_forward(self, x)

    def backward(self, grad_output):
        quit()
