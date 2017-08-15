from __future__ import print_function

import torch

from fm import SecondOrderFM
from torch.autograd import Variable
from torch import nn

class FactorizationMachine(torch.nn.Module):
    def __init__(self, input_features, factors):

        super(FactorizationMachine, self).__init__()

        self.input_features, self.factors = input_features, factors

        self.w0 = nn.Parameter(torch.Tensor(1))
        self.w0.data.uniform_(-0.1, 0.1)
        self.w1 = nn.Parameter(torch.Tensor(input_features))
        self.w1.data.uniform_(-0.1, 0.1)
        self.v = nn.Parameter(torch.Tensor(input_features, factors))
        self.v.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        return SecondOrderFM()(input, self.w0, self.w1, self.v)
