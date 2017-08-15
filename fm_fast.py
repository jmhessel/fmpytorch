import numpy as np
import torch
from torch.autograd import Variable
from torch.autograd import gradcheck
import time

from fm_fast_inner import *


class SecondOrderFM(torch.autograd.Function):
    def forward(self, x, w0, w1, v):
        # Follows the notation of Rendle
        self.x = x
        self.w0 = w0
        self.w1 = w1
        self.v = v

        self.n_feats = x.size()[0]
        self.n_factors = v.size()[1]

        # compute the sum of products for each feature
        self.sum_of_products = np.zeros(self.n_factors)
        self.sum_of_squares = np.zeros(self.n_factors)

        _compute_sop_sos(self.sum_of_products,
                         self.sum_of_squares,
                         self.x.numpy(),
                         self.v.numpy(),
                         self.n_feats,
                         self.n_factors)

        output_factor = _compute_output(self.sum_of_products,
                                        self.sum_of_squares,
                                        self.n_factors)

        return w0 + torch.dot(x,w1) + output_factor

    def backward(self, grad_output):
        tmp_grad_input = torch.zeros(self.n_feats).double()
        for i in range(self.n_feats):
            for f in range(self.n_factors):
                tmp_grad_input[i] += self.sum_of_products[f] * self.v[i,f]
                tmp_grad_input[i] -= self.x[i] * self.v[i,f]**2

        grad_input = torch.mul(self.w1 + tmp_grad_input, grad_output)
        grad_w0 = torch.mul(torch.ones(1).double(), grad_output)
        grad_w1 = torch.mul(self.x, grad_output)
        grad_v = torch.zeros(self.n_feats, self.n_factors).double()
        for i in range(self.n_feats):
            for f in range(self.n_factors):
                grad_v[i,f] = self.x[i] * self.sum_of_products[f]
                grad_v[i,f] -= self.v[i,f] * self.x[i]**2

        grad_v = torch.mul(grad_v, grad_output)
        return grad_input, grad_w0, grad_w1, grad_v
