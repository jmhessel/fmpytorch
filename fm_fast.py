import numpy as np
import time
import torch
from fm_fast_inner import fast_forward

class SecondOrderFM(torch.autograd.Function):
    def forward(self, x, w0, w1, v):
        # Follows the notation of Rendle
        return fast_forward(self, x, w0, w1, v)

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
