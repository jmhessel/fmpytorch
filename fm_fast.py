import numpy as np
import time
import torch
from fm_fast_inner import fast_forward, fast_backward

class SecondOrderFM(torch.autograd.Function):
    def forward(self, x, w0, w1, v):
        # Follows the notation of Rendle
        return fast_forward(self, x, w0, w1, v)

    def backward(self, grad_output):
        return fast_backward(self, grad_output)
