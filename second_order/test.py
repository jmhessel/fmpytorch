'''
A simple test-case that ensures the forward and backward passes
of the fast, cythonized factorization machine match the forward
and backward passes of the slow, autodiff version.
'''
from __future__ import print_function

import numpy as np
import torch
import torch.nn.functional as F

from second_order_naive import SecondOrderInteraction as SOISlow
from second_order_fast import SecondOrderInteraction as SOIFast
from torch.autograd import Variable

INPUT_SIZE = 50
BATCH_SIZE = 32
N_FACTORS = 5
N_TESTS = 10

class ModelSlow(torch.nn.Module):
    def __init__(self):
        super(ModelSlow, self).__init__()
        self.second_order = SOISlow(INPUT_SIZE, N_FACTORS)

    def forward(self, x):
        x = self.second_order(x)
        return x


class ModelFast(torch.nn.Module):
    def __init__(self):
        super(ModelFast, self).__init__()
        self.second_order = SOIFast(INPUT_SIZE, N_FACTORS)

    def forward(self, x):
        x = self.second_order(x)
        return x

    
def _forward_backward_check(dtype):    
    np.random.seed(1)
    torch.manual_seed(1)
    slow = ModelSlow()

    np.random.seed(1)
    torch.manual_seed(1)
    fast = ModelFast()

    if dtype is np.float64:
        slow.double()
        fast.double()
        
    for i in range(N_TESTS):
        x = Variable(torch.from_numpy(np.random.random((32, INPUT_SIZE)).astype(dtype)))
        y = Variable(torch.from_numpy(np.random.random((32, 1)).astype(dtype)))
        
        out_slow = slow(x)
        out_fast = fast(x)
        
        assert np.allclose(out_slow.data.numpy(),
                           out_fast.data.numpy()), "Forward passes differed for {}".format(dtype)

        loss_slow = F.mse_loss(out_slow, y)
        loss_fast = F.mse_loss(out_fast, y)
        loss_slow.backward()
        loss_fast.backward()

        for var_slow, var_fast in zip(slow.parameters(), fast.parameters()):
            assert np.allclose(var_slow.grad.data.numpy(),
                               var_fast.grad.data.numpy()), "Backward passes differed for {}".format(dtype)

    
def test_forward_backward_float():
    _forward_backward_check(np.float32)

def test_forward_backward_double():
    _forward_backward_check(np.float64)
