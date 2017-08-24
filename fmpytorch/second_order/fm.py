'''
The second-order factorization machine class.
'''

from __future__ import print_function

import torch
import torch.nn.functional as F
import torch.optim as optim

from torch import nn
from torch.autograd import Variable

try:
    FAST_VERSION = True
    from .second_order_fast import SecondOrderInteraction
except ImportError:
    FAST_VERSION = False
    from .second_order_naive import SecondOrderInteraction

    
class FactorizationMachine(torch.nn.Module):
    '''Second order factorization machine layer'''
    def __init__(self, input_features, factors):
        '''
        - input_features (int): the length of the input vector.
        - factors (int): the dimension of the interaction terms.
        '''
        
        super(FactorizationMachine, self).__init__()
        if not FAST_VERSION:
            print('Slow version of {0} is being used'.format(__name__))
        self.input_features, self.factors = input_features, factors
        self.linear = nn.Linear(self.input_features, 1)
        self.second_order = SecondOrderInteraction(self.input_features,
                                                   self.factors)

    def forward(self, x):
        # make sure everything is on the CPU.
        self.linear.cpu()
        self.second_order.cpu()
        
        back_to_gpu = False
        
        if x.is_cuda:
            x = x.cpu()
            back_to_gpu = True

        linear = self.linear(x)
        interaction = self.second_order(x)
        res = linear + interaction

        if back_to_gpu:
            res = res.cuda()
            x = x.cuda()

        return res
