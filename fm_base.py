from __future__ import print_function

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from second_order_fast import SecondOrderInteraction
#from second_order_naive import SecondOrderInteraction
from torch import nn
from torch.autograd import Variable

        
class FactorizationMachine(torch.nn.Module):
    def __init__(self, input_features, factors):
        super(FactorizationMachine, self).__init__()
        self.input_features, self.factors = input_features, factors
        self.linear = nn.Linear(self.input_features, 1)
        self.second_order = SecondOrderInteraction(self.input_features,
                                                   self.factors)

    def forward(self, x):
        linear = self.linear(x)
        interaction = self.second_order(x)
        return linear + interaction
    
class FMFF(nn.Module):
    '''Simple factorization machine feed forward model.'''
    def __init__(self, in_dim, n_factors):
        
        super(FMFF, self).__init__()
        self.fm_layer = FactorizationMachine(in_dim, n_factors)

    def forward(self, x):
        return self.fm_layer(x)

    
def test_fm():
    np.random.seed(1)
    torch.manual_seed(1)
    model = FMFF(2, 5)
    model.double()
    opt = optim.Adam(model.parameters(),lr=1.0)
    model.train()

    def true_function(x):
        return np.sum(x + 6.0, axis=1)

    for b in range(100000):
        x = np.random.random(size=(32,2))
        y = true_function(x)
        x, y = Variable(torch.from_numpy(x)), Variable(torch.from_numpy(y))
        x.requires_grad=True
        opt.zero_grad()
        out = model(x)
        loss = F.mse_loss(out, y)
        loss.backward()
        print(x.grad)
        opt.step()

        for param in model.parameters():
            print(param.grad)
        
            
        print(b, loss)
        quit()
        
if __name__ == "__main__":
    test_fm()
    
