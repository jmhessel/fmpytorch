import torch
from torch.autograd import Variable
from torch import nn

import time
     
class SecondOrderFunction(torch.autograd.Function):
    @staticmethod
    def forward(x, v):
        batch_size = x.size()[0]
        n_feats = x.size()[-1]
        output = Variable(torch.zeros(batch_size, n_feats, n_feats).double())
        all_interactions = torch.mm(v, v.t())
        for b in range(batch_size):
            for i in range(n_feats):
                for j in range(i+1, n_feats):
                    output[b,i,j] = all_interactions[i,j] * x[b,i] * x[b,j]

        res = output.sum(1).sum(1,keepdim=True)
        return res


class SecondOrderInteraction(torch.nn.Module):
    def __init__(self, n_feats, n_factors):
        super(SecondOrderInteraction, self).__init__()
        self.n_feats = n_feats
        self.n_factors = n_factors
        self.v = nn.Parameter(torch.Tensor(self.n_feats, self.n_factors))
        self.v.data.uniform_(-0.01, 0.01)

    def forward(self, x):
        return SecondOrderFunction().forward(x, self.v)
