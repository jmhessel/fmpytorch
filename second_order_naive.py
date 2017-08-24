import torch
from torch.autograd import Variable
from torch import nn

import time

class SecondOrderInteraction(torch.nn.Module):
    def __init__(self, n_feats, n_factors):
        super(SecondOrderInteraction, self).__init__()
        self.n_feats = n_feats
        self.n_factors = n_factors
        self.v = nn.Parameter(torch.Tensor(self.n_feats, self.n_factors))
        self.v.data.uniform_(-0.01, 0.01)
        
    def forward(self, x):
        self.batch_size = x.size()[0]
        self.n_feats = x.size()[-1]
        self.n_factors = self.v.size()[-1]
        output = Variable(x.data.new(self.batch_size, self.n_feats, self.n_feats).zero_())
        all_interactions = torch.mm(self.v, self.v.t())
        for b in range(self.batch_size):
            for i in range(self.n_feats):
                for j in range(i+1, self.n_feats):
                    output[b,i,j] = all_interactions[i,j] * x[b,i] * x[b,j]

        res = output.sum(1).sum(1,keepdim=True)
        return res
