'''A quick demo of the factorization machine layer.'''

from __future__ import print_function

import numpy as np
import time
import torch
import torch.nn.functional as F

from fm import FactorizationMachine
from torch.autograd import Variable

N_BATCH = 10000
INPUT_SIZE = 100
HIDDEN_SIZE = 100
N_FACTORS_FM = 5
BATCH_SIZE = 16


class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = torch.nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.dropout = torch.nn.Dropout(.5)
        self.fm = FactorizationMachine(HIDDEN_SIZE, 5)

    def forward(self, x):
        x = self.linear(x)
        x = self.dropout(x)
        x = self.fm(x)
        return x

np.random.seed(1)
torch.manual_seed(1)

model = MyModel()
opt = torch.optim.Adam(model.parameters(), lr=.01)
model.train()

def true_function(input):
    '''A dummy function to learn'''
    return np.sum(input, axis=1)

start = time.time()
for batch in range(10000):
    cur_x = np.random.random(size=(BATCH_SIZE, INPUT_SIZE))
    cur_y = true_function(cur_x)
    cur_x, cur_y = Variable(torch.from_numpy(cur_x)), Variable(torch.from_numpy(cur_y))
    cur_x.float(), cur_y.float()
    opt.zero_grad()
    out = model(cur_x)
    loss = F.mse_loss(out, cur_y)
    loss.backward()
    opt.step()
    print(batch, loss)
end = time.time()

elapsed = end-start
print("{:.3f}ms per batch".format(elapsed/100 * 1000))
