from __future__ import print_function
from itertools import count

import torch
import torch.autograd
import torch.nn.functional as F

from fm_base import FactorizationMachine
from torch.autograd import Variable
from torch import optim

POLY_DEGREE = 4
FACTORS = 5

W_target = torch.randn(POLY_DEGREE, 1).double() * 5
b_target = torch.randn(1).double() * 5

def make_features(x):
    """Builds features i.e. a matrix with columns [x, x^2, x^3, x^4]."""
    x = x.unsqueeze(1)
    return torch.cat([x ** i for i in range(1, POLY_DEGREE+1)], 1).double()


def f(x):
    """Approximated function."""
    return x.mm(W_target) + b_target[0]


def poly_desc(W, b):
    """Creates a string description of a polynomial."""
    result = 'y = '
    for i, w in enumerate(W):
        result += '{:+.2f} x^{} '.format(w, len(W) - i)
    result += '{:+.2f}'.format(b[0])
    return result


def get_batch(batch_size=32):
    """Builds a batch i.e. (x, f(x)) pair."""
    random = torch.randn(batch_size).double()
    x = make_features(random)
    y = f(x)
    return Variable(x).double(), Variable(y).double()


def main():
    fm = FactorizationMachine(POLY_DEGREE, FACTORS)
    optimizer = optim.SGD(fm.parameters(), .0001)
    for batch_idx in count(1):
        # Get data
        batch_x, batch_y = get_batch()

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass
        output = F.smooth_l1_loss(fm(batch_x), batch_y)
        loss = output.data[0]

        # Backward pass
        output.backward()
        optimizer.step()

        # Stop criterion
        if loss < 1e-3:
            break
        print(batch_idx, loss)

    print('Loss: {:.6f} after {} batches'.format(loss, batch_idx))
    print('==> Learned function:\t' + poly_desc(fc.weight.data.view(-1), fc.bias.data))
    print('==> Actual function:\t' + poly_desc(W_target.view(-1), b_target))


if __name__ == '__main__':
    main()
