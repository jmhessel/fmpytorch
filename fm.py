import torch
from torch.autograd import Variable
from torch.autograd import gradcheck

class SecondOrderFM(torch.autograd.Function):

    def forward(self, x, w0, w1):
        # Follows the notation of Rendle
        self.x = x
        self.w0 = w0
        self.w1 = w1
        return w0 + torch.dot(x,w1)

    def backward(self, grad_output):
        grad_input = torch.mul(self.w1, grad_output)
        grad_w0 = torch.mul(torch.ones(1).double(), grad_output)
        grad_w1 = torch.mul(self.x, grad_output)
        return grad_input, grad_w0, grad_w1


def main():
    N_FEATS = 100
    N_FACTORS = 5

    feats_in = Variable(torch.randn(N_FEATS).double(), requires_grad=True)
    w0 = Variable(torch.randn(1).double(), requires_grad=True)
    w1 = Variable(torch.randn(N_FEATS).double(), requires_grad=True)

    test = gradcheck(SecondOrderFM(), (feats_in,w0,w1), eps=1e-6, atol=1e-4)
    print(test)

if __name__ == '__main__':
    main()
