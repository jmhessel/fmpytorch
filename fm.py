import torch
from torch.autograd import Variable
from torch.autograd import gradcheck

class SecondOrderFM(torch.autograd.Function):

    def forward(self, x, w0, w1, v):
        # Follows the notation of Rendle
        self.x = x
        self.w0 = w0
        self.w1 = w1
        self.v = v

        self.n_feats = x.size()[0]
        self.n_factors = v.size()[1]

        # compute the sum of products for each feature
        self.sum_of_products = torch.zeros(self.n_factors).double()
        self.sum_of_squares = torch.zeros(self.n_factors).double()
        output_factor = torch.zeros(1).double()
        for f in range(self.n_factors):
            for i in range(self.n_feats):
                self.sum_of_products[f] += v[i,f] * x[i]
                self.sum_of_squares[f] += v[i,f]**2 * x[i]**2
            output_factor += self.sum_of_products[f]**2
            output_factor -= self.sum_of_squares[f]
        output_factor *= .5

        return w0 + torch.dot(x,w1) + output_factor

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


def main():
    N_FEATS = 100
    N_FACTORS = 5

    feats_in = Variable(torch.randn(N_FEATS).double(), requires_grad=True)
    w0 = Variable(torch.randn(1).double(), requires_grad=True)
    w1 = Variable(torch.randn(N_FEATS).double(), requires_grad=True)
    v = Variable(torch.randn(N_FEATS, N_FACTORS).double(), requires_grad=True)

    test = gradcheck(SecondOrderFM(), (feats_in,w0,w1,v), eps=1e-6, atol=1e-4)
    print(test)

if __name__ == '__main__':
    main()
