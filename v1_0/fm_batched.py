import torch
from torch.autograd import Variable
from torch.autograd import gradcheck
import time

class SecondOrderFMBatched(torch.autograd.Function):
    def forward(self, x, w0, w1, v):
        # Follows the notation of Rendle

        # x is a (batch, dim) tensor
        self.x = x
        self.w0 = w0
        self.w1 = w1
        self.v = v

        self.batch_size = x.size()[0]
        self.n_feats = x.size()[-1]
        self.n_factors = v.size()[-1]

        # maps from batch id to sum of products/sum of squares
        self.sum_of_products = torch.zeros(self.batch_size, self.n_factors).double()
        self.sum_of_squares = torch.zeros(self.batch_size, self.n_factors).double()
        self.output_factor = torch.zeros(self.batch_size).double()
        
        # compute the sum of products for each feature
        for b in range(self.batch_size):
            for f in range(self.n_factors):
                for i in range(self.n_feats):
                    self.sum_of_products[b,f] += v[i,f] * x[b,i]
                    self.sum_of_squares[b,f] += v[i,f]**2 * x[b,i]**2
                self.output_factor[b] += self.sum_of_products[b,f]**2
                self.output_factor[b] -= self.sum_of_squares[b,f]
            self.output_factor[b] *= .5

        output = torch.zeros(self.batch_size).double()
        linear_term = torch.mm(x,w1.unsqueeze(-1))
        for b in range(self.batch_size):
            output[b] = (w0 + linear_term[b] + self.output_factor[b])[0]
        return output

            
    def backward(self, grad_output):
        print(grad_output)
        quit()
        tmp_grad_input = torch.zeros(self.n_feats)
        for i in range(self.n_feats):
            for f in range(self.n_factors):
                tmp_grad_input[i] += self.sum_of_products[f] * self.v[i,f]
                tmp_grad_input[i] -= self.x[i] * self.v[i,f]**2

        grad_input = torch.mul(self.w1 + tmp_grad_input, grad_output)
        grad_w0 = torch.mul(torch.ones(1), grad_output)
        grad_w1 = torch.mul(self.x, grad_output)
        grad_v = torch.zeros(self.n_feats, self.n_factors)
        for i in range(self.n_feats):
            for f in range(self.n_factors):
                grad_v[i,f] = self.x[i] * self.sum_of_products[f]
                grad_v[i,f] -= self.v[i,f] * self.x[i]**2

        grad_v = torch.mul(grad_v, grad_output)
        return grad_input, grad_w0, grad_w1, grad_v


def main():
    #from fm_fast import SecondOrderFM
    N_FEATS = 100
    N_FACTORS = 5
    BATCH_SIZE = 32

    feats_in = Variable(torch.randn(BATCH_SIZE, N_FEATS), requires_grad=True)
    w0 = Variable(torch.randn(1), requires_grad=True)
    w1 = Variable(torch.randn(N_FEATS), requires_grad=True)
    v = Variable(torch.randn(N_FEATS, N_FACTORS), requires_grad=True)

    start = time.time()
    test = gradcheck(SecondOrderFMBatched(), (feats_in,w0,w1,v), eps=1e-6, atol=1e-4)
    end = time.time()
    print(end-start)
    print(test)

if __name__ == '__main__':
    main()
