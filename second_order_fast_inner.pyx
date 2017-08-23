from __future__ import print_function

import numpy as np
import torch

from torch.autograd import Variable

cimport cython
cimport numpy as np

from cython.view cimport array as cvarray

ctypedef np.float64_t REAL_t

### Forward Pass ###
@cython.boundscheck(False)
cdef void _compute_sop_sos(REAL_t[:,:] sop,
                           REAL_t[:,:] sos,
                           REAL_t[:,:] x,
                           REAL_t[:,:] v,
                           int n_feats,
                           int n_factors,
                           int batch_size) nogil:
    cdef int f,i,b
    for b in range(batch_size):
        for f in range(n_factors):
            for i in range(n_feats):
                sop[b,f] = sop[b,f] + v[i,f] * x[b,i]
                sos[b,f] = sos[b,f] + v[i,f]*v[i,f] * x[b,i]*x[b,i]

@cython.boundscheck(False)
cdef void _compute_output(REAL_t[:,:] sop,
                          REAL_t[:,:] sos,
                          int n_factors,
                          int batch_size,
                          REAL_t[:] output) nogil:
    cdef int f, b
    for b in range(batch_size):
        for f in range(n_factors):
            output[b] = output[b] + sop[b,f] * sop[b,f] - sos[b,f]
        output[b] = output[b] * .5

def fast_forward(self, x, v):
    self.x = x
    self.v = v

    self.batch_size = x.size()[0]
    self.n_feats = x.size()[-1]
    self.n_factors = v.size()[-1]

    # compute the sum of products for each feature
    self.sum_of_products = np.zeros((self.batch_size, self.n_factors))
    self.sum_of_squares = np.zeros((self.batch_size, self.n_factors))

    _compute_sop_sos(self.sum_of_products,
                     self.sum_of_squares,
                     self.x.numpy(),
                     self.v.numpy(),
                     self.n_feats,
                     self.n_factors,
                     self.batch_size)

    output_factor = np.zeros(self.batch_size)
    _compute_output(self.sum_of_products,
                    self.sum_of_squares,
                    self.n_factors,
                    self.batch_size,
                    output_factor)
    
    return torch.from_numpy(output_factor).unsqueeze(-1)


### Backward Pass ###

#@cython.boundscheck(False)
cdef void _compute_grad_input(REAL_t[:,:] sop,
                              REAL_t[:,:] x,
                              REAL_t[:,:] v,
                              int n_feats,
                              int n_factors,
                              int batch_size,
                              REAL_t[:,:] tgi) nogil:
    cdef int f,i,b
    for b in range(batch_size):
        for i in range(n_feats):
            for f in range(n_factors):
                tgi[b,i] = tgi[b,i] + sop[b,f] * v[i,f]
                tgi[b,i] = tgi[b,i] - x[b,i] * v[i,f] * v[i,f]

#@cython.boundscheck(False)
cdef void _compute_grad_v(REAL_t[:,:] sop,
                          REAL_t[:,:] x,
                          REAL_t[:,:] v,
                          REAL_t[:,:] dLdy,
                          int n_feats,
                          int n_factors,
                          int batch_size,
                          REAL_t[:,:,:] grad_v) nogil:
    cdef int f,i,b
    for b in range(batch_size):
        for i in range(n_feats):
            for f in range(n_factors):
                grad_v[b,i,f] = (x[b,i] * sop[b,f] - v[i,f] * x[b,i] * x[b,i]) * dLdy[b,0]

def fast_backward(self, grad_output):
    # this contains
    # d y_k / d x_{k, i}
    grad_input = torch.zeros(self.batch_size,
                             self.n_feats).double()

    _compute_grad_input(self.sum_of_products,
                            self.x.numpy(),
                            self.v.numpy(),
                            self.n_feats,
                            self.n_factors,
                            self.batch_size,
                            grad_input.numpy())

    grad_input = grad_input * grad_output

    # this contains d y_i / dv_{j,k}
    grad_v = torch.zeros(self.batch_size,
                         self.n_feats,
                         self.n_factors).double()

    _compute_grad_v(self.sum_of_products,
                    self.x.numpy(),
                    self.v.numpy(),
                    grad_output.numpy(),
                    self.n_feats,
                    self.n_factors,
                    self.batch_size,
                    grad_v.numpy())

    grad_v = grad_v.sum(0)
    
    return grad_input, grad_v
