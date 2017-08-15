from __future__ import print_function

import numpy as np
import torch

from torch.autograd import Variable

cimport cython
cimport numpy as np

from cython.view cimport array as cvarray

ctypedef np.float64_t REAL_t

@cython.boundscheck(False)
cdef void _compute_sop_sos(REAL_t[:] sop,
                           REAL_t[:] sos,
                           REAL_t[:] x,
                           REAL_t[:,:] v,
                           int n_feats,
                           int n_factors) nogil:
    cdef int f,i
    for f in range(n_factors):
        for i in range(n_feats):
            sop[f] = sop[f] + v[i,f] * x[i]
            sos[f] = sos[f] + v[i,f]*v[i,f] * x[i]*x[i]

@cython.boundscheck(False)
cdef void _compute_output(REAL_t[:] sop,
                          REAL_t[:] sos,
                          int n_factors,
                          REAL_t[:] output) nogil:
    cdef int f
    cdef int zero = 0
    for f in range(n_factors):
        output[zero] = output[zero] + sop[f] * sop[f] - sos[f]
    output[zero] = output[zero] * .5

def fast_forward(self, x, w0, w1, v):
    self.x = x
    self.w0 = w0
    self.w1 = w1
    self.v = v

    self.n_feats = x.size()[-1]
    self.n_factors = v.size()[1]

    # compute the sum of products for each feature
    self.sum_of_products = np.zeros(self.n_factors)
    self.sum_of_squares = np.zeros(self.n_factors)

    _compute_sop_sos(self.sum_of_products,
                     self.sum_of_squares,
                     self.x.numpy(),
                     self.v.numpy(),
                     self.n_feats,
                     self.n_factors)

    output_factor = np.zeros(1)
    _compute_output(self.sum_of_products,
                    self.sum_of_squares,
                    self.n_factors,
                    output_factor)
    output_factor = output_factor[0]

    return w0 + torch.dot(x,w1) + output_factor

@cython.boundscheck(False)
cdef void _compute_tmp_grad_input(REAL_t[:] sop,
                                  REAL_t[:] x,
                                  REAL_t[:,:] v,
                                  int n_feats,
                                  int n_factors,
                                  REAL_t[:] tgi) nogil:
    cdef int f,i
    for f in range(n_factors):
        for i in range(n_feats):
            tgi[i] = tgi[i] + sop[f] * v[i,f]
            tgi[i] = tgi[i] - x[i] * v[i,f]*v[i,f]

@cython.boundscheck(False)
cdef void _compute_grad_v(REAL_t[:] sop,
                          REAL_t[:] x,
                          REAL_t[:,:] v,
                          int n_feats,
                          int n_factors,
                          REAL_t[:,:] grad_v) nogil:
    cdef int f,i
    for f in range(n_factors):
        for i in range(n_feats):
            grad_v[i,f] = x[i] * sop[f] - v[i,f] * x[i] * x[i]


def fast_backward(self, grad_output):
    tmp_grad_input = torch.zeros(self.n_feats).double()
    _compute_tmp_grad_input(self.sum_of_products,
                            self.x.numpy(),
                            self.v.numpy(),
                            self.n_feats,
                            self.n_factors,
                            tmp_grad_input.numpy())

    grad_input = torch.mul(self.w1 + tmp_grad_input, grad_output)
    grad_w0 = torch.mul(torch.ones(1).double(), grad_output)
    grad_w1 = torch.mul(self.x, grad_output)
    grad_v = torch.zeros(self.n_feats, self.n_factors).double()

    _compute_grad_v(self.sum_of_products,
                    self.x.numpy(),
                    self.v.numpy(),
                    self.n_feats,
                    self.n_factors,
                    grad_v.numpy())

    grad_v = torch.mul(grad_v, grad_output)
    return grad_input, grad_w0, grad_w1, grad_v
