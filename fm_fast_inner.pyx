from __future__ import print_function

import numpy as np
import torch

from torch.autograd import Variable

cimport cython
cimport numpy as np

from cython.view cimport array as cvarray

ctypedef np.float64_t REAL_t


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

cdef REAL_t _compute_output(REAL_t[:] sop,
                            REAL_t[:] sos,
                            int n_factors) nogil:
    cdef REAL_t output
    cdef int f
    for f in range(n_factors):
        output = output + sop[f] * sop[f] - sos[f]
    return output * .5

def fast_forward(self, x, w0, w1, v):
    self.x = x
    self.w0 = w0
    self.w1 = w1
    self.v = v

    self.n_feats = x.size()[0]
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

    output_factor = _compute_output(self.sum_of_products,
                                    self.sum_of_squares,
                                    self.n_factors)

    return w0 + torch.dot(x,w1) + output_factor
