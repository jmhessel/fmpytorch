from __future__ import print_function

cimport numpy as np
ctypedef np.float64_t REAL_t

cdef _compute_sop_sos(np.ndarray[REAL_t] sop,
                      np.ndarray[REAL_t] sos,
                      np.ndarray[REAL_t] x,
                      np.ndarray[REAL_t, ndim=2] v,
                      int n_factors,
                      int n_feats):
    cdef int f,i
    for f in range(n_factors):
        for i in range(n_feats):
            sop[f] = sop[f] + v[i,f] * x[i]
            sos[f] = sos[f] + v[i,f]*v[i,f] * x[i]*x[i]

cdef _compute_output(np.ndarray[REAL_t] sop,
                     np.ndarray[REAL_t] sos,
                     int n_factors):
    cdef REAL_t output
    cdef int f
    for f in range(n_factors):
        output = output + sop[f] * sop[f] - sos[f]
    return output
