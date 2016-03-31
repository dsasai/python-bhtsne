# distutils: language = c++
import numpy as np
cimport numpy as np
cimport cython
from libcpp cimport bool

cdef extern from "tsne.h":
    cdef cppclass TSNE:
        TSNE(double* X, int N, int D, double* Y, int no_dims, int rand_seed, bool skip_random_init)
        void fit(double perplexity, double theta)

cdef class BHTSNE:
    cdef TSNE* _tsne

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __cinit__(self, X, no_rows, no_cols, Y, no_dims, rand_seed, seed_positions, skip_random_init):
        cdef np.ndarray[np.float64_t, ndim=2, mode='c'] _X = np.ascontiguousarray(X)
        cdef np.ndarray[np.float64_t, ndim=2, mode='c'] _Y = np.ascontiguousarray(Y)
        #cdef np.ndarray[np.float64_t, ndim=2, mode='c'] Y = np.zeros((no_rows, no_dims), dtype=np.float64)
        if skip_random_init:
            Y = seed_positions
        #self._Y = &Y
        self._tsne = new TSNE(&_X[0,0], no_rows, no_cols, &_Y[0,0], no_dims, rand_seed, skip_random_init)

    def __dealloc__(self):
        del self._tsne

    def fit(self, perplexity, theta):
      self._tsne.fit(perplexity, theta)
