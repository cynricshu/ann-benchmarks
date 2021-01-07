from __future__ import absolute_import
import sys
sys.path.append("install/lib-faiss")  # noqa
import numpy
import sklearn.preprocessing
import ctypes
import faiss
from ann_benchmarks.algorithms.base import BaseANN


class Faiss(BaseANN):
    def query(self, v, n):
        if self._metric == 'angular':
            v /= numpy.linalg.norm(v)
        D, I = self.index.search(numpy.expand_dims(
            v, axis=0).astype(numpy.float32), n)
        return I[0]

    def batch_query(self, X, n):
        if self._metric == 'angular':
            X /= numpy.linalg.norm(X)
        self.res = self.index.search(X.astype(numpy.float32), n)

    def get_batch_results(self):
        D, L = self.res
        res = []
        for i in range(len(D)):
            r = []
            for l, d in zip(L[i], D[i]):
                if l != -1:
                    r.append(l)
            res.append(r)
        return res


class FaissLSH(Faiss):
    def __init__(self, metric, n_bits):
        self._n_bits = n_bits
        self.index = None
        self._metric = metric
        self.name = 'FaissLSH(n_bits={})'.format(self._n_bits)

    def fit(self, X):
        if X.dtype != numpy.float32:
            X = X.astype(numpy.float32)
        f = X.shape[1]
        self.index = faiss.IndexLSH(f, self._n_bits)
        self.index.train(X)
        self.index.add(X)


class FaissPQ(Faiss):
    def __init__(self, metric, n_M, n_bits):
        self._metric = metric
        self._n_M = n_M
        self._n_bits = n_bits
        self.name = self.__str__()

    def fit(self, X):
        print("start to fit, X.shape[1]=%d, summary=%s" % (X.shape[1], self.__str__()))

        index = faiss.IndexPQ(
            X.shape[1], self._n_M, self._n_bits, faiss.METRIC_L2)
        index.train(X)
        index.add(X)
        self.index = index

    def __str__(self):
        return 'FaissPQ(n_M=%d, n_bits=%d)' % (self._n_M, self._n_bits)

