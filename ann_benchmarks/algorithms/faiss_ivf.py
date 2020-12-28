from __future__ import absolute_import
import sys
import numpy
import sklearn.preprocessing
import ctypes
import faiss
from ann_benchmarks.algorithms.base import BaseANN
from ann_benchmarks.algorithms.faiss import Faiss


class FaissIVF(Faiss):
    def __init__(self, metric, n_list):
        self._n_list = n_list
        self._metric = metric

    def fit(self, X):
        if self._metric == 'angular':
            X = sklearn.preprocessing.normalize(X, axis=1, norm='l2')

        if X.dtype != numpy.float32:
            X = X.astype(numpy.float32)

        self.quantizer = faiss.IndexFlatL2(X.shape[1])
        index = faiss.IndexIVFFlat(
            self.quantizer, X.shape[1], self._n_list, faiss.METRIC_L2)
        index.train(X)
        index.add(X)
        self.index = index

    def set_query_arguments(self, n_probe):
        faiss.cvar.indexIVF_stats.reset()
        self._n_probe = n_probe
        self.index.nprobe = self._n_probe

    def get_additional(self):
        return {"dist_comps": faiss.cvar.indexIVF_stats.ndis +      # noqa
                faiss.cvar.indexIVF_stats.nq * self._n_list}

    def __str__(self):
        return 'FaissIVF(n_list=%d, n_probe=%d)' % (self._n_list,
                                                    self._n_probe)


class FaissIVFPQ(Faiss):
    def __init__(self, metric, n_list, n_bits):
        self._metric = metric
        self._n_list = n_list
        self._n_bits = n_bits
        self.name = self.__str__()

    def fit(self, X):
        print("start to fit, X.shape[1]=%d, summary=%s" % (X.shape[1], self.__str__()))

        if self._metric == 'angular':
            X = sklearn.preprocessing.normalize(X, axis=1, norm='l2')

        if X.dtype != numpy.float32:
            X = X.astype(numpy.float32)

        self.quantizer = faiss.IndexFlatL2(X.shape[1])
        M = 8
        index = faiss.IndexIVFPQ(
            self.quantizer, X.shape[1], self._n_list, M, self._n_bits, faiss.METRIC_L2)

        index.train(X)
        index.add(X)
        self.index = index

    def __str__(self):
        return 'FaissIVFPQ(n_list=%d, n_probe=%d, n_bits=%d)' % (self._n_list, self._n_probe, self._n_bits)

