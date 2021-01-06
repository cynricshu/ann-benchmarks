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
    def __init__(self, metric, n_list, n_M, n_bits):
        self._metric = metric
        self._n_list = n_list
        self._n_M = n_M
        self._n_bits = n_bits
        self.name = 'FaissIVFPQ(n_list=%d, n_M=%d, n_bits=%d)' % (self._n_list, self._n_M, self._n_bits)

    def fit(self, X):
        # is this necessary?
        if self._metric == 'angular':
            X = sklearn.preprocessing.normalize(X, axis=1, norm='l2')

        if X.dtype != numpy.float32:
            X = X.astype(numpy.float32)

        self.quantizer = faiss.IndexFlatL2(X.shape[1]) # can be changed in further
        #index = faiss.IndexIVFPQ(
        #    self.quantizer, X.shape[1], self._n_list, self._n_M, self._n_bits, faiss.METRIC_L2)
        index = faiss.index_factory(X.shape[1], f"IVF{self._n_list},PQ{self._n_M}x{self._n_bits}", faiss.METRIC_L2)

        index.train(X[:250000])
        index.add(X)
        self.index = index

    def set_query_arguments(self, n_probe):
        faiss.cvar.indexIVF_stats.reset()
        faiss.cvar.indexIVFPQ_stats.reset()
        self._n_probe = n_probe
        self.index.nprobe = self._n_probe

    def __str__(self):
        return 'FaissIVFPQ(n_list=%d, n_M=%d, n_bits=%d, n_probe=%d)' % (self._n_list, self._n_M, self._n_bits, self._n_probe)


class FaissIVFPQFS(Faiss):
    def __init__(self, metric, n_list, n_M):
        self._metric = metric
        self._n_list = n_list
        self._n_M = n_M
        self.name = 'FaissIVFPQFS(n_list=%d, n_M=%d)' % (self._n_list, self._n_M)

    def fit(self, X):
        if self._metric == 'angular':
            X = sklearn.preprocessing.normalize(X, axis=1, norm='l2')

        if X.dtype != numpy.float32:
            X = X.astype(numpy.float32)

        index = faiss.index_factory(X.shape[1], f"IVF{self._n_list},PQ{self._n_M}x4fs", faiss.METRIC_L2)

        index.train(X[:250000])
        index.add(X)
        self.index = index

    def set_query_arguments(self, n_probe):
        #faiss.cvar.indexIVF_stats.reset()
        #faiss.cvar.IVFFastScan_stats.reset()
        self._n_probe = n_probe
        self.index.nprobe = self._n_probe

    def __str__(self):
        return 'FaissIVFPQFS(n_list=%d, n_M=%d, n_probe=%d)' % (self._n_list, self._n_M, self._n_probe)


class FaissIVFPQFSr(Faiss):
    def __init__(self, metric, n_list):
        self._metric = metric
        self._n_list = n_list
        self.name = 'FaissIVFPQFS,RFlat(n_list=%d)' % (self._n_list)

    def fit(self, X):
        if self._metric == 'angular':
            X = sklearn.preprocessing.normalize(X, axis=1, norm='l2')

        if X.dtype != numpy.float32:
            X = X.astype(numpy.float32)

        M = X.shape[1] // 2
        index_build_str = f"IVF{self._n_list},PQ{M}x4fs,RFlat"
        print(f"index_build_str={index_build_str}")
        index = faiss.index_factory(X.shape[1], index_build_str, faiss.METRIC_L2)
        index.train(X[:250000])
        index.add(X)

        self.index = index
    
    def set_query_arguments(self, n_probe, n_reorder_k):
        faiss.cvar.indexIVF_stats.reset()
        faiss.cvar.IVFFastScan_stats.reset()
        self._n_probe = n_probe
        self._n_reorder_k = n_reorder_k 
        self.index.nprobe = self._n_probe

    def __str__(self):
        return 'FaissIVFPQFS,RFlat(n_list=%d, n_probe=%d, n_reorder_k=%d)' % (self._n_list, self._n_probe, self._n_reorder_k)
