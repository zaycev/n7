#!/usr/bin/env python
# coding: utf-8

# Copyright (C) Vladimir M. Zaytsev <zaytsev@usc.edu>
# Author: Vladimir M. Zaytsev <zaytsev@usc.edu>
# URL: <http://zvm.me/>
# For more information, see README.md
# For license information, see LICENSE

import numpy as np

from sklearn.decomposition import KernelPCA
from sklearn.feature_extraction.text import TfidfTransformer


class FeatureMatrix(object):

    def __init__(self, searcher):
        self.X = None
        self.searcher = searcher
        self.tweet_id_index_map = None
        self.index_tweet_id_map = None

    def from_vectors(self, i_vectors, n=100):
        ni = 0
        m = len(self.searcher.index.id_term_map)
        self.tweet_id_index_map = dict()
        self.index_tweet_id_map = dict()
        self.X = np.zeros((n, m))
        print n, m
        for tweet_id, tweet_vector in i_vectors:
            tweet_vector = self.searcher.term_filter(tweet_vector)
            new_index = len(self.tweet_id_index_map)
            # print tweet_vector
            self.tweet_id_index_map[tweet_id] = new_index
            self.index_tweet_id_map[new_index] = tweet_id
            x = new_index
            for term_id in tweet_vector:
                y = self.searcher.index.id_index_map[term_id]
                # print term_id, "=>", y
                self.X[x, y] += 1
            ni += 1
            if ni >= n:
                break
        # print self.X

    def apply_tfids(self, norm="l2"):
        tfids = TfidfTransformer(norm=norm)
        self.X = tfids.fit_transform(self.X).toarray()

    def apply_kpca(self, n_components=128, kernel="rbf"):
        kpca = KernelPCA(n_components=n_components, kernel=kernel)
        self.X = kpca.fit_transform(self.X)

    def apply_pca(self, n_components=128):
        pca = PCA(n_components=n_components)
        self.X = pca.fit_transform(self.X)

    def project2d(self, kernel="rbf"):
        kpca = KernelPCA(n_components=2, kernel=kernel)
        return kpca.fit_transform(self.X)
