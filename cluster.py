#!/usr/bin/env python
# coding: utf-8

# Copyright (C) Vladimir M. Zaytsev <zaytsev@usc.edu>
# Author: Vladimir M. Zaytsev <zaytsev@usc.edu>
# URL: <http://zvm.me/>
# For more information, see README.md
# For license information, see LICENSE

import logging

import time

import numpy as np
import pylab as pl

from itertools import cycle
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.datasets.samples_generator import make_blobs
from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import make_circles


from n7.models import FeatureMatrix
from n7.search import TextIndex, Searcher

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    tw_index = TextIndex("n7-data/index")
    tw_search = Searcher(tw_index)
    model = FeatureMatrix(tw_search)

    model.from_vectors(tw_search.iterate(), n=10000)

    kpca = KernelPCA(n_components=128, kernel="rbf", fit_inverse_transform=True, gamma=10)
    X_kpca1 = kpca.fit_transform(model.X)
    kpca2 = KernelPCA(n_components=2, kernel="rbf", fit_inverse_transform=True, gamma=10)
    X_kpca2 = kpca2.fit_transform(model.X)

    k_means = KMeans(init='k-means++', n_clusters=3, n_init=10)
    t0 = time.time()
    k_means.fit(X_kpca1)
    t_batch = time.time() - t0
    k_means_labels = k_means.labels_
    k_means_cluster_centers = k_means.cluster_centers_
    k_means_labels_unique = np.unique(k_means_labels)


    labels = k_means.labels_

    pl.close('all')
    pl.figure(1)
    pl.clf()

    for x in X_kpca2:
        pl.plot(x[0], x[1], 'o', markerfacecolor="r", markeredgecolor='k')

    colors = cycle('bgrcmybgrcmybgrcmybgrcmy')
    for k, col in zip(set(labels), colors):
        if k == -1:
            # Black used for noise.
            col = 'k'
            markersize = 6
        class_members = [index[0] for index in np.argwhere(labels == k)]

pl.show()

