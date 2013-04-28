#!/usr/bin/env python
# coding: utf-8

# Copyright (C) Vladimir M. Zaytsev <zaytsev@usc.edu>
# Author: Vladimir M. Zaytsev <zaytsev@usc.edu>
# URL: <http://zvm.me/>
# For more information, see README.md
# For license information, see LICENSE

import logging

import numpy as np
import pylab as pl

from itertools import cycle
from sklearn.cluster import MiniBatchKMeans, KMeans, Ward
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.datasets.samples_generator import make_blobs
from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import make_circles


from n7.cluster import Clusterer
from n7.model import FeatureMatrix
from n7.search import TextIndex, Searcher

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    tw_index = TextIndex("n7-data/index")
    tw_search = Searcher(tw_index)
    model = FeatureMatrix(tw_search)
    
    model.from_vectors(tw_search.iterate(), n=500)
    # model.apply_tfids() 
    model.apply_kpca(n_components=2)
    
    c = Clusterer(model)
    
    c.affinity()

    # k_means = Ward(n_clusters=7, compute_full_tree=True, n_components=2)
    # k_means.fit(model.X)
    # k_means_labels = k_means.labels_
    # labels = k_means.labels_

    # pl.close('all')
    # pl.figure(1)
    # pl.clf()
    # 
    # colors = "rbgcmybgrcmybgrcmybgrcm"
    # X2d = model.project2d()
    # for i in xrange(len(X2d)):
    #     x = X2d[i]
    #     pl.plot(x[0], x[1], "x", markerfacecolor=colors[labels[i]], markeredgecolor=colors[labels[i]])
    #     pl.plot(x[0], x[1], "x", markerfacecolor="r", markeredgecolor="r")
    # 
    # colors = cycle('bg')
    # for k, col in zip(set(labels), colors):
    #     class_members = [index[0] for index in np.argwhere(labels == k)]

    # pl.show()
