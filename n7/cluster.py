#!/usr/bin/env python
# coding: utf-8

# Copyright (C) Vladimir M. Zaytsev <zaytsev@usc.edu>
# Author: Vladimir M. Zaytsev <zaytsev@usc.edu>
# URL: <http://zvm.me/>
# For more information, see README.md
# For license information, see LICENSE


import numpy as np
import pylab as pl

from itertools import cycle
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.datasets.samples_generator import make_blobs
from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import make_circles
from sklearn.cluster import MiniBatchKMeans, KMeans, Ward, AffinityPropagation, DBSCAN


class Clusterer(object):

    def __init__(self, feature_matrix):
        self.fm = feature_matrix

    def kmean(self, n_clusters, plot=True):
        k_means = Ward(n_clusters=n_clusters, compute_full_tree=True, n_components=2)
        k_means.fit(fm.X)
        k_means_labels = k_means.labels_
        
        pl.close('all')
        pl.figure(1)
        pl.clf()
        
        if plot:
            colors = "rbgcmybgrcmybgrcmybgrcm"
            X2d = model.project2d()
            for i in xrange(len(X2d)):
                x = X2d[i]
                pl.plot(x[0], x[1], "x", markerfacecolor=colors[labels[i]], markeredgecolor=colors[labels[i]])
                pl.plot(x[0], x[1], "x", markerfacecolor="r", markeredgecolor="r")
        
        return k_means.labels_

    def dbscan(self, plot=True):
        X = self.fm.X
        from scipy.spatial import distance
        from sklearn.cluster import DBSCAN
        from sklearn import metrics
        from sklearn.datasets.samples_generator import make_blobs
        D = distance.squareform(distance.pdist(X))
        S = 1 - (D / np.max(D))
        
        db = DBSCAN(eps=0.5, min_samples=5).fit(S)
        core_samples = db.core_sample_indices_
        labels = db.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

        if plot:
            print 'Estimated number of clusters: %d' % n_clusters_
            colors = cycle('bgrcmybgrcmybgrcmybgrcmy')
            for k, col in zip(set(labels), colors):
                if k == -1:
                    # Black used for noise.
                    col = 'k'
                    markersize = 6
                class_members = [index[0] for index in np.argwhere(labels == k)]
                cluster_core_samples = [index for index in core_samples
                                        if labels[index] == k]
                for index in class_members:
                    x = X[index]
                    if index in core_samples and k != -1:
                        markersize = 14
                    else:
                        markersize = 6
                    pl.plot(x[0], x[1], 'o', markerfacecolor=col,
                            markeredgecolor='k', markersize=markersize)
            pl.title('Estimated number of clusters: %d' % n_clusters_)
            pl.show()
        return labels
        
            
    def affinity(self, plot=True):
        X = self.fm.X
        af = AffinityPropagation().fit(X)
        cluster_centers_indices = af.cluster_centers_indices_
        labels = af.labels_

        n_clusters_ = len(cluster_centers_indices)
        
        if plot:
            print 'Estimated number of clusters: %d' % n_clusters_
            import pylab as pl
            from itertools import cycle
            pl.close('all')
            pl.figure(1)
            pl.clf()
            colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
            for k, col in zip(range(n_clusters_), colors):
                class_members = labels == k
                cluster_center = X[cluster_centers_indices[k]]
                pl.plot(X[class_members, 0], X[class_members, 1], col + '.')
                pl.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                        markeredgecolor='k', markersize=14)
                for x in X[class_members]:
                    pl.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)
            pl.title('Estimated number of clusters: %d' % n_clusters_)
            pl.show()
        
        return labels
        