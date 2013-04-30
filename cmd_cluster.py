#!/usr/bin/env python
# coding: utf-8

# Copyright (C) Vladimir M. Zaytsev <zaytsev@usc.edu>
# Author: Vladimir M. Zaytsev <zaytsev@usc.edu>
# URL: <http://zvm.me/>
# For more information, see README.md
# For license information, see LICENSE

#   SYNOPSIS:
#       python cmd_model_index.py <path to index directory> <dataset size for TFIDF> <dataset size for PCA>


import gc
import sys
import logging

from n7.model import FSetLoader
from n7.cluster import Clusterer
from sklearn.decomposition import PCA


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logging.info("CLUSTERING")

    pca_components = int(sys.argv[2]) if len(sys.argv) > 2 else -1

    input_matrix_name = sys.argv[3] if len(sys.argv) > 3 else "X_tfidf.pkl"
    pca_model_name = sys.argv[4] if len(sys.argv) > 4 else "model_kpca.pkl"

    loader = FSetLoader()

    X = loader.load_model(input_matrix_name)
    pca = loader.load_model(pca_model_name)
    
    logging.info("APPLYING PCA ON %dx%d examples" % (X.shape[0], X.shape[1]))
    X = pca.transform(X)
    logging.info("TRANSFOMATION DONE %r" % X)

    # X1, Y1 = f_set.load_from_csv([
    #     "n7-data/labeled/data_aol.csv",
    #     "n7-data/labeled/data_google.csv",
    #     "n7-data/labeled/data_mcdonalds.csv",
    # ])
    # 
    # logging.info("LOADED FEATURE MATRIX %dx%d" % (X0.shape[0], X0.shape[1]))
    
    cl = Clusterer()
    
    if pca_components > 0:
        pca = PCA(n_components=pca_components)
        X = pca.fit_transform(X)
        import gc
        gc.collect()
        print X.shape
    
    # cl.draw(X)
    # cl.kmeans(X, 66)
    cl.aff(X)
    # cl.dbscan(X)
    
    


