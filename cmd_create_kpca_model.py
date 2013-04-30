#!/usr/bin/env python
# coding: utf-8

# Copyright (C) Vladimir M. Zaytsev <zaytsev@usc.edu>
# Author: Vladimir M. Zaytsev <zaytsev@usc.edu>
# URL: <http://zvm.me/>
# For more information, see README.md
# For license information, see LICENSE

#   SYNOPSIS:
#       python cmd_model_index.py <path to index directory> <dataset size for TFIDF> <dataset size for PCA>


import sys
import logging

from n7.model import FSetLoader
from sklearn.decomposition import KernelPCA


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    input_matrix_name = sys.argv[1] if len(sys.argv) > 1 else "X_tfidf.pkl"
    output_model_name = sys.argv[2] if len(sys.argv) > 2 else "model_kpca.pkl"

    loader = FSetLoader()

    X = loader.load_model(input_matrix_name)
    model = KernelPCA(n_components=128, kernel="sigmoid")
    logging.info("FITTING PCA on %dx%d examples" % (X.shape[0], X.shape[1]))
    model.fit(X.toarray())
    logging.info("FITTING DONE: %r" % model)
    loader.save_model(model, output_model_name)
    loader.save_model(model.lambdas_, output_model_name + ".ev")

