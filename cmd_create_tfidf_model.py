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
from sklearn.feature_extraction.text import TfidfTransformer


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    input_matrix_name = sys.argv[1] if len(sys.argv) > 1 else "X_tf.pkl"
    output_model_name = sys.argv[2] if len(sys.argv) > 2 else "model_tfidf.pkl"

    loader = FSetLoader()

    X_tf = loader.load_model(input_matrix_name)
    model = TfidfTransformer()
    logging.info("FITTING TFIDF on %dx%d examples" % (X_tf.shape[0], X_tf.shape[1]))
    model.fit(X_tf)
    logging.info("FITTING DONE: %r" % model)
    loader.save_model(model, output_model_name)

