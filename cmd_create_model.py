#!/usr/bin/env python
# coding: utf-8

# Copyright (C) Vladimir M. Zaytsev <zaytsev@usc.edu>
# Author: Vladimir M. Zaytsev <zaytsev@usc.edu>
# URL: <http://zvm.me/>
# For more information, see README.md
# For license information, see LICENSE

#   SYNOPSIS:
#       python cmd_model_index.py <path to index directory> <dataset size>


import gc
import csv
import sys
import logging


from n7 import N7_DATA_DIR
from n7.model import FeatureSet

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    index_directory = sys.argv[1]
    
    if len(sys.argv) > 2:
        tfidf_examples = int(sys.argv[2])
    else:
        tfidf_examples = 10 ** 6
    if len(sys.argv) > 3:
        pca_examples = int(sys.argv[3])
    else:
        pca_examples = 10 ** 5
    
    logging.info("CREATING MODEL")
    logging.info("INDEX DIRECTORY: %s" % index_directory)
    
    if tfidf_examples > 0:
        # step #1 create TFIDF model
        f_set = FeatureSet(index_directory, ft_terms_tf=True)
        f_set.fit_tfidf_from_index(training_examples=tfidf_examples)
        f_set.save_tfidf_model("model_tfidf_1.pkl")
        f_set = None
        gc.collect()

    # step #2 create PCA+TFIDF model
    if pca_examples > 0:
        f_set = FeatureSet(index_directory, ft_terms_tfidf=True)
        f_set.load_tfidf_model("model_tfidf_1.pkl")
        f_set.fit_pca_from_index(training_examples=pca_examples)
        f_set.save_pca_model("model_pca_1.pkl")
        f_set = None
        gc.collect()
