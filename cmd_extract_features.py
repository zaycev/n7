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

from n7 import N7_DATA_DIR
from n7.model import FeatureSet
from sklearn.externals import joblib

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    index_directory = sys.argv[1]
    
    if len(sys.argv) > 2:
        dataset_size = int(sys.argv[2])
    else:
        dataset_size = 10 ** 5
    
    logging.info("EXTRACTING FEATURE MATRIX")
    logging.info("INDEX DIRECTORY: %s" % index_directory)
    

    f_set = FeatureSet(index_directory,
                       ft_number_of_words=True,
                       ft_number_of_hash_tags=True,
                       ft_number_of_user_names=True,
                       ft_number_of_bad_words=True,
                       ft_number_of_links=True,
                       ft_number_of_punct=True,
                       ft_emoticons=True,
                       ft_terms_tfidf=True,
                       pca=True,
                       ft_scale=True)
    f_set.load_tfidf_model("model_tfidf_1.pkl")
    f_set.load_pca_model("model_pca_1.pkl")
    
    X = f_set.fm_from_index(training_examples=dataset_size)
    
    f_set.save_fm(X)

