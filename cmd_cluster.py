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

from n7.model import FeatureSet
from n7.cluster import Clusterer

if __name__ == "__main__":

    index_directory = sys.argv[1]
    logging.basicConfig(level=logging.INFO)
    logging.info("CLUSTERING")

    
    
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
    
    
    X0 = FeatureSet.load_fm()
    X1, Y1 = f_set.load_from_csv([
        "n7-data/labeled/data_aol.csv",
        "n7-data/labeled/data_google.csv",
        "n7-data/labeled/data_mcdonalds.csv",
    ])
    
    logging.info("LOADED FEATURE MATRIX %dx%d" % (X0.shape[0], X0.shape[1]))
    
    cl = Clusterer()
    
    cl.draw(X0, X1, Y1)
    # cl.kmean(X0, 20)
    
    


