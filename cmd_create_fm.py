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
import numpy as np

from n7.model import FSetLoader
from n7.model import FeatureSet


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    index_dir = sys.argv[1]
    input_matrix_name = sys.argv[2] if len(sys.argv) > 2 else "X_tfidf.pkl"
    pca_model_name = sys.argv[2] if len(sys.argv) > 2 else "model_kpca.pkl"

    loader = FSetLoader()
    f_set = FeatureSet(index_dir,
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
    f_set.load_tfidf_model("model_tfidf.pkl")
    f_set.load_pca_model("model_kpca.pkl")



    pca = loader.load_model(pca_model_name)
    X0 = loader.load_model(input_matrix_name)
    logging.info("APPLYING PCA ON %dx%d examples" % (X0.shape[0], X0.shape[1]))
    X0 = pca.transform(X0)
    logging.info("TRANSFOMATION DONE %r" % X0)
    Y0 = [-1] * X0.shape[0]

    X1, Y1 = f_set.load_from_csv([
        "n7-data/labeled/data_aol.csv",
        "n7-data/labeled/data_google.csv",
        "n7-data/labeled/data_mcdonalds.csv",
        "n7-data/labeled/data_badwords_100.csv",
        "n7-data/labeled/data_microsoft.csv",
        "n7-data/labeled/data_timewarner.csv",
    
    ])
    
    print "X0 shape", X0.shape
    print "X1 shape", X1.shape
    
    X = np.concatenate(X0, X1)
    Y = Y0 + Y1

    loader.save_model((X, Y), "XY.pkl")
    

    


