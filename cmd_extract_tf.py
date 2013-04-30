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

from n7.model import FeatureSet


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    index_directory = sys.argv[1]

    if len(sys.argv) > 2:
        dataset_size = int(sys.argv[2])
    else:
        dataset_size = 10 ** 5

    matrix_name = sys.argv[3] if len(sys.argv) > 3 else "X_tf.pkl"

    logging.info("EXTRACTING FEATURE MATRIX")
    logging.info("INDEX DIRECTORY: %s" % index_directory)

    f_set = FeatureSet(index_directory,
                       ft_number_of_words=False,
                       ft_number_of_hash_tags=False,
                       ft_number_of_user_names=False,
                       ft_number_of_bad_words=False,
                       ft_number_of_links=False,
                       ft_number_of_punct=False,
                       ft_emoticons=False,
                       ft_terms_tf=True,
                       pca=False,
                       ft_scale=True)

    X = f_set.fm_from_index(training_examples=dataset_size)
    
    f_set.save_fm(X, matrix_name, sparse=True)

