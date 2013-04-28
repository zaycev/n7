#!/usr/bin/env python
# coding: utf-8

# Copyright (C) Vladimir M. Zaytsev <zaytsev@usc.edu>
# Author: Vladimir M. Zaytsev <zaytsev@usc.edu>
# URL: <http://zvm.me/>
# For more information, see README.md
# For license information, see LICENSE

#   SYNOPSIS:
#       python cmd_model_index.py <path to index directory>


import gc
import csv
import sys
import logging


from n7 import N7_DATA_DIR
from n7.model import FeatureSet


FIT_MODELS = True

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    index_directory = sys.argv[1]
    test_tweets = open("%s/test.tweets.csv" % N7_DATA_DIR, "r")

    reader = csv.reader(test_tweets, delimiter=",", quotechar="\"")

    logging.info("CREATING MODEL")
    logging.info("TWEETS CORPUS: %s" % test_tweets)
    logging.info("INDEX DIRECTORY: %s" % index_directory)

    if FIT_MODELS:
        # step #1 create TFIDF model
        f_set = FeatureSet(index_directory, ft_terms_tf=True)
        f_set.fit_index(training_examples=1000)
        f_set.save_tfidf_model("tfidf_1.pck")
        f_set = None
        gc.collect()

        # step #2 create PCA+TFIDF model
        f_set = FeatureSet(index_directory, ft_terms_tfidf=True)
        f_set.load_tfidf_model("tfidf_1.pck")
        f_set.fit_index(training_examples=1000)
        f_set.save_pca_model("pca_1.pck")
        f_set = None
        gc.collect()

    f_set = FeatureSet(index_directory,
                       ft_number_of_words=True,
                       ft_number_of_hash_tags=True,
                       ft_number_of_user_names=True,
                       ft_number_of_bad_words=True,
                       ft_number_of_links=True,
                       ft_number_of_punct=False,
                       ft_emoticons=True,
                       ft_terms_tfidf=True,
                       pca=True,
                       ft_scale=True,
                       allowed_terms=["idea", "money", "remember"])

    f_set.load_tfidf_model("tfidf_1.pck")
    f_set.load_pca_model("pca_1.pck")
    f_set.text_to_vector("money ideas #fuck them all !!!! :-) #test #test @user ... . . .")
