#!/usr/bin/env python
# coding: utf-8

# Copyright (C) Vladimir M. Zaytsev <zaytsev@usc.edu>
# Author: Vladimir M. Zaytsev <zaytsev@usc.edu>
# URL: <http://zvm.me/>
# For more information, see README.md
# For license information, see LICENSE

import sys
import logging

from n7.data import Loader
from n7.search import TextIndex, Searcher

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    tw_index = TextIndex("n7-data/index")
    loader = Loader()

    # tw_index.learn_terms(open(sys.argv[1], "r"))
    # tw_index.load_terms()
    # tw_index.create_text_index()

    tw_search = Searcher(tw_index)
#    model = FeatureMatrix(tw_search)

#    model.from_vectors(tw_search.iterate())



    # plist = tw_search.search("#hate")
    # tw_search.pprint_plist(plist)

    #
    # bad_tweets = set()
    #
    # for term in loader.bad_words():
    #     tweets = tw_search.search(term)
    #     if tweets is not None:
    #         bad_tweets |= tweets
    #     print term, "=>", len(tweets)
    #
    # print len(bad_tweets)
    #
    # tw_search.pprint_plist(bad_tweets)