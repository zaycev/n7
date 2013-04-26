#!/usr/bin/env python
# coding: utf-8

# Copyright (C) Vladimir M. Zaytsev <zaytsev@usc.edu>
# Author: Vladimir M. Zaytsev <zaytsev@usc.edu>
# URL: <http://zvm.me/>
# For more information, see README.md
# For license information, see LICENSE

import numpy as np
import scipy as sp

from scipy import sparse


class FeatureMatrix(object):

    def __init__(self, searcher):
        self.X = None
        self.searcher = searcher
        self.tweet_id_index_map = None
        self.index_tweet_id_map = None

    def from_vectors(self, i_vectors, n=100):
        ni = 0
        m = len(self.searcher.index.id_term_map)
        self.tweet_id_index_map = dict()
        self.index_tweet_id_map = dict()
        self.X = np.zeros((n, m))
        print n, m
        for tweet_id, tweet_vector in i_vectors:
            tweet_vector = self.searcher.term_filter(tweet_vector)
            new_index = len(self.tweet_id_index_map)
            self.tweet_id_index_map[tweet_id] = new_index
            self.index_tweet_id_map[new_index] = tweet_id
            x = new_index
            for term_id in tweet_vector:
                y = self.searcher.index.id_index_map[term_id]
                self.X[x, y] += 1
            ni += 1
            if ni >= n:
                break