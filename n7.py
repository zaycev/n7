#!/usr/bin/env python
# coding: utf-8

# Copyright (C) Vladimir M. Zaytsev <zaytsev@usc.edu>
# Author: Vladimir M. Zaytsev <zaytsev@usc.edu>
# URL: <http://zvm.me/>
# For more information, see README.md
# For license information, see LICENSE

import csv
import sys
import logging
import leveldb
import numencode
import twokenize
import collections
import marshal as pickle


class TextIndex(object):
    
    def __init__(self, data_dir):
        self.terms_store_loc = "%s/terms_store.db" % data_dir
        self.tweet_index_loc = "%s/tweet_index.db" % data_dir
        self.tweet_store_loc = "%s/tweet_vector.db" % data_dir
        self.term_id_map = None
        self.id_term_map = None
        self.id_term_freq = None

    def load_terms(self, min_threshold=10, max_threshold=0.5):
        pass

    def write_tweet_vectors(self, tweet_vectors):
        w_batch = leveldb.WriteBatch()
        for tweet_id, tweet_sp_vector in tweet_vectors:
            vector_key = numencode.encode_uint(tweet_id)
            vector_value = pickle.dumps(tweet_sp_vector)
            w_batch.Put(vector_key, vector_value)
        tweet_store = leveldb.LevelDB(self.tweet_store_loc)
        tweet_store.Write(w_batch)
        logging.info("FLUSH %d VECTORS" % len(tweet_vectors))

    def write_terms(self, term_id_map, term_freq):
        w_batch = leveldb.WriteBatch()
        for term, term_id in term_id_map.iteritems():
            term_key = numencode.encode_uint(term_id)
            term_value = pickle.dumps((term, term_freq[term_id]))
            w_batch.Put(term_key, term_value)
        term_store = leveldb.LevelDB(self.terms_store_loc)
        term_store.Write(w_batch)
        logging.info("FLUSH %d TERMS" % len(term_id_map))

    def learn_terms(self, tweets_file_object, cache_size=100000):
        reader = csv.reader(tweets_file_object, delimiter=",", quotechar="\"")
        term_freq = collections.Counter()
        term_id_map = dict()
        tweet_vectors = []
        for row in reader:
            tweet_id = int(row[0])
            tweet_text = row[-1]
            terms = [t.lower() for t in twokenize.tokenize(tweet_text)]
            tweet_sp_vector = []
            for term in terms:
                if term not in term_id_map:
                    term_id = len(term_id_map)
                    term_id_map[term] = term_id
                else:
                    term_id = term_id_map[term]
                term_freq[term_id] += 1
                tweet_sp_vector.append(term_id)
            tweet_vectors.append((tweet_id, tweet_sp_vector))
            if len(tweet_vectors) >= cache_size:
                self.write_tweet_vectors(tweet_vectors)
                tweet_vectors = []
        self.write_tweet_vectors(tweet_vectors)
        self.write_terms(term_id_map, term_freq)


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    tw_index = TextIndex("n7-data/index")
    tw_index.learn_terms(sys.stdin)