#!/usr/bin/env python
# coding: utf-8

# Copyright (C) Vladimir M. Zaytsev <zaytsev@usc.edu>
# Author: Vladimir M. Zaytsev <zaytsev@usc.edu>
# URL: <http://zvm.me/>
# For more information, see README.md
# For license information, see LICENSE

import csv
import lz4
import nltk
import array
import logging
import leveldb
import cPickle
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
        self.total_freq = 0
        self.id_index_map = None

    def encode_plist(self, plist):
        plist.sort()
        numencode.delta_encode(plist)
        plist_array = array.array("L", plist)
        plist_str = plist_array.tostring()
        return lz4.compressHC(plist_str)

    def decode_plist(self, plist_data):
        plist_str = lz4.decompress(plist_data)
        plist_array = array.array("L")
        plist_array.fromstring(plist_str)
        numencode.delta_decode(plist_array)
        return list(plist_array)

    def load_terms(self, min_threshold=10, max_threshold=0.5):
        term_store = leveldb.LevelDB(self.terms_store_loc)
        self.term_id_map = dict()
        self.id_term_map = dict()
        self.id_term_freq = collections.Counter()
        self.total_freq = 0
        self.id_index_map = dict()
        loaded = 0
        total = 0
        for _, value in term_store.RangeIter():
            total += 1
            _, term_freq = pickle.loads(value)
            self.total_freq += term_freq
        for key, value in term_store.RangeIter():
            term_id = numencode.decode_uint(key)
            term, term_freq = pickle.loads(value)
            if term_freq > min_threshold and float(term_freq) / float(self.total_freq) < max_threshold:
                self.id_term_map[term_id] = term
                self.term_id_map[term] = term_id
                self.id_term_freq[term_id] = term_freq
                self.id_index_map[term_id] = len(self.id_index_map)
                loaded += 1
        logging.info("LOADED %d of %d TERMS, TOTAL DF %d" % (loaded, total, self.total_freq))
        # for term_id, term_freq in self.id_term_freq.most_common(100):
        #     print term_id, self.id_term_map[term_id], term_freq

    def update_index(self, posting_lists):
        tweet_index = leveldb.LevelDB(self.tweet_index_loc)
        w_batch = leveldb.WriteBatch()
        prev_sz = 0.0
        total_sz = 0.0
        for term_id, term_plist in posting_lists.iteritems():
            term_key = numencode.encode_uint(term_id)
            try:
                plist_data = tweet_index.Get(term_key)
                plist = self.decode_plist(plist_data)
                prev_sz += len(plist)
                plist.extend(term_plist)
            except KeyError:
                plist = term_plist
            total_sz += len(plist)
            plist_data = self.encode_plist(plist)
            w_batch.Put(term_key, plist_data)
        tweet_index.Write(w_batch, sync=True)
        logging.info("UPDATED %d POSTING LISTS, AVG SIZE %f (PREV: %f)" % (
            len(posting_lists),
            total_sz / len(posting_lists),
            prev_sz / len(posting_lists)
        ))

    def create_text_index(self, cache_size=500000):
        tweet_store = leveldb.LevelDB(self.tweet_store_loc)
        posting_lists = dict()
        cache_counter = 0
        for tweet_key, tweet_vector_value in tweet_store.RangeIter():
            tweet_id = numencode.decode_uint(tweet_key)
            tweet_bag = pickle.loads(tweet_vector_value)
            for term_id in tweet_bag:
                if term_id in self.id_term_freq:
                    if term_id in posting_lists:
                        posting_lists[term_id].append(tweet_id)
                    else:
                        posting_lists[term_id] = [tweet_id]
            cache_counter += 1
            if cache_counter > cache_size:
                self.update_index(posting_lists)
                posting_lists = dict()
                cache_counter = 0
        self.update_index(posting_lists)

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

    def learn_terms(self, tweets_file_object, cache_size=500000):
        reader = csv.reader(tweets_file_object, delimiter=",", quotechar="\"")
        term_freq = collections.Counter()
        term_id_map = dict()
        tweet_vectors = []
        for row in reader:
            tweet_id = int(row[0])
            tweet_text = row[-1]
            terms = [t.lower().encode("utf-8") for t in twokenize.tokenize(tweet_text)]
            tweet_sp_vector = []
            counted_ids = []
            for term in terms:
                if term not in term_id_map:
                    term_id = len(term_id_map)
                    term_id_map[term] = term_id
                else:
                    term_id = term_id_map[term]
                if term_id not in counted_ids:
                    term_freq[term_id] += 1
                    counted_ids.append(term_id)
                tweet_sp_vector.append(term_id)
            tweet_vectors.append((tweet_id, tweet_sp_vector))
            if len(tweet_vectors) >= cache_size:
                self.write_tweet_vectors(tweet_vectors)
                tweet_vectors = []
        self.write_tweet_vectors(tweet_vectors)
        self.write_terms(term_id_map, term_freq)


class Searcher(object):

    def __init__(self, index_object):
        self.index = index_object
        self.index.load_terms(500, 0.4)
        self.vectors = leveldb.LevelDB(self.index.tweet_store_loc)
        self.plists = leveldb.LevelDB(self.index.tweet_index_loc)

    def search(self, term):
        if isinstance(term, basestring):
            if isinstance(term, unicode):
                term = term.encode("utf-8")
            term_id = self.index.term_id_map.get(term)
        else:
            term_id = term
        if term_id is None:
            return set()
        term_key = numencode.encode_uint(term_id)
        plist_data = self.plists.Get(term_key)
        plist = self.index.decode_plist(plist_data)
        return set(plist)

    def iterate(self):
        for tweet_key, tweet_data in self.vectors.RangeIter():
            tweet_id = numencode.decode_uint(tweet_key)
            tweet_vector = pickle.loads(tweet_data)
            yield tweet_id, tweet_vector

    def term_filter(self, term_ids):
        return filter(lambda term_id: term_id in self.index.id_term_map, term_ids)

    def pprint_plist(self, plist):
        tagger = cPickle.load(open("models/tagger.pickle", "r"))
        for tweet_id in plist:
            tweet_key = numencode.encode_uint(tweet_id)
            tweet_vec = pickle.loads(self.vectors.Get(tweet_key))
            tweet_tokens = [self.index.id_term_map[tid] for tid in tweet_vec]
            print tagger.tag(tweet_tokens)
            print