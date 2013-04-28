#!/usr/bin/env python
# coding: utf-8

# Copyright (C) Vladimir M. Zaytsev <zaytsev@usc.edu>
# Author: Vladimir M. Zaytsev <zaytsev@usc.edu>
# URL: <http://zvm.me/>
# For more information, see README.md
# For license information, see LICENSE

import gc
import logging
import numpy as np
import cPickle as pickle

from n7 import N7_DATA_DIR
from n7.data import Loader
from n7.search import Searcher
from n7.search import TextIndex
from n7.data import TwitterTextUtil

from emoticons import Sad_RE
from emoticons import Happy_RE

from sklearn.externals import joblib
from sklearn.decomposition import KernelPCA
from sklearn.feature_extraction.text import TfidfTransformer

class FeatureSet(object):

    def __init__(self, index_dir,

                 allowed_terms=None,           # list of allowed terms which will be used (need for testing)
                 disallowed_terms=None,        # list of disallowed terms which will be ignored

                 ft_number_of_words=False,      # use number of regular words as feature
                 ft_number_of_hash_tags=False,  # use number of hash-tags as feature
                 ft_number_of_user_names=False, # use number of twitter user names as feature
                 ft_number_of_bad_words=False, # use number of bad words as feature
                 ft_number_of_links=False,     #
                 ft_number_of_nes=False,       # use number of named entities as feature
                 ft_number_of_punct=False,     #
                 ft_emoticons=False,           #
                 ft_total_hate_score=False,    # use total hate score as feature
                 ft_terms_binary=False,        # use vector space model with binary function as feature
                 ft_terms_tf=False,            # use vector space model with frequency function as feature
                 ft_terms_tfidf=False,         # use vector space model with tfidf function as feature
                 ft_scale=False,

                 terms_max_df=0.5,             # specifies max document frequency in feature selection (normalized)
                 terms_min_df=50,              # specifies min document frequency in feature selection

                 tfidf_model=None,             #

                 pca=False,                    # apply pca to output vector
                 pca_model=None,               #

                 data_n7_dir=N7_DATA_DIR,      #

                 dtype=np.float32,             #

                 verbose=False):                #

        logging.info("CREATING MODEL")

        self.tweet_id_index_map = dict()
        self.index_tweet_id_map = dict()

        self.index = TextIndex(index_dir)
        self.full_index = TextIndex(index_dir)
        self.full_index.load_terms(0, 1.0)
        self.searcher = Searcher(self.index, terms_min_df, terms_max_df)
        self.verbose = verbose

        self.allowed_terms = allowed_terms
        self.disallowed_terms = disallowed_terms

        self.ft_number_of_words = ft_number_of_words
        self.ft_number_of_hash_tags = ft_number_of_hash_tags
        self.ft_number_of_user_names = ft_number_of_user_names
        self.ft_number_of_bad_words = ft_number_of_bad_words
        self.ft_number_of_nes = ft_number_of_nes
        self.ft_number_of_links = ft_number_of_links
        self.ft_total_hate_score = ft_total_hate_score
        self.ft_terms_binary = ft_terms_binary
        self.ft_terms_tf = ft_terms_tf
        self.ft_terms_tfidf = ft_terms_tfidf
        self.ft_scale = ft_scale
        self.ft_number_of_punct = ft_number_of_punct
        self.ft_emoticons = ft_emoticons

        self.terms_max_df = terms_max_df
        self.terms_min_df = terms_min_df

        self.data_n7_dir = data_n7_dir

        self.tfidf_model = tfidf_model

        self.pca = pca
        self.pca_model = pca_model

        self.dtype = dtype

        self.twitter = TwitterTextUtil()

        if self.allowed_terms:
            allowed_terms = dict()
            for term in self.allowed_terms:
                if term in self.index.term_id_map:
                    allowed_terms[self.index.term_id_map[term]] = term
            self.allowed_terms = allowed_terms
            if self.verbose:
                logging.info("ALLOWED TERMS: %r" % self.allowed_terms)

        # create <term id> :-> <vector index> map
        if ft_terms_binary or ft_terms_tf or ft_terms_tfidf:
            for term_id in self.index.id_term_map.iterkeys():

                if self.allowed_terms is not None:
                    if term_id not in self.allowed_terms:
                        continue

                if self.disallowed_terms is not None:
                    term = self.index.term_id_map.get(term_id)
                    if term in self.disallowed_terms:
                        continue

                new_index_value = len(self.tweet_id_index_map)
                self.index_tweet_id_map[new_index_value] = term_id
                self.tweet_id_index_map[term_id] = new_index_value
                if self.verbose:
                    print "ADDED: %d as %d" % (term_id, new_index_value)

            if self.verbose:
                print self.tweet_id_index_map

            print "\tMODEL: %d terms" % len(self.tweet_id_index_map)

        loader = Loader(data_n7_dir)

        if self.ft_number_of_bad_words:
            self.bad_words = loader.bad_words(add_hashtags=False)
            print "\tMODEL: %d bad words" % len(self.bad_words)

    def text_to_vector(self, text):
        tokens = self.index.tokenize(text)
        if self.verbose:
            print tokens
        return self.terms_to_vector(text, tokens)

    def terms_to_vector(self, text, terms):
        term_ids = []
        outputs = []

        # PREPROCESSING

        for term in terms:
            term_id = self.index.term_id_map.get(term)
            if term_id is not None:
                term_ids.append(term_id)
        if self.verbose:
            print term_ids

        if self.allowed_terms:
            term_ids = filter(lambda term_id: term_id in self.allowed_terms, term_ids)
            if self.verbose:
                print term_ids

        # COMPUTING FEATURES

        if self.ft_terms_binary:
            bin_vector = self.__ft_bin_vector__(term_ids, terms, scale=self.ft_scale)
            if self.pca:
                print self.pca_model
                bin_vector = np.asarray(self.pca_model.transform(bin_vector)).reshape(-1)
            if self.verbose:
                print "bin_vector", bin_vector
            outputs.append(bin_vector)

        if self.ft_terms_tf:
            tf_vector = self.__ft_tf_vector__(term_ids, terms, scale=self.ft_scale)
            if self.pca:
                tf_vector = np.asarray(self.pca_model.transform(tf_vector)).reshape(-1)
            if self.verbose:
                print "tf_vector", tf_vector
            outputs.append(tf_vector)

        if self.ft_terms_tfidf:
            tfifd_vector = self.__ft_tfidf_vector__(term_ids, terms, scale=self.ft_scale)
            if self.pca:
                tfifd_vector = np.asarray(self.pca_model.transform(tfifd_vector)).reshape(-1)
            if self.verbose:
                print "tfifd_vector", tfifd_vector
            outputs.append(tfifd_vector)

        if self.ft_number_of_words:
            number_of_words = self.__ft_number_of_words__(term_ids, terms, scale=self.ft_scale)
            if self.verbose:
                print "number_of_words", number_of_words
            outputs.append(number_of_words)

        if self.ft_number_of_hash_tags:
            number_of_hash_tags = self.__ft_number_of_hash_tags__(term_ids, terms, scale=self.ft_scale)
            if self.verbose:
                print "number_of_hash_tags", number_of_hash_tags
            outputs.append(number_of_hash_tags)

        if self.ft_number_of_user_names:
            number_of_user_names = self.__ft_number_of_user_names__(term_ids, terms, scale=self.ft_scale)
            if self.verbose:
                print "number_of_user_names", number_of_user_names
            outputs.append(number_of_user_names)

        if self.ft_number_of_links:
            number_of_links = self.__ft_number_of_links__(term_ids, terms, scale=self.ft_scale)
            if self.verbose:
                print "number_of_links", number_of_links
            outputs.append(number_of_links)

        if self.ft_number_of_bad_words:
            number_of_bad_words = self.__ft_number_of_bad_words__(term_ids, terms, scale=self.ft_scale)
            if self.verbose:
                print "number_of_bad_words", number_of_bad_words
            outputs.append(number_of_bad_words)

        if self.ft_number_of_punct:
            number_of_punct = self.__ft_number_of_punct__(term_ids, terms, scale=self.ft_scale)
            if self.verbose:
                print "number_of_punct", number_of_punct
            outputs.append(number_of_punct)

        if self.ft_emoticons:
            emoticons_vector = self.__ft_emoticons_vector__(term_ids, terms, scale=self.ft_scale)
            if self.verbose:
                print "emoticons_vector", emoticons_vector
            outputs.append(emoticons_vector)

        outputs = np.concatenate(outputs)
        if self.verbose:
            print outputs

        return outputs

    def __scale_array__(self, array):
        return 1 - 1 / (array + 1)

    def __ft_emoticons_vector__(self, term_ids, terms, scale=False):
        vector = np.zeros(2, dtype=self.dtype)
        for term in terms:
            if Sad_RE.match(term):
                vector[0] += 1
            if Happy_RE.match(term):
                vector[0] += 1
        return vector

    def __ft_bin_vector__(self, term_ids, terms, scale=False):
        vector = np.zeros(len(self.tweet_id_index_map), dtype=self.dtype)
        for term_id in term_ids:
            term_index = self.tweet_id_index_map.get(term_id)
            if term_index is not None:
                vector[term_index] = 1
        return vector

    def __ft_tf_vector__(self, term_ids, terms, scale=False):
        vector = np.zeros(len(self.tweet_id_index_map), dtype=self.dtype)
        for term_id in term_ids:
            term_index = self.tweet_id_index_map.get(term_id)
            if term_index is not None:
                vector[term_index] += 1
        return vector

    def __ft_tfidf_vector__(self, term_ids, terms, scale=False):
        vector = np.zeros(len(self.tweet_id_index_map), dtype=self.dtype)
        for term_id in term_ids:
            term_index = self.tweet_id_index_map.get(term_id)
            if term_index is not None:
                vector[term_index] += 1
        vector = self.tfidf_model.transform(vector).toarray()[0]
        return vector

    def __ft_number_of_words__(self, term_ids, terms, scale=False):
        nw = np.zeros(1, dtype=self.dtype)
        for term in terms:
            is_word = True
            if self.twitter.is_hashtag(term):
                is_word = False
            if self.twitter.is_link(term):
                is_word = False
            if self.twitter.is_username(term):
                is_word = False
            if self.twitter.is_punct(term):
                is_word = False
            if self.verbose:
                if is_word:
                    print "%s is a word" % term
                else:
                    print "%s is not a word" % term
            if is_word:
                nw[0] += 1
        return self.__scale_array__(nw) if scale else nw

    def __ft_number_of_hash_tags__(self, term_ids, terms, scale=False):
        nw = np.zeros(1, dtype=self.dtype)
        for term in terms:
            if self.twitter.is_hashtag(term):
                nw[0] += 1
        return self.__scale_array__(nw) if scale else nw

    def __ft_number_of_user_names__(self, term_ids, terms, scale=False):
        nw = np.zeros(1, dtype=self.dtype)
        for term in terms:
            if self.twitter.is_username(term):
                nw[0] += 1
        return self.__scale_array__(nw) if scale else nw

    def __ft_number_of_links__(self, term_ids, terms, scale=False):
        nw = np.zeros(1, dtype=self.dtype)
        for term in terms:
            if self.twitter.is_link(term):
                if self.verbose:
                    print "%s is is link" % term
                nw[0] += 1
        return self.__scale_array__(nw) if scale else nw

    def __ft_number_of_punct__(self, term_ids, terms, scale=False):
        nw = np.zeros(1, dtype=self.dtype)
        for term in terms:
            if self.twitter.is_punct(term):
                nw[0] += 1
        return self.__scale_array__(nw) if scale else nw

    def __ft_number_of_bad_words__(self, term_ids, terms, scale=False):
        nw = np.zeros(1, dtype=self.dtype)
        for term in terms:
            for bad_w in self.bad_words:
                if len(bad_w) > 3:
                    if bad_w in term:
                        if self.verbose:
                            print "%s is bad word" % term
                        nw[0] += 1
                        break
                else:
                    if bad_w == term:
                        if self.verbose:
                            print "%s is bad word" % term
                        nw[0] += 1
                        break
        return self.__scale_array__(nw) if scale else nw
        
    def fit_pca(self, X, n_components=128, kernel="rbf"):
        self.pca_model = KernelPCA(n_components=n_components, kernel=kernel)
        logging.info("FITTING PCA MODEL FROM %d EXAMPLES" % X.shape[0])
        self.pca_model.fit(X)
        logging.info("FITTING DONE")
        
    def fit_pca_from_index(self, training_examples=10, n_components=128, kernel="rbf"):
        X = self.fm_from_index(training_examples)
        self.fit_pca(X, n_components, kernel)
        
    def fit_tfidf(self, X):
        self.tfidf_model = TfidfTransformer()
        logging.info("FITTING TFIDF MODEL FROM %d EXAMPLES" % X.shape[0])
        self.tfidf_model.fit(X)
        logging.info("FITTING DONE")
        
    def fit_tfidf_from_index(self, training_examples=10):
        X = self.fm_from_index(training_examples)
        self.fit_tfidf(X)
        
    def save_pca_model(self, file_path=None):
        if file_path is None:
            file_path = "%s/models/model_tfidf.pkl" % self.data_n7_dir
        else:
            file_path = "%s/models/%s" % (self.data_n7_dir, file_path)
        joblib.dump(self.pca_model, file_path, compress=9)

    def load_pca_model(self, file_path=None):
        if file_path is None:
            file_path = "%s/models/model_tfidf.pkl" % self.data_n7_dir
        else:
            file_path = "%s/models/%s" % (self.data_n7_dir, file_path)
        self.pca_model = joblib.load(file_path)
        logging.info("LOADED PCA MODEL %r" % self.pca_model)

    def save_tfidf_model(self, file_path=None):
        if file_path is None:
            file_path = "%s/models/model_tfidf.pkl" % self.data_n7_dir
        else:
            file_path = "%s/models/%s" % (self.data_n7_dir, file_path)
        joblib.dump(self.tfidf_model, file_path, compress=9)

    def load_tfidf_model(self, file_path=None):
        if file_path is None:
            file_path = "%s/models/model_tfidf.pkl" % self.data_n7_dir
        else:
            file_path = "%s/models/%s" % (self.data_n7_dir, file_path)
        self.tfidf_model = joblib.load(file_path)
        logging.info("LOADED TFIDF MODEL %r" % self.tfidf_model)
        
    def fm_from_index(self, training_examples=10):
        v_size = len(self.text_to_vector(""))
        X = np.zeros((training_examples, v_size), dtype=self.dtype)
        ni = 0
        for tweet_id, tweet_vector in self.searcher.iterate():
            tokens = [self.full_index.id_term_map[term_id] for term_id in tweet_vector]
            f_vect = self.terms_to_vector(None, tokens)
            X[ni:] = f_vect
            ni += 1
            if ni >= training_examples:
                break
        return X
        
        
        '''
        X = []
        ni = 0
        for tweet_id, tweet_vector in self.searcher.iterate():
            tokens = [self.full_index.id_term_map[term_id] for term_id in tweet_vector]
            f_vect = self.terms_to_vector(None, tokens)
            X.append(f_vect)
            ni += 1
            if ni >= training_examples:
                break
        logging.info("LOADED %d EXAMPLES" % ni)
        X = np.array(X)
        gc.collect()
        self.fit_tfidf(X)
        '''

    def info(self):
        pass




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
            # print tweet_vector
            self.tweet_id_index_map[tweet_id] = new_index
            self.index_tweet_id_map[new_index] = tweet_id
            x = new_index
            for term_id in tweet_vector:
                y = self.searcher.index.id_index_map[term_id]
                # print term_id, "=>", y
                self.X[x, y] += 1
            ni += 1
            if ni >= n:
                break
        # print self.X

    def apply_tfids(self, norm="l2"):
        tfids = TfidfTransformer(norm=norm)
        self.X = tfids.fit_transform(self.X).toarray()

    def apply_kpca(self, n_components=128, kernel="rbf"):
        kpca = KernelPCA(n_components=n_components, kernel=kernel)
        self.X = kpca.fit_transform(self.X)

    def apply_pca(self, n_components=128):
        pca = PCA(n_components=n_components)
        self.X = pca.fit_transform(self.X)

    def project2d(self, kernel="rbf"):
        kpca = KernelPCA(n_components=2, kernel=kernel)
        return kpca.fit_transform(self.X)
