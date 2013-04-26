#!/usr/bin/env python
# coding: utf-8

# Copyright (C) Vladimir M. Zaytsev <zaytsev@usc.edu>
# Author: Vladimir M. Zaytsev <zaytsev@usc.edu>
# URL: <http://zvm.me/>
# For more information, see README.md
# For license information, see LICENSE


# The model for estimating hate intensity for given word w_j ussing seed words:
#   v(w_j) = a_0 + \sum_{i=1}^N{a_i*w(w_i)f(d_{i,j})}
# * w_j         - is the word we mean to characterize
# * w_1 .. w_N  - seed words
# * v(w_i)      - hate intensity for the seed words
# * d_{i,j}     - semantic similarity between words i and j
# * f(.)        - some function, for example linear, log, exp, sqrt
# The model is described in this paper (Malandrakis, 2011):
# http://www.telecom.tuc.gr/~potam/preprints/conf/11_INTERSPEECH_text_affect.pdf

import csv
import sys
import nltk
import pickle
import logging
import sqlite3
import collections
import numpy as np

from scipy import stats
from twokenize import tokenize
from nltk.corpus import wordnet as wn


SQL_FIND = \
"""
SELECT text
FROM n7_tweet, n7_index
WHERE
        n7_tweet.id=n7_index.id
    AND n7_index.token=?;
"""

def retrive_texts(sql_cursor, keyword):
    rows = sql_cursor.execute(SQL_FIND, (keyword, ))
    return [r[0] for r in rows]


def sset_sim(ssset_1, ssset_2):
    sim = []
    if len(ssset_1) == 0 or len(ssset_2) == 0:
        return 0.0
    for synset_1 in ssset_1:
        for synset_2 in ssset_2:
            s1, s2, s3 = synset_1.path_similarity(synset_2), \
                         synset_1.wup_similarity(synset_2), \
                         1.0
            if s1 > 0 and s2 > 0 and s3 > 0:
                new_sim = stats.hmean((s1, s2, s3))
                if new_sim > 0:
                    sim.append(new_sim)
        
    return stats.hmean(sim) if len(sim) > 0 else 0.0

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    
    

    logging.info("loading seeds")
    seeds = []
    seedreader = csv.reader(open(sys.argv[1], "r"), delimiter=',')
    for row in seedreader:
        seeds.append((row[0], float(row[1])))
    
    logging.info("loading database and index")
    sql = sqlite3.connect(sys.argv[2])
    sql_cursor = sql.cursor()
    logging.info("loading word frequencies")
    fdic = pickle.load(open(sys.argv[3], "rb"))
    
    th1 = int(sys.argv[4])
    th2 = len(fdic) / 3
    
    sym = sys.argv[5]
    
    seed_words = set([word for word, intense in seeds])
    candidates = set()
    similarities = dict()
    
    logging.info("selecting candidates")
    for token, freq in fdic.most_common():
        if th1 < freq < th2:
            candidates.add(token.encode("utf-8"))
    
    tk_sysnsets = [(token, wn.synsets(token)) for token in candidates]
    
    logging.info("estimating similarities")
    for seed, intense in seeds:
        similarities[seed] = dict()
        sd_synset = wn.synsets(seed)
        for token, tk_synset in tk_sysnsets:
            sim = sset_sim(tk_synset, sd_synset)
            similarities[seed][token] = sim
        similarities[seed][seed] = 1.0
    
    logging.info("assigning indexes")
    token_index = dict()
    
    candidates |= seed_words
    for token in candidates:
        token_index[token] = len(token_index)
    
    #print similarities
    
    for seed, seed_tokens in similarities.iteritems():
        print seed
        items = seed_tokens.items()
        items.sort(key=lambda item: -item[1])
        for token, sim in items:
            print token_index[token], token, sim
        print
        print
    
    #print token_index