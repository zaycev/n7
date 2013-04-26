#!/usr/bin/env python
# coding: utf-8

# Copyright (C) Vladimir M. Zaytsev <zaytsev@usc.edu>
# Author: Vladimir M. Zaytsev <zaytsev@usc.edu>
# URL: <http://zvm.me/>
# For more information, see README.md
# For license information, see LICENSE

import csv
import sys
import nltk
import logging

from twokenize import tokenize
from nltk.collocations import BigramCollocationFinder

def all_tokens(tweetreader):
    i = 0
    for r in tweetreader:
        i += 1
        tokens = tokenize(r[-1])
        for t in tokens:
            yield t
        if i >= 50000:
            return

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logging.info("start processing")

    #csv_out = open("%s.csv" % sys.argv[1], "w")
    #pkl_out = open("%s.pkl" % sys.argv[1], "w")
    
    tweetreader = csv.reader(sys.stdin, delimiter=',', quotechar='"')
    
    
    tokens = all_tokens(tweetreader)
    
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(tokens)
    
    print finder.nbest(bigram_measures.pmi, 100)
    
    
    
    
    