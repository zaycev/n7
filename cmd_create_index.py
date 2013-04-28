#!/usr/bin/env python
# coding: utf-8

# Copyright (C) Vladimir M. Zaytsev <zaytsev@usc.edu>
# Author: Vladimir M. Zaytsev <zaytsev@usc.edu>
# URL: <http://zvm.me/>
# For more information, see README.md
# For license information, see LICENSE

#   SYNOPSIS:
#       python cmd_create_index.py <path to tweets csv> <path to index directory>


import sys
import logging


from n7.search import TextIndex

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    tweets_file_path = sys.argv[1]
    index_directory = sys.argv[2]
    
    logging.info("CREATING INDEX")
    logging.info("TWEETS CORPUS: %s" % tweets_file_path)
    logging.info("OUTPUT DIRECTORY: %s" % index_directory)
    
    tw_index = TextIndex(index_directory)

    tw_index.learn_terms(open(tweets_file_path, "r"))
    tw_index.load_terms()
    tw_index.create_text_index()

    logging.info("DONE")