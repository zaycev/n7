#!/usr/bin/env python
# coding: utf-8

# Copyright (C) Vladimir M. Zaytsev <zaytsev@usc.edu>
# Author: Vladimir M. Zaytsev <zaytsev@usc.edu>
# URL: <http://zvm.me/>
# For more information, see README.md
# For license information, see LICENSE

#   SYNOPSIS:
#       python cmd_model_index.py <path to index directory> <dataset size for TFIDF> <dataset size for PCA>


import gc
import sys
import logging

from n7.ssl import fit_predict
from n7.model import FSetLoader
from sklearn.decomposition import PCA


if __name__ == "__main__":

    loader = FSetLoader()

