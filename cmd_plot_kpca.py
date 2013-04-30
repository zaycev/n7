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

import matplotlib.pyplot as plot
from n7.model import FSetLoader


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logging.info("CLUSTERING")

    pca_model_name = sys.argv[1] if len(sys.argv) > 1 else "model_kpca.pkl.ev"

    loader = FSetLoader()
    Y = loader.load_model(pca_model_name)
    X = range(0, len(Y))
    
    plot.plot(X, Y, marker="o")
    
    plot.show()
    


