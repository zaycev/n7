#!/usr/bin/env python
# coding: utf-8

# Copyright (C) Vladimir M. Zaytsev <zaytsev@usc.edu>
# Author: Vladimir M. Zaytsev <zaytsev@usc.edu>
# URL: <http://zvm.me/>
# For more information, see README.md
# For license information, see LICENSE


class Loader(object):

    def __init__(self, root_dir="n7-data"):
        self.root_dir = root_dir

    def bad_words(self, add_hashtags=True):
        fl = open("%s/badwords.txt" % self.root_dir, "r")
        bad_words = fl.read().split("\n")
        for i in xrange(len(bad_words)):
            bad_words[i] = bad_words[i].replace("*", "")
            bad_words[i] = bad_words[i].replace(" ", "")

        if add_hashtags:
            hashtags = []
            for term in bad_words:
                hashtags.append("#%s" % term)
            bad_words += hashtags
        return set(bad_words)


