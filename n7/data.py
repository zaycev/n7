#!/usr/bin/env python
# coding: utf-8

# Copyright (C) Vladimir M. Zaytsev <zaytsev@usc.edu>
# Author: Vladimir M. Zaytsev <zaytsev@usc.edu>
# URL: <http://zvm.me/>
# For more information, see README.md
# For license information, see LICENSE

import re


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


RE_URL = re.compile(
    r"""((?:[a-z][\w-]+:(?:/{1,3}|[a-z0-9%])|www\d{0,3}[.]|[a-z0-9.\-]"""
    r"""+[.‌​][a-z]{2,4}/)(?:[^\s()<>]+|(([^\s()<>]+|(([^\s()<>]+)))*))+"""
    r"""(?:(([^\s()<>]+|(‌​([^\s()<>]+)))*)|[^\s`!()[]{};:'".,<>?«»“”‘’]"""
    r"""))""", re.DOTALL)

RE_PUNCT = re.compile("[\.,\?!{}()\[\]:;¿¡]")


class TwitterTextUtil(object):
    re_url = RE_URL
    re_puct = RE_PUNCT

    def __init__(self):
        pass

    def is_link(self, token):
        if self.re_url.match(token):
            return True
        return False

    def is_username(self, token):
        if token and len(token) >= 2 and token[0] == "@":
            return True
        return False

    def is_hashtag(self, token):
        if token and len(token) >= 2 and token[0] == "#":
            return True
        return False

    def is_punct(self, token):
        if token and self.re_puct.match(token):
            return True
        return False

    def get_links(self, tokens):
        return filter(self.is_link, tokens)

    def get_usernames(self, tokens):
        return filter(self.is_username, tokens)

    def get_hashtags(self, tokens):
        return filter(self.is_hashtag, tokens)
