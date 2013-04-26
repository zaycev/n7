#!/usr/bin/env python
# coding: utf-8

# Copyright (C) USC Information Sciences Institute
# Author: Vladimir M. Zaytsev <zaytsev@usc.edu>
# URL: <http://nlg.isi.edu/>
# For more information, see README.md
# For license information, see LICENSE

from django.conf.urls import url
from django.conf.urls import patterns
from django.shortcuts import redirect


urlpatterns = patterns(
    "",
    url(r"^$", "n7.web.n7.views.demo", name="demo"),
    url(r"^triples/$", "n7.web.n7.views.trainer", name="trainer"),
    url(r"^novels/$", "n7.web.n7.views.trainer_add", name="trainer_post"),
)

