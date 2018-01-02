#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

if len(sys.argv) < 2:
    print "usage: {} method-name < input.txt > output.txt".format(sys.argv[0])
    exit(1)

method = sys.argv[1]

for idx, line in enumerate(sys.stdin):
    print "{}\t{}\t{}".format(method, idx+1, line.strip())
