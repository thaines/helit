#! /usr/bin/env python

# Copyright 2016 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import random
import numpy

from corpus import Corpus



corpus = Corpus(noisy = True)

print 'Corpus loaded, contains %i blocks' % len(corpus)
print



print 'Counts:'
counts = corpus.get_counts()

for i in xrange(128):
  if counts[i]!=0:
    print chr(i), ':', counts[i]

print 



print 'Adjacencies:'
adj = corpus.get_adj()
adj_index = corpus.get_adj_index()

for i in xrange(32):
  a, b = numpy.unravel_index(numpy.argmax(adj), adj.shape)
  print 'Pair %i: %s%s (With %i)'%(i+1, adj_index[a], adj_index[b], adj[a,b])
  adj[a,b] = 0
  
print



print 'Random block:'

i = random.randrange(len(corpus))

block = corpus[i]

print block.text
print
print '   -', block.attribution
