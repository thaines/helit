#! /usr/bin/env python

# Copyright 2016 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import sys
import numpy

from corpus import Corpus



## Setup stuff...
pages = 1
if len(sys.argv)>1:
  pages = int(sys.argv[1])
words = pages * 200 # Seems to be a good guess at a low average of words per page.

corpus = Corpus()



# Shuffle the number of blocks we have, so we can draw random chunks...
chunk_size = 1000
roar = numpy.arange(len(corpus))
numpy.random.shuffle(roar)



# The scoring function needs weights - create them...
letter_weight = numpy.asarray(corpus.get_counts(), dtype=numpy.float32)
letter_weight /= letter_weight.max()
letter_weight[:32] = 0.0
letter_weight[127:] = 0.0

adj_weight = numpy.asarray(corpus.get_adj(), dtype=numpy.float32)
adj_weight /= adj_weight.max()
if ' ' in corpus.get_adj_index():
  loc = corpus.get_adj_index().index(' ')
  adj_weight[loc,loc] = 0.0



# Random pangram, because why not?..
f = open('data/pangrams.txt')
pangrams = filter(lambda l: len(l)>=26, f.readlines())
f.close()

print 'A pangram:'
print
print numpy.random.choice(pangrams)
print



# Define a scoring function for a block, given the numbers we have obtained thus far...
def score(count, adj, block):
  if block.words==0:
    return 0.0

  # Get the potential difference made by the block...
  b_count = numpy.zeros(256, dtype=numpy.float32)
  b_adj = numpy.zeros(corpus.get_adj().shape, dtype=numpy.float32)
  block.stats(b_count, b_adj, corpus.get_adj_index())
  
  # Calculate the score for individual letters...
  score = 32.0 * ((b_count * letter_weight) / (count*count + 1.0)).sum()
  
  # Sum in the score for adjacencies...
  score += ((b_adj * adj_weight) / (adj*adj + 1.0)).sum()
  
  # Return the score divided by the number of words, but reduce the score a bit if its less than 10 words...
  ret = score / block.words
  if block.words<10:
    ret *= (1.0 + 2.0*(block.words/10.0)) / 3.0
  return ret



# Keep drawing blocks, by picking the best in a chunk, until we exhaust our word limit...
chunk_index = 0

count = numpy.zeros(corpus.get_counts().shape, dtype=numpy.int32)
adj  = numpy.zeros(corpus.get_adj().shape, dtype=numpy.int32)

while words>0:
  # Score all entries...
  scores = numpy.empty(chunk_size, dtype=numpy.float32)
  base = chunk_index * chunk_size
  for i in xrange(chunk_size):
    scores[i] = score(count, adj, corpus[roar[base+i]])
  
  # Pick the largest...
  index = roar[base + scores.argmax()]
  
  # Update the stats...
  corpus[index].stats(count, adj, corpus.get_adj_index())
  
  # Print it out...
  print 'Excerpt %i: (Block %i, from %s)'%(chunk_index+1, index, corpus[index].attribution)
  print
  print ' '.join(corpus[index].text.split())
  print
  print
  
  # Move on...
  words -= corpus[index].words
  chunk_index += 1
