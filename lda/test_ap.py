#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



# Tests with the Associated press data set provided with the original implimentation of LDA, and spits out the same data for comparison.

import time
import math
import os.path

import lda
print 'Algorithm =',lda.getAlgorithm()

from utils.prog_bar import ProgBar



# Verify existance of ap directory...
if not os.path.isdir('ap'):
  print "You must download the associated press dataset from http://www.cs.princeton.edu/~blei/lda-c/ and put it into the directory 'ap' within the folder you are running this test program."
  sys.exit(1)



# Fitter parameters object...
params = lda.Params()
params.setRuns(8)
params.setSamples(10)
params.setBurnIn(1000)
params.setLag(100)



# Create a corpus and load in the documents...
print 'Loading documents...'
c = lda.Corpus(100)

f = open('ap/ap.dat','r')
for line in f:
  bits = line.split()[1:]
  
  dic = dict()
  for bit in bits:
    d = bit.split(':')
    dic[int(d[0])] = int(d[1])
  
  doc = lda.Document(dic)
  c.add(doc)
print 'Loaded\n'

print 'Topics =', c.maxTopicIdent()+1
print 'Documents =', c.maxDocumentIdent()+1
print 'Unique words =', c.maxWordIdent()+1
print 'Total words =', c.wordCount()
print ''



# Train...
print 'Trainning...'
p = ProgBar()
c.fit(params, p.callback)
del p



# Iterate the topics and splat out their top 20 words into a text file...
print 'Saving top words...'
words = open('ap/vocab.txt','r').readlines()

f = open('ap/results-gibbs.txt','w')
for topic in c.topicList():
  f.write('Topic '+str(topic.getIdent())+':\n')
  topWords = topic.getTopWords()
  for i in xrange(20):
    f.write(str(-math.log(topic.probWord(topWords[i])))+': '+words[topWords[i]])
  f.write('\n')
