#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



# Tests with a very simple toy model - 2 topics, 4 words...

import time

import numpy
import numpy.random
import scipy

import lda
print 'Algorithm =',lda.getAlgorithm()

from utils.prog_bar import ProgBar



# Algorithm parameters...
topicCount = 2
docCount = 40
testCount = 10
wordCount = 80

# Fitter parameters object...
params = lda.Params()
params.setRuns(8)
params.setSamples(10)
params.setBurnIn(1000)
params.setLag(100)



# Multinomial distributions for each of the topics word generation...
topicA = numpy.array([4,2,1,0],dtype=numpy.float_)
topicA /= topicA.sum()

topicB = numpy.array([0,1,2,4],dtype=numpy.float_)
topicB /= topicB.sum()



# Make a corpus...
c = lda.Corpus(topicCount)


# Generate documents and stuff them in...
testDocs = []
for i in xrange(docCount+testCount):
  # Probability of word comming from topicA...
  probA = numpy.random.random()
  
  # Generate a dictionary of words...
  d = dict()
  tac = numpy.random.binomial(wordCount,probA)
  wfa = numpy.random.multinomial(tac,topicA)
  wfb = numpy.random.multinomial(wordCount-tac,topicB)
  for j in xrange(topicA.shape[0]):
    d[j] = wfa[j] + wfb[j]
  
  # Make the document, add to corpus or test set...
  doc = lda.Document(d)
  doc.topicA = probA # For verification
  if i<docCount:
    c.add(doc)
  else:
    testDocs.append(doc)

# Add an unusual document...
d = dict()
d[0] = wordCount
doc = lda.Document(d)
doc.topicA = -1.0
c.add(doc)



# Train...
print 'Trainning...'
p = ProgBar()
c.fit(params,p.callback)
del p



# Print out the document multinomial topic point estimates...
topicsWords = c.topicsWords()
print 'documents = {actual,estimate,nll}:'
for doc in c.documentList():
  nll = doc.negLogLikelihood(topicsWords)
  print '[',doc.topicA,(1.0-doc.topicA),']',doc.getModel(), nll
print ''

# Print out ground truth topic multinomial distributions...
print 'topics = {actual}:'
print topicA
print topicB
print ''

# Print out the topic multinomial topic point estimates...
print 'topics = {estimate}:'
for topic in c.topicList():
  print topic.getNormModel()
print ''


# Print out the test set with their estimates...
print 'test documents = {actual,estimate,nll}:'
for doc in testDocs:
  doc.fit(topicsWords,c.alpha,params)
  nll = doc.negLogLikelihood(topicsWords)
  print '[',doc.topicA,(1.0-doc.topicA),']',doc.getModel(), nll
print ''
