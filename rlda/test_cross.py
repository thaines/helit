#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2010 Tom SF Haines

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.



import math
import random

import numpy
import cv

import rlda
print 'Algorithm =', rlda.getAlgorithm()


import sys
sys.path.append(sys.path[0]+'/..')
from utils.prog_bar import ProgBar
from utils.cvarray import *



# Problem - we have 1 word and 25 identifiers. The identifiers form a 5x5 grid in which we draw a cross from 2 topics that generate the one word. We then solve with 1 word, 4 regions.

docWordCount = 100
docCount = 100
docRendWidth = 10
docRendHeight = docCount/docRendWidth



# Define a function to generate a dictionary to be input to a Document, using the input rules...
def makeDoc(ratio = None):
  ret = dict()
  if ratio==None: ratio = random.random()

  global docWordCount
  for i in xrange(docWordCount):
    if random.random()<0.0: # Increase to try with noise.
      key = (random.randrange(5)*5 + random.randrange(5), 0)
    elif random.random()<ratio:
      key = (1*5 + random.randrange(5), 0)
    else:
      key = (random.randrange(5)*5 + 1, 0)

    if key in ret: ret[key] += 1
    else: ret[key] = 1

  return ret



# Generate a list of documents...
docList = map(lambda _: makeDoc(), xrange(docCount))



# Create the corpus and parameters...
c = rlda.Corpus(3,2)
c.setIdentWordCounts(25,1)
for dic in docList:
  doc = rlda.Document(dic)
  c.add(doc)

params = rlda.Params()
params.setRuns(32)



# Solve the corpus...
p = ProgBar()
c.fit(params,p.callback)
del p

ir = c.getIR()
wrt = c.getWRT()



# Output a visual representation of the documents...
docRend = []
maxValue = 0.0
for doc in docList:
  rend = numpy.zeros((5,5),dtype=numpy.float_)
  for key,value in doc.iteritems():
    x = key[0]%5
    y = key[0]/5
    rend[y,x] += value
    maxValue = max((maxValue,float(value)))

  docRend.append(rend)

docRend = map(lambda x: x * 255.0/maxValue,docRend)
docRend = map(lambda x: numpy.repeat(numpy.repeat(x,5,axis=0),5,axis=1), docRend)
docVertRend = []
for i in xrange(docRendWidth):
  b = i*docRendHeight
  v = numpy.vstack(docRend[b:b+docRendHeight])
  docVertRend.append(v)
docAllRend = numpy.hstack(docVertRend)
img = array2cv(docAllRend)
cv.SaveImage('test_cross_docs.png',img)



# Output the regions...
maxData = c.getIR().max()
for r in xrange(c.getRegionCount()):
  data = c.getIR()[:,r]
  data *= 255.0/maxData
  data = numpy.reshape(data,(5,5))
  data = numpy.repeat(numpy.repeat(data,25,axis=0),25,axis=1)
  img = array2cv(data)
  cv.SaveImage('test_cross_region_%i.png'%r,img)



# Output the topics...
maxData = 0.0
topicRendList = []
for t in xrange(c.getTopicCount()):
  data = numpy.zeros(25,dtype=numpy.float_)

  for r in xrange(c.getRegionCount()):
    data += c.getIR()[:,r] * c.getWRT()[0,r,t]

  maxData = max((maxData,data.max()))
  topicRendList.append(data)

for t,data in enumerate(topicRendList):
  data *= 255.0/maxData
  data = numpy.reshape(data,(5,5))
  data = numpy.repeat(numpy.repeat(data,25,axis=0),25,axis=1)
  img = array2cv(data)
  cv.SaveImage('test_cross_topic_%i.png'%t,img)



# Test the document fitting...
for i in xrange(9):
  ratio = float(i)/8.0
  doc = rlda.Document(makeDoc(ratio))
  doc.fit(ir,wrt,params)
  model = doc.getModel()
  model = model/model.sum()
  print 'document fitting test (actual ratio, fitted model)', ratio, model
