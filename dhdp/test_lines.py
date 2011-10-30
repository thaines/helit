#! /usr/bin/env python

# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import math
import random
import os
import shutil

import numpy
import numpy.random
import cv

import dhdp
print 'Algorithm =', dhdp.getAlgorithm()


from utils.prog_bar import ProgBar
from utils.cvarray import *
from utils import cvarray



# Simple Dual-HDP test, as taken from the relevant paper. Uses a 5x5 grid with 10 topics - the horizontal and vertical lines of grid locations. Has clustering, specifically documents contain either horizontal or vertical lines, not both.



# Parameters...
samplesPerDoc = 100 # Number of samples in a document.
documentsToTrain = 50 # Number of documents for training. (50 in paper)
docGridWidth = 10 # Width of the grid into which documents are written to disk
gridScale = 5
gridSpace = 2



# Code to sample a document
def sampleTopic(topic):
  r = random.randrange(5)
  if topic<5: # Horizontal.
    return topic*5 + r
  else: # Vertical.
    return r*5 + (topic-5)


def sampleDocument():
  # Create the multinomial to draw topics from...
  multi = numpy.random.mtrand.dirichlet(numpy.ones(5))
  if random.choice([True,False]):
    multi = numpy.hstack((numpy.zeros(5),multi))
  else:
    multi = numpy.hstack((multi,numpy.zeros(5)))

  # Draw sample counts...
  counts = numpy.random.multinomial(samplesPerDoc,multi)

  # Generate and return the resulting dictionary...
  ret = dict()
  for t in xrange(10):
    for _ in xrange(counts[t]):
      word = sampleTopic(t)
      if word in ret: ret[word] += 1
      else: ret[word] = 1
  return ret


# Convert a document as a dictionary to a grid shaped numpy array...
def docDicToArray(docDic):
  ret = numpy.zeros((5,5), dtype=numpy.float_)
  for key,value in docDic.iteritems():
    x = key//5
    y = key%5
    ret[x,y] += value
  ret /= ret.max()
  return ret



# Create a corpus...
c = dhdp.Corpus()
c.setWordCount(5*5)



# Fill in documents...
docs = []
for _ in xrange(documentsToTrain):
  docDic = sampleDocument()
  docs.append(docDicToArray(docDic))
  doc = dhdp.Document(docDic)
  c.add(doc)



# (Re)create the output directory...
try: shutil.rmtree('test_lines')
except: pass
os.makedirs('test_lines')



# Save the documents to an image...
docs = map(lambda d: numpy.repeat(numpy.repeat(d,gridScale,axis=1),gridScale,axis=0),docs)

rows = []
for i in xrange(documentsToTrain//docGridWidth):
  row = docs[i*docGridWidth:(i+1)*docGridWidth]
  rowExt = []
  for r in row:
    rowExt.append(r)
    rowExt.append(numpy.zeros((gridSpace,gridScale*5), dtype=numpy.float_))
  rowExt = rowExt[:-1]
  rows.append(numpy.vstack(rowExt))

stack = []
for r in rows:
  stack.append(r)
  stack.append(numpy.zeros((r.shape[0],gridSpace), dtype=numpy.float_))
stack = stack[:-1]
docImage = numpy.hstack(stack).T * 255.0
img = cvarray.array2cv(docImage)
cv.SaveImage('test_lines/docs.png',img)



# Train...
params = dhdp.Params()
params.runs = 1
params.samples = 1
#params.burnIn = 10000
#c.setOneCluster(True)
#c.setCalcBeta(True)


print 'Fitting model...'
p = ProgBar()
model = c.sampleModel(params,p.callback)
del p

#model.bestSampleOnly()


print 'Discovered topics:', model.getSample(0).getTopicCount()
print 'Discovered clusters:', model.getSample(0).getClusterCount()
if c.getCalcBeta(): print 'Beta:', model.getSample(0).getBeta()



# Visualise the topics...
sam = model.getSample(0)

for t in xrange(sam.getTopicCount()):
  multi = sam.getTopicMultinomial(t)
  dicForm = dict()
  for i in xrange(multi.shape[0]): dicForm[i] = multi[i]
  topicGrid = docDicToArray(dicForm)
  topicGrid = numpy.repeat(numpy.repeat(topicGrid,gridScale,axis=1),gridScale,axis=0)
  img = cvarray.array2cv(topicGrid*255.0)
  cv.SaveImage('test_lines/topic_%i.png'%t,img)



# Provide some information on the clusters...
out = open('test_lines/clusters.txt','w')
out.write('cluster draw conc = %f\n\n'%sam.getClusterDrawConc())

for c in xrange(sam.getClusterCount()):
  out.write('cluster %i:\n'%c)
  out.write('weight = %i\n'%sam.getClusterDrawWeight(c))
  out.write('inst count = %i\n'%sam.getClusterInstCount(c))
  out.write('conc = %f\n'%sam.getClusterInstConc(c))
  for ci in xrange(sam.getClusterInstCount(c)):
    out.write('inst %i: {topic = %i, count = %i}\n'%(ci, sam.getClusterInstTopic(c,ci), sam.getClusterInstWeight(c,ci)))
  out.write('\n')
out.close()



# Provide a visualisation of the clusters...
cImg = []
maxWeight = max(map(lambda c: sam.getClusterDrawConc() + sam.getClusterDrawWeight(c), xrange(sam.getClusterCount())))

topicWidth = 10
topicHeight = 50
weightHeight = 10
spacer = 5

twl = []
twlMax = 0.0

for c in xrange(sam.getClusterCount()):
  tWeight = numpy.zeros(sam.getTopicCount(),dtype=numpy.float_)

  wSum = sam.getTopicConc()
  for t in xrange(sam.getTopicCount()):
    tWeight[t] += sam.getClusterInstConc(c) * sam.getTopicUseWeight(t)
    wSum += sam.getTopicUseWeight(t)
  tWeight /= wSum

  for ci in xrange(sam.getClusterInstCount(c)):
    tWeight[sam.getClusterInstTopic(c,ci)] += sam.getClusterInstWeight(c,ci)

  twl.append(tWeight)
  twlMax = max((twlMax,tWeight.max()))

for c in xrange(sam.getClusterCount()):
  tWeight = twl[c] / twlMax
  
  img = numpy.zeros((topicHeight+weightHeight+spacer,topicWidth*sam.getTopicCount(),3), dtype=numpy.float_)

  offset = int(topicWidth*sam.getTopicCount()*(sam.getClusterDrawConc() + sam.getClusterDrawWeight(c))/maxWeight)
  img[topicHeight:topicHeight+weightHeight,0:offset,1] = 1.0

  for t in xrange(sam.getTopicCount()):
    col = 0 if (t%2==0) else 2
    h = int(tWeight[t]*topicHeight)
    img[topicHeight-h:topicHeight,t*topicWidth:(t+1)*topicWidth,col] = 1.0
  
  cImg.append(img)

cluImg = numpy.vstack(cImg) * 255.0
cv.SaveImage('test_lines/clusters.png', cvarray.array2cv(cluImg))
