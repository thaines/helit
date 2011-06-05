#! /usr/bin/env python
# -*- coding: utf-8 -*-


# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



# Test taken from the Gibbs sampling paper. Note that it uses openCV to write output images.

import time

import numpy
import numpy.random

import cv
from utils import cvarray

import lda

from utils.prog_bar import ProgBar



# Grid is defined as 5x5, with word identifiers...
grid = numpy.array([[ 0,  1,  2,  3,  4],
                    [ 5,  6,  7,  8,  9],
                    [10, 11, 12, 13, 14],
                    [15, 16, 17, 18, 19],
                    [20, 21, 22, 23, 24]], dtype=numpy.int_)

# 10 topics exist - the vertical lines and the horizontal lines.
# Setup the actual topic word multinomials...
topics = []
for r in xrange(5):
  t = numpy.zeros(25,dtype=numpy.float_)
  for c in xrange(5):
    t[r*5 + c] = 1.0
  t /= t.sum()
  topics.append(t)

for c in xrange(5):
  t = numpy.zeros(25,dtype=numpy.float_)
  for r in xrange(5):
    t[r*5 + c] = 1.0
  t /= t.sum()
  topics.append(t)



# Create the lda object, generate the documents...
vlda = lda.VLDA(10, 25)

alphaDist = numpy.ones(10,dtype=numpy.float_)

inputImageSet = []
for d in xrange(1000): # Trainning set only.
  # Create a multinomial distribution to draw topics from...
  topicDist = numpy.random.mtrand.dirichlet(alphaDist)
  
  # Draw how many to get from each topic...
  samples = numpy.random.multinomial(100,topicDist)
  
  # Iterate each topic and draw words from it...
  words = numpy.zeros(25,dtype=numpy.int_)
  for t in xrange(10):
    words += numpy.random.multinomial(samples[t],topics[t])
  inputImageSet.append(words)
  
  # Convert the word counts into a dictionary...
  dic = dict()
  for i in xrange(25):
    if words[i]!=0:
      dic[i] = words[i]
  
  # Create the document...
  vlda.add(dic)



# Save out the input documents for confirmation (50x20 grid)...
docImageSet = []
for words in inputImageSet:
  image = numpy.asfarray(words)
  image *= 255.0/image.max()
  image = numpy.reshape(image,(5,5))
  image = numpy.repeat(numpy.repeat(image,5,axis=0),5,axis=1)
  image = numpy.append(image,numpy.atleast_2d(numpy.zeros(image.shape[1])),axis=0)
  image = numpy.append(image,numpy.atleast_2d(numpy.zeros(image.shape[0])).T,axis=1)
  docImageSet.append(image)

docVertSet = []
for i in xrange(50):
  docVertSet.append(numpy.vstack(docImageSet[i*20:(i+1)*20]))
docSet = numpy.hstack(docVertSet)
img = cvarray.array2cv(docSet)
cv.SaveImage('test_grid_docs.png',img)



# Train...
print 'Trainning...'
#p = ProgBar()
#passes = vlda.solve()
#del p
passes = vlda.solveHuman()
print 'Took %i passes'%passes



# Generate an image of the final distributions associated with the learned documents...
# Get pixel values...
tImages = []
for topic in xrange(vlda.numTopics()):
  # Get distribution...
  dist = vlda.getBeta(topic)
  
  # Reshape...
  dist = numpy.reshape(dist,(5,5))
  
  # Make brightest value one...
  dist *= 255.0/dist.max()
  
  # Store...
  tImages.append(dist)


# Make 'em bigger, insert borders between them in the list...
sImages = []
border = numpy.zeros(16*5,dtype=numpy.float_)

for image in tImages:
  nImg = numpy.repeat(numpy.repeat(image,16,axis=0),16,axis=1)
  sImages.append(nImg)
  sImages.append(border)

# Glue them together...
final = numpy.vstack(sImages[:-1])

# Save via openCV, which requires some conversion...
img = cvarray.array2cv(final)
cv.SaveImage('test_grid_topics.png',img)
