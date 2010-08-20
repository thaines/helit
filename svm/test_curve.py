#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2010 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import math

import numpy
import cv

import svm

from utils.cvarray import *
from utils.prog_bar import ProgBar



# Get samples in the uniform [0..1]X[0..1]...
samples = numpy.random.random((400,2))

# Label them depending on which side of a sin curve they are on...
labels = []
for i in xrange(samples.shape[0]):
  if samples[i,0] < 0.5+0.3*math.cos(6.0 * (samples[i,1]-0.5)):
    labels.append('a')
  else:
    labels.append('b')

# Make the dataset...
ds = svm.Dataset()
ds.addMatrix(samples,labels)

# Render them out to an image for visualisation...
img = numpy.zeros((400,400,3),dtype=numpy.float_)
colSet = [(255.0,64.0,64.0),(0.0,255.0,255.0)]
for i,label in enumerate(labels):
  x = int(samples[i,0]*399.0)
  y = int(samples[i,1]*399.0)
  c = colSet[0] if label=='a' else colSet[1]
  for ci in xrange(3):
    img[x,y,ci] = c[ci]
cv.SaveImage('test_curve_samples.png',array2cv(img))

# Create the parameters set - use the standard model selection set...
ps = svm.ParamsSet(True)



# Train a model, print out basic info...
p = ProgBar()
mm = svm.MultiModel(ps,ds,callback=p.callback)
del p


guess = mm.multiClassify(samples)
print 'Success rate =',len(filter(lambda x:x==True,map(lambda a,b:a==b, guess,labels))) / float(samples.shape[0])

print 'models ='
for params in mm.paramsList():
  print params



# Output a correct/incorrect map...
img = numpy.zeros((400,400,3),dtype=numpy.float_)
for i,label in enumerate(labels):
  x = int(samples[i,0]*399.0)
  y = int(samples[i,1]*399.0)
  c = (0.0,255.0,0.0) if label==guess[i] else (0.0,0.0,255.0)
  for ci in xrange(3):
    img[x,y,ci] = c[ci]
cv.SaveImage('test_curve_correct.png',array2cv(img))



# Output a classification region map...
img = numpy.zeros((400,400,3),dtype=numpy.float_)
x,y = numpy.meshgrid(xrange(400),xrange(400))
pixels = numpy.column_stack((x.flatten(),y.flatten()))
vectors = numpy.asfarray(pixels)/400.0
labels = mm.multiClassify(vectors)
labToCol = {'a':colSet[0],'b':colSet[1]}

for i in xrange(len(labels)):
  c = labToCol[labels[i]]
  x = int(pixels[i,0])
  y = int(pixels[i,1])
  for ci in xrange(3):
    img[x,y,ci] = c[ci]

cv.SaveImage('test_curve_regions.png',array2cv(img))



# Output the locations and weights of the support vectors...
sv = mm.getModel('a','b')[0].getSupportVectors()
sw = mm.getModel('a','b')[0].getSupportWeights()
maxSW = sw.max()
img = numpy.zeros((400,400,3),dtype=numpy.float_)
for i in xrange(sv.shape[0]):
  x = int(sv[i,0]*399.0)
  y = int(sv[i,1]*399.0)

  img[x,y,0] = 0.0 if sw[i]<0.0 else 255.0
  img[x,y,1] = 255.0
  img[x,y,2] = abs(255.0*(sw[i]/maxSW))

cv.SaveImage('test_curve_support.png',array2cv(img))
