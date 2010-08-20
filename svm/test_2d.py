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



# Get samples in a uniform range...
a = numpy.random.random((250,2))
b = numpy.random.random((250,2))
c = numpy.random.random((250,2))
d = numpy.random.random((250,2))

a[:,:] *= 0.5
b[:,:] *= 0.5
c[:,:] *= 0.5
d[:,:] *= 0.5

b[:,0] += 0.5
c[:,1] += 0.5
d[:,0] += 0.5
d[:,1] += 0.5

# Transform the samples, remap them back to [0..1] for conveniance...
for dMat in [a,b,c,d]:
  for i in xrange(dMat.shape[0]):
    ang = math.atan2(dMat[i,1]-0.5,dMat[i,0]-0.5)
    rad = math.sqrt((dMat[i,0]-0.5)**2.0 + (dMat[i,1]-0.5)**2.0)
    ang += rad*5.0
    dMat[i,0] = 0.5 + rad * math.cos(ang)
    dMat[i,1] = 0.5 + rad * math.sin(ang)

min0 = min((a[:,0].min(),b[:,0].min(),c[:,0].min(),d[:,0].min()))
min1 = min((a[:,1].min(),b[:,1].min(),c[:,1].min(),d[:,1].min()))
max0 = max((a[:,0].max(),b[:,0].max(),c[:,0].max(),d[:,0].max()))
max1 = max((a[:,1].max(),b[:,1].max(),c[:,1].max(),d[:,1].max()))

for dMat in [a,b,c,d]:
  dMat[:,0] = (dMat[:,0] - min0) / (max0-min0)
  dMat[:,1] = (dMat[:,1] - min1) / (max1-min1)

# Render them out to an image for visualisation...
img = numpy.zeros((400,400,3),dtype=numpy.float_)
colSet = [(255.0,0.0,0.0),(0.0,255.0,0.0),(0.0,0.0,255.0),(255.0,255.0,0.0)]
for dMat,col in [(a,colSet[0]),(b,colSet[1]),(c,colSet[2]),(d,colSet[3])]:
  for i in xrange(dMat.shape[0]):
    x = int(dMat[i,0]*399.0)
    y = int(dMat[i,1]*399.0)
    for ci in xrange(3):
      img[x,y,ci] = col[ci]
cv.SaveImage('test_2d_samples.png',array2cv(img))

# Create the dataset object...
ds = svm.Dataset()
ds.addMatrix(a,['a']*a.shape[0])
ds.addMatrix(b,['b']*b.shape[0])
ds.addMatrix(c,['c']*c.shape[0])
ds.addMatrix(d,['d']*d.shape[0])



# Create the parameters set - use the standard model selection set...
ps = svm.ParamsSet(True)
#ps.addPoly()



# Train a model, print out basic info...
p = ProgBar()
mm = svm.MultiModel(ps,ds,callback=p.callback,pool=True)
del p


print 'a success =',len(filter(lambda x:x=='a', mm.multiClassify(a))) / float(a.shape[0])
print 'b success =',len(filter(lambda x:x=='b', mm.multiClassify(b))) / float(b.shape[0])
print 'c success =',len(filter(lambda x:x=='c', mm.multiClassify(c))) / float(c.shape[0])
print 'd success =',len(filter(lambda x:x=='d', mm.multiClassify(d))) / float(d.shape[0])

print 'models ='
for params in mm.paramsList():
  print params



# Output a classification region map...
img = numpy.zeros((400,400,3),dtype=numpy.float_)
x,y = numpy.meshgrid(xrange(400),xrange(400))
pixels = numpy.column_stack((x.flatten(),y.flatten()))
vectors = numpy.asfarray(pixels)/400.0
labels = mm.multiClassify(vectors)
labToCol = {'a':colSet[0],'b':colSet[1],'c':colSet[2],'d':colSet[3]}

for i in xrange(len(labels)):
  c = labToCol[labels[i]]
  x = int(pixels[i,0])
  y = int(pixels[i,1])
  for ci in xrange(3):
    img[x,y,ci] = c[ci]

cv.SaveImage('test_2d_regions.png',array2cv(img))
