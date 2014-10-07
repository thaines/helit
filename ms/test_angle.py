#! /usr/bin/env python

# Copyright 2014 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import numpy
import numpy.random

import cv
from utils.cvarray import *
from utils.prog_bar import ProgBar

from ms import MeanShift



# Tests the conversion capability with the angle convertor - runs through basic things that the MeanShift object can do.



# Create three clusters by drawing pairs of angles, using a wrapped Gaussian over radians...
clu1 = numpy.random.multivariate_normal([numpy.pi*0.5, 0.0], [[0.1, 0.0],[0.0,0.1]], 128)
clu2 = numpy.random.multivariate_normal([-numpy.pi*0.5, numpy.pi*0.5], [[0.1, 0.0],[0.0,0.1]], 128)
clu3 = numpy.random.multivariate_normal([numpy.pi*0.5, numpy.pi*0.5], [[0.1, 0.0],[0.0,0.1]], 128)

data = numpy.concatenate((clu1, clu2, clu3), axis=0)



# Visualise by drawing lines - kinda crazy...
img = numpy.zeros((256, 1024), dtype=numpy.float32)
for row in data:
  for y in xrange(img.shape[0]):
    t = float(y) / float(img.shape[0]-1)
    x = row[0] * t + row[1] * (1.0-t)
    if x<-numpy.pi: x += numpy.pi
    if x>numpy.pi: x -= numpy.pi
    x = int(img.shape[1] * (x + numpy.pi) / (numpy.pi * 2.0))
    
    img[y, x] += 1.0

img *= 255.0 / img.max()
cv.SaveImage('angle_input.png', array2cv(img))



# Try using mean shift to cluster...
ms = MeanShift()
ms.set_data(data, 'df', None, 'AA')
ms.set_kernel('composite(2:fisher(16.0),2:fisher(16.0))')

modes, indices = ms.cluster()
print 'Found %i modes' % modes.shape[0]



# Another crazy visualisation, this time the modes...
img = numpy.zeros((256, 1024), dtype=numpy.float32)
for row in modes:
  for y in xrange(img.shape[0]):
    t = float(y) / float(img.shape[0]-1)
    x = row[0] * t + row[1] * (1.0-t)
    if x<-numpy.pi: x += numpy.pi
    if x>numpy.pi: x -= numpy.pi
    x = int(img.shape[1] * (x + numpy.pi) / (numpy.pi * 2.0))
    
    img[y, x] += 1.0

img *= 255.0 / img.max()
cv.SaveImage('angle_modes.png', array2cv(img))



# Plot the pdf, different visualisation style...
img = numpy.zeros((64, 64), dtype=numpy.float32)

for y in xrange(img.shape[0]):
  for x in xrange(img.shape[1]):
    ang_x = 2.0 * numpy.pi * (x / float(img.shape[1]-1)) - numpy.pi
    ang_y = 2.0 * numpy.pi * (y / float(img.shape[0]-1)) - numpy.pi
    img[y,x] = ms.prob(numpy.array([ang_x, ang_y]))

img *= 255.0 / img.max()
cv.SaveImage('angle_pdf.png', array2cv(img))



# Try drawing...
img = numpy.zeros((256, 1024), dtype=numpy.float32)
for row in ms.draws(512):
  for y in xrange(img.shape[0]):
    t = float(y) / float(img.shape[0]-1)
    x = row[0] * t + row[1] * (1.0-t)
    if x<-numpy.pi: x += numpy.pi
    if x>numpy.pi: x -= numpy.pi
    x = int(img.shape[1] * (x + numpy.pi) / (numpy.pi * 2.0))
    
    img[y, x] += 1.0

img *= 255.0 / img.max()
cv.SaveImage('angle_draw.png', array2cv(img))



# New data sets to multiply together, single angles this time...
data_a = numpy.random.normal(numpy.pi*90.0/180.0, numpy.pi*15.0/180.0, 128)
data_b = numpy.random.normal(numpy.pi*120.0/180.0, numpy.pi*15.0/180.0, 128)

mult_a = MeanShift()
mult_a.set_data(data_a, 'd', None, 'A')
mult_a.set_kernel('fisher(128.0)')

mult_b = MeanShift()
mult_b.set_data(data_b, 'd', None, 'A')
mult_b.set_kernel('fisher(512.0)')



# Do multiplication...
data_ab = numpy.empty((128,1), dtype=numpy.float32)
MeanShift.mult((mult_a, mult_b), data_ab)

mult_ab = MeanShift()
mult_ab.set_data(data_ab, 'df', None, 'A')
mult_ab.copy_all(mult_b)



# Visualise all angles...
img = numpy.zeros((64, 1024,3), dtype=numpy.float32)

for i in xrange(img.shape[1]):
  ang = 2.0 * numpy.pi * i / float(img.shape[1])
  img[:,i,0] = mult_a.prob(numpy.array([ang]))
  img[:,i,1] = mult_ab.prob(numpy.array([ang]))
  img[:,i,2] = mult_b.prob(numpy.array([ang]))



img *= 255.0 / img.max()
cv.SaveImage('angle_mult.png', array2cv(img))
