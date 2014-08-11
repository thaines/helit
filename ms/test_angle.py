#! /usr/bin/env python

# Copyright 2013 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import numpy
import numpy.random

import cv
from utils.cvarray import *
from utils.prog_bar import ProgBar

from ms import MeanShift



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
ms.set_data(data, 'df')
ms.set_kernel('angle(16.0)')

modes, indices = ms.cluster()
print 'Found %i modes' % modes.shape[0]



# Another crazy visualisation, this time the modes...
img = numpy.zeros((256, 1024), dtype=numpy.float32)
for row in modes:
  for y in xrange(img.shape[0]):
    t = float(y) / float(img.shape[0]-1)
    x = row[0] * (1.0-t) + row[1] * t
    if x<-numpy.pi: x += numpy.pi
    if x>numpy.pi: x -= numpy.pi
    x = int(img.shape[1] * (x + numpy.pi) / (numpy.pi * 2.0))
    
    img[y, x] += 1.0

img *= 255.0 / img.max()
cv.SaveImage('angle_modes.png', array2cv(img))



# Plot the pdf, different visualisation style...
img = numpy.zeros((8, 8), dtype=numpy.float32)

for y in xrange(img.shape[0]):
  for x in xrange(img.shape[1]):
    ang_x = 2.0 * numpy.pi * (x / float(img.shape[1]-1)) - numpy.pi
    ang_y = 2.0 * numpy.pi * (y / float(img.shape[0]-1)) - numpy.pi
    img[y,x] = ms.prob(numpy.array([ang_x, ang_y]))
    
    print ang_x, ang_y, img[y,x]

img *= 255.0 / img.max()
cv.SaveImage('angle_pdf.png', array2cv(img))



# New data sets to multiply together, single angles this time...


# Do multiplication and visualise the output angle...
