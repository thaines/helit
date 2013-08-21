#! /usr/bin/env python

# Copyright 2013 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import numpy
import numpy.random

import cv
from utils.cvarray import *

from ms import MeanShift



# Sample from a circle + noise model to create some data...
samples = 8192
theta = 2.0 * numpy.pi * numpy.random.random(samples)
radius = 3.0 + (numpy.random.beta(2.0, 2.0, samples)-0.5)

x = radius * numpy.cos(theta)
y = radius * numpy.sin(theta)

data = numpy.concatenate((x.reshape((-1,1)), y.reshape((-1,1))), axis=1)



# Setup the mean shift object...
ms = MeanShift()
ms.set_data(data, 'df')
ms.set_spatial('kd_tree')



# Loop the circle; project points to it...
steps = 16
for i in xrange(steps):
  angle = 2.0 * numpy.pi * float(i) / float(steps)
  rad = 3.0 + (numpy.random.beta(2.0, 2.0)-0.5)
  
  x = rad * numpy.cos(angle)
  y = rad * numpy.sin(angle)
  
  proj = ms.manifold(numpy.array([x,y]),1)
  
  print '(%.3f,%.3f) -> (%.3f,%.3f) | rad: %.3f -> %.3f' % (x, y, proj[0], proj[1], numpy.sqrt(x*x+y*y), numpy.sqrt(numpy.square(proj).sum()))
print


# Do some visualisation...
dim = 512
image = numpy.zeros((dim, dim, 3), dtype=numpy.float32)

for r in xrange(data.shape[0]):
  loc = data[r,:]
  loc = (loc + 4.0) / 8.0
  loc *= dim
  image[int(loc[1]+0.5), int(loc[0]+0.5), :] = 64.0

print 'Projecting samples to line...'
to_render = 1024
line = ms.manifolds(data[:to_render], 1)
print 'Done'

for r in xrange(line.shape[0]):
  loc = line[r,:]
  loc = (loc + 4.0) / 8.0
  loc *= dim
  image[int(loc[1]+0.5), int(loc[0]+0.5), :] = 255.0

image = array2cv(image)
cv.SaveImage('manifold.png', image)
