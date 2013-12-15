#! /usr/bin/env python

# Copyright 2013 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import random
import numpy
import numpy.random

import cv
from utils.cvarray import *

from ms import MeanShift



# Test using the von-Mises Fisher kernel, for directional entities - little different from other kernels as it requires that all input feature vectors be on the unit hyper-sphere...



# Generate some data - do the 3 great circles on the coordinate system, with noise either side...
data = []
samples = 128

for ex_dim in xrange(3):
  theta = 2.0 * numpy.pi * numpy.random.random(samples)
  deflection = 0.1 * numpy.random.random(samples) - 0.05
  
  cos_theta = numpy.cos(theta)
  sin_theta = numpy.sin(theta)
  
  which = [deflection, cos_theta, sin_theta]
  chunk = numpy.concatenate((which[ex_dim].reshape((-1,1)), which[(ex_dim+1)%3].reshape((-1,1)), which[(ex_dim+2)%3].reshape((-1,1))), axis=1)
  
  data.append(chunk)
  
data = numpy.concatenate(data, axis=0)
data /= numpy.sqrt(numpy.square(data).sum(axis=1)).reshape((-1,1))



# Setup mean shift...
ms = MeanShift()
ms.set_data(data, 'df')

ms.set_kernel('fisher(128.0)')
ms.set_spatial('kd_tree')



# Parameters for output images...
scale = 128
height = scale * 2
width = int(2.0 * numpy.pi * scale)



# Visualise the samples on a mercator projection...
print 'Samples...'

image = numpy.zeros((height, width, 3), dtype=numpy.float32)
for vec in data:
  x = numpy.arctan2(vec[1], vec[0])
  if x<0.0: x += 2.0*numpy.pi
  x = x * width / (2.0 * numpy.pi)
  if x>=width: x -= width
  
  y = 0.5*(1.0+vec[2]) * height

  image[y,x,:] = 255.0

image = array2cv(image)
cv.SaveImage('fisher_mercator_input.png', image)



# Make a mercator projection probability map, save it out...
print 'KDE...'
## Locations to sample...
x_to_nx = numpy.cos(numpy.linspace(0.0, 2.0 * numpy.pi, width, False))
x_to_ny = numpy.sin(numpy.linspace(0.0, 2.0 * numpy.pi, width, False))
y_to_nz = numpy.linspace(-0.99, 0.99, height)

nx = x_to_nx.reshape((1,-1,1)).repeat(height, axis=0)
ny = x_to_ny.reshape((1,-1,1)).repeat(height, axis=0)
nz = y_to_nz.reshape((-1,1,1)).repeat(width, axis=1)

block = numpy.concatenate((nx, ny, nz), axis=2)
block[:,:,:2] *= numpy.sqrt(1.0 - numpy.square(y_to_nz)).reshape((-1,1,1))

## Calculate the probability for each location...
prob = ms.probs(block.reshape((-1,3)))
prob = prob.reshape((height, width))

## Save it out...
image = array2cv(255.0 * prob / prob.max())
cv.SaveImage('fisher_mercator_kde.png', image)



# Draw a new set of samples; visualise them...
print 'Draw...'

draw = ms.draws(8*1024, 0)

image = numpy.zeros((height, width, 3), dtype=numpy.float32)
for vec in draw:
  x = numpy.arctan2(vec[1], vec[0])
  if x<0.0: x += 2.0*numpy.pi
  x = x * width / (2.0 * numpy.pi)
  if x>=width: x -= width
  
  y = 0.5*(1.0+vec[2]) * height

  image[y,x,:] = 255.0

image = array2cv(image)
cv.SaveImage('fisher_mercator_draw.png', image)



# Do mean shift on it, output a colour coded set of regions, same projection...
print 'MS...'
## Actual work...
modes, indices = ms.cluster()
clusters = ms.assign_clusters(block.reshape(-1,3))

## Create an image...
clusters = clusters.reshape((height, width))
image = numpy.zeros((height, width, 3), dtype=numpy.float32)

for i in xrange(clusters.max()+1):
  colour = numpy.random.random(3)
  image[clusters==i,:] = colour.reshape((1,3))

## Save it...
image = array2cv(255.0 * image)
cv.SaveImage('fisher_mercator_ms.png', image)
