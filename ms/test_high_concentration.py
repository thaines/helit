#! /usr/bin/env python

# Copyright 2014 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import random
import numpy
import numpy.random

import cv
from utils.cvarray import *

from ms import MeanShift



# Repeat test fisher and test mirror fisher sequences, but with really high concentration parameters and lots of samples - a stress test basically.



# Parameters for output images...
scale = 256
height = scale * 2
width = int(2.0 * numpy.pi * scale)



# Bunch of arrays used to help with visualisation...
x_to_nx = numpy.cos(numpy.linspace(0.0, 2.0 * numpy.pi, width, False))
x_to_ny = numpy.sin(numpy.linspace(0.0, 2.0 * numpy.pi, width, False))
y_to_nz = numpy.linspace(-0.99, 0.99, height)

nx = x_to_nx.reshape((1,-1,1)).repeat(height, axis=0)
ny = x_to_ny.reshape((1,-1,1)).repeat(height, axis=0)
nz = y_to_nz.reshape((-1,1,1)).repeat(width, axis=1)

block = numpy.concatenate((nx, ny, nz), axis=2)
block[:,:,:2] *= numpy.sqrt(1.0 - numpy.square(y_to_nz)).reshape((-1,1,1))



# Helper function for below - visualises a distribution...
def visualise(fn, ms):
  prob = ms.probs(block.reshape((-1,3)))
  prob = prob.reshape((height, width))

  image = array2cv(255.0 * prob / prob.max())
  cv.SaveImage(fn, image)



# Generate some data - do the 3 great circles on the coordinate system, with noise either side...
data = []
samples = 1024*16

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
print 'Fisher:'
ms = MeanShift()
ms.set_data(data, 'df')

ms.set_kernel('fisher(4096.0)')
ms.set_spatial('kd_tree')



# Visualise the samples on a mercator projection...
print '  Samples...'

image = numpy.zeros((height, width, 3), dtype=numpy.float32)
for vec in data:
  x = numpy.arctan2(vec[1], vec[0])
  if x<0.0: x += 2.0*numpy.pi
  x = x * width / (2.0 * numpy.pi)
  if x>=width: x -= width
  
  y = 0.5*(1.0+vec[2]) * height

  image[y,x,:] = 255.0

image = array2cv(image)
cv.SaveImage('hc_fisher_mercator_input.png', image)



# Make a mercator projection probability map, save it out...
print '  KDE...'
visualise('hc_fisher_mercator_kde.png', ms)



# Draw a new set of samples; visualise them...
print '  Draw...'

draw = ms.draws(8*1024)

image = numpy.zeros((height, width, 3), dtype=numpy.float32)
for vec in draw:
  x = numpy.arctan2(vec[1], vec[0])
  if x<0.0: x += 2.0*numpy.pi
  x = x * width / (2.0 * numpy.pi)
  if x>=width: x -= width
  
  y = 0.5*(1.0+vec[2]) * height
  
  try:
    image[y,x,:] = 255.0
  except:
    print 'Bad draw:', vec

image = array2cv(image)
cv.SaveImage('hc_fisher_mercator_draw.png', image)



# Do mean shift on it, output a colour coded set of regions, same projection...
print '  MS...'
## Actual work...
ms.merge_range = 0.01
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
cv.SaveImage('hc_fisher_mercator_ms.png', image)



# Test multiplication...
ang = numpy.pi * (10.0 / 180.0)
mult = numpy.array([[numpy.cos(ang), -numpy.sin(ang), 0.0],
                    [numpy.sin(ang),  numpy.cos(ang), 0.0],
                    [           0.0,             0.0, 1.0]], dtype=numpy.float32)

print '  Mult...'
mult_data = data.copy()
for i in xrange(mult_data.shape[0]):
  mult_data[i,:] = numpy.dot(mult, mult_data[i,:])

ms_b = MeanShift()
ms_b.set_data(mult_data, 'df')
ms_b.set_kernel('fisher(4096.0)')
ms_b.set_spatial('kd_tree')

count = 1024*4
draws = numpy.empty((count,3), dtype=numpy.float32)
MeanShift.mult([ms, ms_b], draws)

ms_out = MeanShift()
ms_out.set_data(draws, 'df')
ms_out.set_kernel('fisher(4096.0)')
ms_out.set_spatial('kd_tree')

visualise('hc_fisher_mult_other.png', ms_b)
visualise('hc_fisher_mult.png', ms_out)



print '  Done.'
print



# Now do the mirrored Fisher distribution...
print 'Mirrored Fisher:'

# Generate some data - do a circle projected onto a sphere...
samples = 1024
data = numpy.empty((samples, 3), dtype=numpy.float32)

data[:,1] = 0.05 * numpy.random.random(samples) + 0.65

theta = 2.0 * numpy.pi * numpy.random.random(samples)
data[:,0] = numpy.cos(theta)
data[:,2] = numpy.sin(theta)

norm = numpy.sqrt(1.0 - data[:,1]*data[:,1])
data[:,0] *= norm
data[:,2] *= norm



# Setup mean shift...
ms = MeanShift()
ms.set_data(data, 'df')

ms.set_kernel('mirror_fisher(4096.0)')
ms.set_spatial('kd_tree')



# Visualise the samples on a mercator projection...
print '  Samples...'

image = numpy.zeros((height, width, 3), dtype=numpy.float32)
for vec in data:
  x = numpy.arctan2(vec[1], vec[0])
  if x<0.0: x += 2.0*numpy.pi
  x = x * width / (2.0 * numpy.pi)
  if x>=width: x -= width
  
  y = 0.5*(1.0+vec[2]) * height

  image[y,x,:] = 255.0

image = array2cv(image)
cv.SaveImage('hc_mirror_fisher_mercator_input.png', image)



# Make a mercator projection probability map, save it out...
print '  KDE...'
visualise('hc_mirror_fisher_mercator_kde.png', ms)



# Draw a new set of samples; visualise them...
print '  Draw...'

draw = ms.draws(8*1024)

image = numpy.zeros((height, width, 3), dtype=numpy.float32)
for vec in draw:
  x = numpy.arctan2(vec[1], vec[0])
  if x<0.0: x += 2.0*numpy.pi
  x = x * width / (2.0 * numpy.pi)
  if x>=width: x -= width
  
  y = 0.5*(1.0+vec[2]) * height
  
  try:
    image[y,x,:] = 255.0
  except:
    print 'Bad draw:', vec

image = array2cv(image)
cv.SaveImage('hc_mirror_fisher_mercator_draw.png', image)



# Do mean shift on it, output a colour coded set of regions, same projection...
print '  MS...'
## Actual work...
ms.merge_range = 0.01
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
cv.SaveImage('hc_mirror_fisher_mercator_ms.png', image)



# Test multiplication...
ang = numpy.pi * (10.0 / 180.0)
mult = numpy.array([[numpy.cos(ang), -numpy.sin(ang), 0.0],
                    [numpy.sin(ang),  numpy.cos(ang), 0.0],
                    [           0.0,             0.0, 1.0]], dtype=numpy.float32)

print '  Mult...'
mult_data = data.copy()
for i in xrange(mult_data.shape[0]):
  mult_data[i,:] = numpy.dot(mult, mult_data[i,:])

ms_b = MeanShift()
ms_b.set_data(mult_data, 'df')
ms_b.set_kernel('mirror_fisher(4096.0)')
ms_b.set_spatial('kd_tree')

count = 1024
draws = numpy.empty((count,3), dtype=numpy.float32)
MeanShift.mult([ms, ms_b], draws)

ms_out = MeanShift()
ms_out.set_data(draws, 'df')
ms_out.set_kernel('mirror_fisher(4096.0)')
ms_out.set_spatial('kd_tree')

visualise('hc_mirror_fisher_mult_other.png', ms_b)
visualise('hc_mirror_fisher_mult.png', ms_out)



print '  Done.'
print
