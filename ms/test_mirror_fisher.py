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
from utils.prog_bar import ProgBar

from ms import MeanShift



# Test the mirrored version of the von_mises Fisher distribution, this time in 5D...

# Create a dataset - just a bunch of points in one direction, so we can test the mirroring effect (Abuse MeanShift object to do this)...
print 'Mirrored draws:'

vec = numpy.array([1.0, 0.5, 0.0, -0.5, -1.0])
vec /= numpy.sqrt(numpy.square(vec).sum())

print 'Base dir =', vec

draw = MeanShift()
draw.set_data(vec, 'f')
draw.set_kernel('fisher(256.0)')

data = draw.draws(32)

#print 'Input:'
#print data



# Create a mean shift object from the draws, but this time with a mirror_fisher kernel...
mirror = MeanShift()
mirror.set_data(data, 'df')
mirror.set_kernel('mirror_fisher(64.0)')

resample = mirror.draws(16)

for row in resample:
  print '[%6.3f %6.3f %6.3f %6.3f %6.3f]' % tuple(row)
print



# Test probabilities by ploting them...
print 'Probability single 2D mirror Fisher:'
mirror = MeanShift()
mirror.set_data(numpy.array([1.0, 0.0]), 'f')
mirror.set_kernel('mirror_fisher(16.0)')

angles = numpy.linspace(0.0, numpy.pi*2.0, num=70, endpoint=False)
vecs = numpy.concatenate((numpy.cos(angles)[:,numpy.newaxis], numpy.sin(angles)[:,numpy.newaxis]),axis=1)
probs = mirror.probs(vecs)
probs /= probs.max()

steps = 12
for i in xrange(steps):
  threshold = 1.0 - (i+1)/float(steps)
  
  print ''.join(map(lambda p: '#' if p>threshold else ' ', probs))
print



# Test that mean shift still works; also test loo bandwidth estimation...
## Abuse the mean shift object to draw 8 uniform locations on a sphere...
usphere = MeanShift()
usphere.set_data(numpy.array([1.0, 0.0, 0.0]), 'f')
usphere.set_kernel('fisher(1e-6)') # Should be zero, but I don't support that.

centers = usphere.draws(8)
print('Distribution modes:')
print(centers)

## Use those modes to create a weighted mirror-fisher object, from which to draw lots of data...
weights = map(lambda x: 2.0 / (2.0+x), xrange(centers.shape[0]))
cext = numpy.concatenate((centers, numpy.array(weights)[:,numpy.newaxis]), axis=1)

wmf = MeanShift()
wmf.set_data(cext, 'df', 3)
wmf.set_kernel('mirror_fisher(24.0)')

## Create lots of data...
data = wmf.draws(1024)

swmf = MeanShift()
swmf.set_data(data, 'df')
swmf.set_kernel('mirror_fisher(128.0)')

## Get the indices of pixels into directions for a Mercator projection...
scale = 128
height = scale * 2
width = int(2.0 * numpy.pi * scale)

x_to_nx = numpy.cos(numpy.linspace(0.0, 2.0 * numpy.pi, width, False))
x_to_ny = numpy.sin(numpy.linspace(0.0, 2.0 * numpy.pi, width, False))
y_to_nz = numpy.linspace(-0.99, 0.99, height)

nx = x_to_nx.reshape((1,-1,1)).repeat(height, axis=0)
ny = x_to_ny.reshape((1,-1,1)).repeat(height, axis=0)
nz = y_to_nz.reshape((-1,1,1)).repeat(width, axis=1)

block = numpy.concatenate((nx, ny, nz), axis=2)
block[:,:,:2] *= numpy.sqrt(1.0 - numpy.square(y_to_nz)).reshape((-1,1,1))

## Visualise the probability of the data...
print 'Calculating Mercator probability:'
p = ProgBar()
locs = block.reshape((-1,3))
prob = numpy.empty(locs.shape[0], dtype=numpy.float32)
step = locs.shape[0] / scale

for i in xrange(scale):
  p.callback(i, scale)
  prob[i*step:(i+1)*step] = swmf.probs(locs[i*step:(i+1)*step,:])
del p

prob = prob.reshape((height, width))
image = array2cv(255.0 * prob / prob.max())
cv.SaveImage('mirror_fisher_mercator_kde.png', image)

## Apply mean shift and visualise the clustering...
swmf.merge_range = 0.1
modes, indices = swmf.cluster()

print 'Meanshift clustering:'
p = ProgBar()
clusters = numpy.empty(locs.shape[0], dtype=numpy.int32)

for i in xrange(scale):
  p.callback(i, scale)
  clusters[i*step:(i+1)*step] = swmf.assign_clusters(locs[i*step:(i+1)*step,:])
del p

clusters = clusters.reshape((height, width))
image = numpy.zeros((height, width, 3), dtype=numpy.float32)

for i in xrange(clusters.max()+1):
  colour = numpy.random.random(3)
  image[clusters==i,:] = colour.reshape((1,3))

image = array2cv(255.0 * image)
cv.SaveImage('mirror_fisher_mercator_ms.png', image)



# Try out multiplication, and throughroughly verify it works; for more fun include weighting...

