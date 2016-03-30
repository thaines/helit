#! /usr/bin/env python

# Copyright 2016 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import sys

import numpy
from scipy.misc import imread, imsave

from transform import *



# Weird - takes an image, splits it into a grid and plots the values in a circle around each point, then plots them back again to form an image of rings of values - just a way to test the Offset method...
grid_size = 8
samples = 64
sample_grid = 8

assert(sample_grid * sample_grid==samples)



# Load a file, make into a nice floaty image...
if len(sys.argv)<2:
  print('Usage:')
  print('  ./test_rings.py <image fn>')
  sys.exit(1)

fn = sys.argv[1]

image = imread(fn).astype(numpy.float32)
shape = image.shape
image = {'r' : image[:,:,0], 'g' : image[:,:,1], 'b' : image[:,:,2]}



# Calculate ring, choose grid...
radius = 0.25 * 32
ang = numpy.linspace(0.0, numpy.pi*2.0, samples, endpoint=False)

ring = numpy.concatenate((radius * numpy.sin(ang)[:,None], radius * numpy.cos(ang)[:,None]), axis=1)
ring = ring.astype(numpy.float32)

cy = numpy.arange(grid_size/2, image['r'].shape[0], grid_size)
cx = numpy.arange(grid_size/2, image['r'].shape[1], grid_size)

gy, gx = numpy.meshgrid(cy, cx, indexing='ij')

grid = numpy.concatenate((gy.reshape((-1,1)), gx.reshape((-1,1))), axis=1)
grid = grid.astype(numpy.float32)



# Evaluate...
feats = offsets(image, grid, ring)



# Build a weird image out of them...
out = numpy.zeros((cy.shape[0] * sample_grid, cx.shape[0] * sample_grid, 3), dtype=numpy.uint8)

for i, name in enumerate(['r', 'g', 'b']):
  data = feats[name]
  
  for y in xrange(gy.shape[0]):
    for x in xrange(gx.shape[0]):
      block = data[y * gx.shape[0] + x,:].reshape((sample_grid, sample_grid))
      out[y*sample_grid:(y+1)*sample_grid, x*sample_grid:(x+1)*sample_grid, i] = block.astype(numpy.uint8)



# Save...
out_fn = os.path.splitext(fn)[0] + '_rings.png'
imsave(out_fn, out)
