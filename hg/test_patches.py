#! /usr/bin/env python

# Copyright 2016 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import sys

import numpy
from scipy.misc import imread, imsave

from transform import *



# Load a file, make into a nice floaty image...
if len(sys.argv)<2:
  print('Usage:')
  print('  ./test_patches.py <image fn>')
  sys.exit(1)

fn = sys.argv[1]

image = imread(fn).astype(numpy.float32)
shape = image.shape
image = {'r' : image[:,:,0], 'g' : image[:,:,1], 'b' : image[:,:,2]}



# Offsets for each patch...
psize = 8

indices = numpy.linspace(-psize*0.5, psize*0.5, num=psize).astype(numpy.float32)
offsets = numpy.transpose(numpy.meshgrid(indices, indices, indexing='ij'))
offsets = offsets.reshape((-1,2))[:,::-1].astype(numpy.float32)



# Generate a bunch of patch locations from a jittered grid, with random orientations...
sy = numpy.linspace(psize*0.5, shape[0] - psize*0.5, num=shape[0]/psize).astype(numpy.float32)
sx = numpy.linspace(psize*0.5, shape[1] - psize*0.5, num=shape[1]/psize).astype(numpy.float32)

sy += psize * (numpy.random.rand(sy.shape[0]) - 0.5)
sx += psize * (numpy.random.rand(sx.shape[0]) - 0.5)

points = numpy.transpose(numpy.meshgrid(sy, sx)).reshape((-1, 2)).astype(numpy.float32)[:,::-1]

angles = numpy.pi * 2.0 * numpy.random.rand(points.shape[0])
rotations = numpy.transpose((numpy.cos(angles), numpy.sin(angles))).astype(numpy.float32)



# Sample patches...
patches = rotsets(image, points, rotations, offsets)



# Put them all back together into a weird image...
output = numpy.zeros((psize * sy.shape[0], psize * sx.shape[0], 3), dtype=numpy.float32)
patch = numpy.empty((psize, psize, 3))

for py in xrange(sy.shape[0]):
  for px in xrange(sx.shape[0]):
    ind = py * sx.shape[0] + px
    
    patch[:,:,0] = patches['r'][ind,:].reshape((psize, psize))
    patch[:,:,1] = patches['g'][ind,:].reshape((psize, psize))
    patch[:,:,2] = patches['b'][ind,:].reshape((psize, psize))
    
    output[py*psize:(py+1)*psize,px*psize:(px+1)*psize,:] = patch



# Save image out...
output_fn = os.path.splitext(fn)[0] + '_patches.png'
imsave(output_fn, output)
