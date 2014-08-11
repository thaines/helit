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



# Setup a basic weighted draw - even distribution of exemplars, draw represented using weights...
pixels = 512
samples = 64

data = numpy.empty((samples*samples, 3))

i = 0
for y in xrange(samples):
  for x in xrange(samples):
    data[i,0] = y / float(samples-1)
    data[i,1] = x / float(samples-1)
    
    dist = numpy.sqrt((data[i,0]-0.5)**2 + (data[i,1]-0.5)**2) * numpy.pi * 7.0
    data[i,2] = (1.0+numpy.sin(dist)) / (6.0 + numpy.abs(numpy.sqrt(dist)-3.0))
    
    i += 1

ms = MeanShift()
ms.set_data(data, 'df', 2)
ms.set_kernel('triangular')
ms.set_spatial('kd_tree')



# Choose a reasonable size...
print 'Selecting size using loo:'
p = ProgBar()
ms.scale_loo_nll(callback = p.callback)
del p



# Plot the pdf, for reference...
image = numpy.zeros((pixels, pixels, 3), dtype=numpy.float32)

print 'Rendering probability map:'
p = ProgBar()
for row in xrange(pixels):
  p.callback(row, pixels)
  sam = numpy.append(numpy.linspace(0.0, 1.0, pixels).reshape((-1,1)), (row / float(pixels-1)) * numpy.ones(pixels).reshape((-1,1)), axis=1)
  image[row, :, :] = ms.probs(sam).reshape((-1,1))
del p

image *= 255.0 / image.max()
image = array2cv(image)
cv.SaveImage('draw_weighted_density.png', image)



# Draw a bunch of points from it and plot them...
samples = numpy.random.randint(16, 512)
draw = ms.draws(1024)

image = numpy.zeros((pixels, pixels, 3), dtype=numpy.float32)

for r in xrange(draw.shape[0]):
  loc = draw[r,:]
  loc *= pixels
  try:
    image[int(loc[1]+0.5), int(loc[0]+0.5), :] = 255.0
  except: pass # Deals with out of range values.

image = array2cv(image)
cv.SaveImage('draw_weighted_samples.png', image)
