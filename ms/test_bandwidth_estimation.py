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



base_samples = 64
scale = 1.0
size = 5.0
circle = True
line = True



# Sample from a circle + noise model to create some data...
if circle:
  samples = 8 * base_samples
  theta = 2.0 * numpy.pi * numpy.random.random(samples)
  radius = scale*3.0 + (numpy.random.beta(5.0, 5.0, samples)-0.5)

  x = radius * numpy.cos(theta)
  y = radius * numpy.sin(theta)

  data1 = numpy.concatenate((x.reshape((-1,1)), y.reshape((-1,1))), axis=1)


# More data - from a line...
if line:
  samples = 4 * base_samples
  x = scale * (numpy.random.beta(3.0, 3.0, samples)*9.0 - 4.5)
  y = numpy.random.normal(scale=0.2, size=samples)

  data2 = numpy.concatenate((x.reshape((-1,1)), y.reshape((-1,1))), axis=1)


# Munge it all together...
if circle and line: data = numpy.concatenate((data1, data2), axis=0)
elif circle: data = data1
else: data = data2
numpy.random.shuffle(data)



# Visualise the samples...
dim = 512
image = numpy.zeros((dim, dim, 3), dtype=numpy.float32)

for r in xrange(data.shape[0]):
  loc = data[r,:]
  loc = (loc + size) / (2.0*size)
  loc *= dim
  try:
    image[int(loc[1]+0.5), int(loc[0]+0.5), :] = 255.0
  except: pass # Deals with out of range values.

image = array2cv(image)
cv.SaveImage('bandwidth_samples.png', image)



# Setup the mean shift object...
ms = MeanShift()
ms.set_data(data, 'df')
ms.set_kernel('gaussian')
ms.set_spatial('kd_tree')



# Progress bar version of scale_loo_nll...
def scale_loo_nll():
  p = ProgBar()
  ms.scale_loo_nll(callback = p.callback)
  del p

  

# Iterate and try out a bunch of different algorithms...
for name, alg in [('human_picked', lambda: ms.set_scale(numpy.array([5.0, 5.0]))), ('Silverman',ms.scale_silverman), ('Scott', ms.scale_scott), ('loo_nll', scale_loo_nll)]:
  # Calculate and print out the scales...
  print '<', name, '>'
  alg()
  print 'Scale:', ms.get_scale()
  print 'loo nll for this scale =', ms.loo_nll()
  mean, sd = ms.stats()
  print 'mean = (%f, %f); sd = (%f, %f)'%(mean[0], mean[1], sd[0], sd[1])
  
  # Render out a normalised probability map...
  image = numpy.zeros((dim, dim, 3), dtype=numpy.float32)
  
  p = ProgBar()
  for row in xrange(dim):
    p.callback(row, dim)
    sam = numpy.append(numpy.linspace(-size, size, dim).reshape((-1,1)), ((row / (dim-1.0) - 0.5) * 2.0 * size) * numpy.ones(dim).reshape((-1,1)), axis=1)
    image[row, :, :] = ms.probs(sam).reshape((-1,1))
  del p
  
  print 'Largest sampled probability =', image.max()
  image *= 255.0 / image.max()
  
  image = array2cv(image)
  cv.SaveImage('bandwidth_%s.png'%name, image)
  print
