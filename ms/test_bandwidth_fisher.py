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

# Does bandwidth estimation with a Fisher distribution, using points on a sphere.



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



# Generate a data set...
samples = 128
data = numpy.empty((samples, 3), dtype=numpy.float32)

data[:,1] = 0.1 * numpy.random.random(samples) + 0.6

theta = 2.0 * numpy.pi * numpy.random.random(samples)
data[:,0] = numpy.cos(theta)
data[:,2] = numpy.sin(theta)

norm = numpy.sqrt(1.0 - data[:,1]*data[:,1])
data[:,0] *= norm
data[:,2] *= norm



# Create a panel of possible distributions...
def ms_by_conc(power, code=''):
  ms = MeanShift()
  ms.quality = 0.5
  ms.set_kernel('fisher(%.1f%s)' % (2**power, code))
  ms.set_spatial('kd_tree')
  
  return ms

options = map(ms_by_conc, xrange(8)) + [ms_by_conc(8,'c'), ms_by_conc(8,'a')] + map(ms_by_conc, xrange(9,16))



# Create it and do the bandwidth estimation...
ms = MeanShift()
ms.set_data(data, 'df')

p = ProgBar()
best = ms.scale_loo_nll_array(options, p.callback)
del p

print 'Selected kernel =', ms.get_kernel()
print 'LOO score =', best



# Visualise the best option...
visualise('bandwidth_fisher.png', ms)



# Also visualise all the rest, for sanity checking...
for option in [ms_by_conc(8,'c'), ms_by_conc(8,'a')]: #options:
  ms.copy_all(option)
  visualise('bandwidth_fisher_%s.png' % option.get_kernel(), ms)
