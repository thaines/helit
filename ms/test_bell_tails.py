#! /usr/bin/env python

# Copyright 2016 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import numpy
import numpy.random
from scipy.misc import imsave

from utils.prog_bar import ProgBar

from ms import MeanShift



# Multiplies the three great bells (Gaussian, Cauchy and Logistic - not their actual name, but it should be:-P) such that only their tails overlap, drawing a lot of samples then visualising a density estimate of the resulting shape - a rather roundabout demonstration of how they are different...



great_bells = [('gaussian', 4.0), ('cauchy', 6.0), ('logistic', 8.0)]

for bell, gap in great_bells:
  print '%s:' % bell
  
  # Setup two single sample models...
  a = MeanShift()
  a.set_data(numpy.array([-0.5*gap], dtype=numpy.float32), 'f')
  a.set_kernel(bell)
  a.quality = 1.0
  
  b = MeanShift()
  b.set_data(numpy.array([0.5*gap], dtype=numpy.float32), 'f')
  b.set_kernel(bell)
  b.quality = 1.0
  
  # Multiply them and generate new distribution...
  draw = numpy.empty((1024, 1), dtype=numpy.float32)
  MeanShift.mult([a,b], draw)
  
  ab = MeanShift()
  ab.set_data(draw, 'df')
  ab.set_kernel('triangular')
  ab.quality = 1.0
  
  p = ProgBar()
  ab.scale_loo_nll(callback = p.callback)
  del p
  
  # Visualise...
  image = numpy.zeros((512, 1024, 3), dtype=numpy.float32)
  x = numpy.linspace(-gap, gap, image.shape[1])
  iy, ix = numpy.meshgrid(numpy.arange(image.shape[0]), numpy.arange(image.shape[1]), indexing='ij')
  
  ya = a.probs(x[:,None])
  ya = (image.shape[0]-1 - (image.shape[0]-1) * (ya / ya.max())).astype(numpy.int32)
  
  yab = ab.probs(x[:,None])
  yab = (image.shape[0]-1 - (image.shape[0]-1) * (yab / yab.max())).astype(numpy.int32)
  
  yb = b.probs(x[:,None])
  yb = (image.shape[0]-1 - (image.shape[0]-1) * (yb / yb.max())).astype(numpy.int32)
  
  image[iy >= ya[None,:],0] = 0.5
  image[iy >= yab[None,:],1] = 1.0
  image[iy >= yb[None,:],2] = 0.5
  
  imsave('bell_%s.png' % bell, image)
  