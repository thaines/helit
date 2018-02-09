#! /usr/bin/env python

# Copyright 2018 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import numpy
import numpy.random
from scipy.misc import imsave

from utils.prog_bar import ProgBar

from ms import MeanShift



# Visualises each kernel...
for kernel in MeanShift.kernels():
  if MeanShift.info_config(kernel)!=None:
    continue
  
  a = MeanShift()
  a.set_data(numpy.array([0.0], dtype=numpy.float32), 'f')
  a.set_kernel(kernel)
  a.quality = 1.0
  
  image = numpy.ones((384, 1024), dtype=numpy.float32)
  x = numpy.linspace(-4, 4, image.shape[1])
  iy, ix = numpy.meshgrid(numpy.arange(image.shape[0]), numpy.arange(image.shape[1]), indexing='ij')
  
  ya = a.probs(x[:,None])
  ya = (image.shape[0]-1 - (image.shape[0]-1) * (ya / ya.max())).astype(numpy.int32)
  
  image[iy >= ya[None,:]] = 0.0
  
  imsave('kernel_%s.png' % kernel, image)



# Generate some data...
a = numpy.random.normal(-2.0, 1.5, 256)
b = numpy.random.normal(2.0, 0.2, a.shape[0])
t = numpy.random.beta(0.2, 0.2, a.shape[0])
samples = ((1.0 - t) * a + t * b).astype(numpy.float32)



# Fit each kernel in turn...
for kernel in MeanShift.kernels():
  if MeanShift.info_config(kernel)!=None:
    continue
  print(kernel)
  
  m = MeanShift()
  m.set_data(samples[:,None], 'df')
  m.set_kernel(kernel)
  m.quality = 1.0
  
  p = ProgBar()
  m.scale_loo_nll(callback = p.callback)
  del p
  
  image = numpy.ones((512, 1024), dtype=numpy.float32)
  x = numpy.linspace(-5, 4, image.shape[1])
  iy, ix = numpy.meshgrid(numpy.arange(image.shape[0]), numpy.arange(image.shape[1]), indexing='ij')

  ya = m.probs(x[:,None])
  ya = (image.shape[0]-1 - (image.shape[0]-1) * (ya / (ya.max()*1.05))).astype(numpy.int32)
  
  image[iy >= ya[None,:]] = 0.0
  
  imsave('kernel_demo_%s.png' % kernel, image)
