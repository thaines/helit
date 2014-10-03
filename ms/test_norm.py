#! /usr/bin/env python

# Copyright 2014 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import numpy
from ms import MeanShift



# Verifies I didn't screw up the distributions - does monte-carlo integration to verify that they have volume 1. Obviously its only doing a finite number of samples, so the numbers output won't be exactly 1, but at least verifies things aren't too nuts.



# Parameters...
samples = 1024 * 1024
dimensions = [1, 2, 3]
dir_dimensions = [2, 3, 4]
dir_area = [numpy.pi * 2.0, numpy.pi * 4.0, 2.0 * numpy.pi**2]
dir_conc = [2.0, 16.0, 128.0, 1024.0]



# Do the 'simple' kernels...
for kernel in ['uniform', 'triangular', 'epanechnikov', 'cosine', 'gaussian', 'cauchy']:
  for dim in dimensions:
    # Create a mean shift object with a single sample of the provided kernel type...
    ms = MeanShift()
    ms.set_data(numpy.array([0.0]*dim, dtype=numpy.float32), 'f')
    ms.set_kernel(kernel)
    ms.quality = 1.0
    
    # Create a uniform sample over a suitably large region (Yes I am assuming I got the uniform kernel right!)...
    uniform = MeanShift()
    uniform.set_data(numpy.array([0.0]*dim, dtype=numpy.float32), 'f')
    uniform.set_kernel('uniform')
    uniform.set_scale(numpy.ones(dim) / ms.get_range())
    sample = uniform.draws(samples)
    sp = uniform.prob(sample[0,:])
    
    # Evaluate the probabilities of the uniform set...
    p = ms.probs(sample)
    
    # Print their average, which we are hoping is one...
    volume =  p.mean() / sp
    print 'Kernel = %s; Dims = %i | Monte-Carlo volume = %.3f' % (kernel, dim, volume)
  print


# Now for the directional kernels...
for kernel in ['fisher', 'mirror_fisher']:
  for dim, area in zip(dir_dimensions, dir_area):
    for conc in dir_conc:
      # Create a mean shift object pointing in the [1, 0, ...] direction with the given concentration...
      ms = MeanShift()
      ms.set_data(numpy.array([1.0] + [0.0]*(dim-1), dtype=numpy.float32), 'f')
      ms.set_kernel('%s(%.1f)' % (kernel, conc))
      ms.quality = 1.0
      
      # Create uniform samples on the hyper-sphere with which we are dealing - abuse the MeanShift object by drawing with a Gaussian kernel and normalising...
      uniform = MeanShift()
      uniform.set_data(numpy.array([0.0]*dim, dtype=numpy.float32), 'f')
      uniform.set_kernel('gaussian')
      sample = uniform.draws(samples)
      
      div = numpy.sqrt(numpy.square(sample).sum(axis=1))
      sample /= div[:, numpy.newaxis]
      
      # Evaluate the probabilities of the uniform directions...
      p = ms.probs(sample)
      
      # Print their average - should again be one...
      volume = p.mean() * area
      print 'Kernel = %s; Dims = %i | Monte-Carlo volume = %.3f' % (ms.get_kernel(), dim, volume)
  print
