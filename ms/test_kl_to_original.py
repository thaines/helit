#! /usr/bin/env python

# Copyright 2014 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import numpy
from ms import MeanShift



# This draws lots of samples from each kernel and then does a kernel density estimate on them, and approximates the KL-divergance between them. Note that because its stochastic it can get negative results!



# Parameters...
samples = 1024 * 16
samples_dir = 1024 * 2
scale = 16.0

dimensions = [1, 2, 3]
dir_dimensions = [2, 3, 4]
dir_conc = [2.0, 16.0, 128.0, 1024.0]



# Do the 'simple' kernels...
for kernel in ['uniform', 'triangular', 'epanechnikov', 'cosine', 'gaussian', 'cauchy']:
  for dim in dimensions:
    # Create a mean shift object with a single sample of the provided kernel type...
    ms = MeanShift()
    ms.set_data(numpy.array([0.0]*dim, dtype=numpy.float32), 'f')
    ms.set_kernel(kernel)
    ms.quality = 1.0
    
    # Draw lots of samples from it...
    sample = ms.draws(samples)
    
    # Get the probability of each...
    p1 = ms.probs(sample)
    
    # Throw away samples where p1 is 0 - they are a result of the range optimisation, and break the below...
    keep = p1>1e-6
    sample = sample[keep,:]
    p1 = p1[keep]
    
    # Do a KDE of the samples, including bandwidth estimation...
    kde = MeanShift()
    kde.set_data(sample, 'df')
    kde.set_kernel('uniform') #  Keep is simple!
    kde.set_spatial('kd_tree')
    kde.set_scale(numpy.array([scale]*dim, dtype=numpy.float32))
    
    # Calculate a stochastic KL-divergance between the kde and the actual distribution...
    p2 = kde.probs(sample)
    kld = numpy.sum(numpy.log(p1/p2)) / samples
    
    # Print output to screen...
    print 'Kernel = %s; Dims = %i | KL-divergance = %.6f' % (kernel, dim, kld)
  
  print



# Now for the directional kernels...
for kernel in ['fisher', 'mirror_fisher']:
  for dim in dir_dimensions:
    for conc in dir_conc:
      # Create a mean shift object pointing in the [1, 0, ...] direction with the given concentration...
      ms = MeanShift()
      ms.set_data(numpy.array([1.0] + [0.0]*(dim-1), dtype=numpy.float32), 'f')
      ms.set_kernel('%s(%.1f)' % (kernel, conc))
      ms.quality = 1.0
      
      # Draw lots of samples from it...
      sample = ms.draws(samples_dir)
    
      # Get the probability of each...
      p1 = ms.probs(sample)
      
      # Throw away samples where p1 is 0 - they are a result of the range optimisation, and break the below...
      keep = p1>1e-6
      sample = sample[keep,:]
      p1 = p1[keep]
      
      # Do a KDE of the samples, including bandwidth estimation...
      kde = MeanShift()
      kde.set_data(sample, 'df')
      kde.set_kernel('fisher(%f)' % (conc*32)) #  Hardly ideal - need something more independent/safer!
      kde.set_spatial('kd_tree')
      
      # Calculate a stochastic KL-divergance between the kde and the actual distribution...
      p2 = kde.probs(sample)
      kld = numpy.sum(numpy.log(p1/p2)) / samples_dir
    
      # Print output to screen...
      print 'Kernel = %s; Dims = %i | KL-divergance = %.6f' % (ms.get_kernel(), dim, kld)
  
  print
