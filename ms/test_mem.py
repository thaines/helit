#! /usr/bin/env python

# Copyright 2014 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import sys

import numpy
import numpy.random

from ms import MeanShift



# Create a mean shift object and do various options, printing out its memeory usage after each...
ms = MeanShift()
print 'Size after creation: %i' % sys.getsizeof(ms)

data = numpy.random.random((2048,6))
data = numpy.array(data, dtype=numpy.float32)
point = numpy.array([0.5] * data.shape[1])

ms.set_data(data, 'df')
print 'Size after adding some data: %i' % sys.getsizeof(ms)

ms.set_spatial('brute_force')
ms.prob(point)
print 'Size after calling prob with brute force spatial: %i' % sys.getsizeof(ms)

ms.set_spatial('kd_tree')
ms.prob(point)
print 'Size after calling prob with kd-tree spatial: %i' % sys.getsizeof(ms)



# Test out a Composite Fisher/Gaussian kernel, because that runs through a bunch of other code paths...
ms.set_kernel('composite(3:gaussian,3:fisher(32.0))')
ms.prob(point) # Garbage, as does not satisfy 'direction is nomalised' constraint.
print 'Size with crazy composite kernel: %i' % sys.getsizeof(ms)



# Test out the memory breakdown method...
print
print 'Breakdown in final state:'
mem = ms.memory()

for key, value in mem.iteritems():
  if key=='kernel_ref_count' or key=='total':
    continue
  
  if key=='kernel':
    print '  %s: %i bytes (ref count = %i)' % (key, value, mem['kernel_ref_count'])
  else:
    print '  %s: %i bytes' % (key, value)

print 'total = %i bytes' % mem['total']
print
