#! /usr/bin/env python

# Copyright 2014 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import random
import numpy

from ms import MeanShift



# Create some data (!)...
data = ([-2] * 3) + ([0] * 8) + ([1] * 4) + ([2] * 5)
data = numpy.array(data, dtype=numpy.int32)



# Setup the mean shift object...
ms = MeanShift()
ms.set_data(data, 'd')
ms.set_kernel('discrete')
ms.set_spatial('kd_tree')



# Iterate and calculate the probability at a bunch of points, then plot...
sam = numpy.arange(-2.5, 2.5, 0.1)
prob = numpy.array(map(lambda v: ms.prob(numpy.array([v])), sam))

print 'Distribution:'
for threshold in numpy.arange(prob.max(), 0.0, -prob.max()/10.0):
  print ''.join(map(lambda p: '|' if p>threshold else ' ', prob))



# Draw a bunch of points, make a new ms and then plot, again...
ms2 = MeanShift()
ms2.set_data(ms.draws(50), 'df')
ms2.set_kernel('discrete')
ms2.set_spatial('kd_tree')

sam = numpy.arange(-2.5, 2.5, 0.1)
prob = numpy.array(map(lambda v: ms2.prob(numpy.array([v])), sam))

print 'Distribution of draw:'
for threshold in numpy.arange(prob.max(), 0.0, -prob.max()/10.0):
  print ''.join(map(lambda p: '|' if p>threshold else ' ', prob))



# Another distribution and its probability...
data = ([-2] * 3) + ([-1] * 2) + ([0] * 2) + ([1] * 3) + ([2] * 8)
data = numpy.array(data, dtype=numpy.int32)

ms3 = MeanShift()
ms3.set_data(data, 'd')
ms3.set_kernel('discrete')
ms3.set_spatial('kd_tree')

sam = numpy.arange(-2.5, 2.5, 0.1)
prob = numpy.array(map(lambda v: ms3.prob(numpy.array([v])), sam))

print 'Distribution to multiply first with:'
for threshold in numpy.arange(prob.max(), 0.0, -prob.max()/10.0):
  print ''.join(map(lambda p: '|' if p>threshold else ' ', prob))


 
# Multiply ms and ms3, plot probability...
mult = numpy.empty((64, 1), dtype=numpy.float32)
MeanShift.mult((ms, ms3), mult)

ms4 = MeanShift()
ms4.set_data(mult, 'df')
ms4.set_kernel('discrete')
ms4.set_spatial('kd_tree')

sam = numpy.arange(-2.5, 2.5, 0.1)
prob = numpy.array(map(lambda v: ms4.prob(numpy.array([v])), sam))

print 'Distribution of multiplication (expected to be wrong!):'
for threshold in numpy.arange(prob.max(), 0.0, -prob.max()/10.0):
  print ''.join(map(lambda p: '|' if p>threshold else ' ', prob))
