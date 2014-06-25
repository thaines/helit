#! /usr/bin/env python

# Copyright 2014 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import random
import numpy

from ms import MeanShift



# Create a dataset - draws from a Gaussian...
data = numpy.array(map(lambda _: random.normalvariate(0.0, 2.0), xrange(1000)))



# Setup three mean shift objects with the same data set and draw from them to demonstrate that you get the exact same output...
print 'Should all be the same:'
ms = map(lambda _: MeanShift(), xrange(3))
for i in xrange(len(ms)):
  ms[i].set_data(data, 'd')
  ms[i].set_kernel('gaussian')
  ms[i].set_spatial('kd_tree')
  
  print 'From', i, '|', ms[i].draw()
print



# Link the second to the first and draw again - first two should be different, third the same as the first...
ms[1].link_rng(ms[0])

print '#2 different:'
for i in xrange(len(ms)):
  print 'From', i, '|', ms[i].draw()
print



# Skip one for the third, so the same pattern should appear again...
ms[2].draw()

print '#2 different (again):'
for i in xrange(len(ms)):
  print 'From', i, '|', ms[i].draw()
print



# Link them all, so they are all different...
ms[2].link_rng(ms[1])

print 'All different:'
for i in xrange(len(ms)):
  print 'From', i, '|', ms[i].draw()
print
