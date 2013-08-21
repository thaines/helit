#! /usr/bin/env python

# Copyright 2013 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import random
import numpy
import numpy.random

from ms import MeanShift



# Create a simple data set...
a = numpy.random.normal(3.0, 1.0, 100)
b = numpy.random.normal(5.0, 0.5, 50)

data = numpy.concatenate((a,b))



# Setup the mean shift object...
ms = MeanShift()
ms.set_data(data, 'd')

ms.set_kernel(random.choice(filter(lambda s: s!='fisher', ms.kernels())))
ms.set_spatial(random.choice(ms.spatials()))



# Print out basic stats...
print 'kernel = %s; spatial = %s' % (ms.get_kernel(), ms.get_spatial())
print 'exemplars = %i; features = %i' % (ms.exemplars(), ms.features())
print 'quality = %.3f; epsilon = %.3f; iter_cap = %i' % (ms.quality, ms.epsilon, ms.iter_cap)
print

# Query the mode of various points...
for x in numpy.arange(0.0, 7.0, 0.4):
  mode = ms.mode(numpy.array([x]))
  print '%.3f: mode = %.3f' % (x, mode)
