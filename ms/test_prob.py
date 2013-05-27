#! /usr/bin/env python

# Copyright 2013 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import random
import numpy

from ms import MeanShift



# Create a dataset - equally spaced samples weighted by a Gaussian, such that it should estimate a Gaussian...
x = numpy.arange(-5.0, 5.0, 0.02)
y = numpy.exp(-0.5 * x**2.0 / 2.0)

data = numpy.concatenate((x.reshape((-1,1)), y.reshape((-1,1))), axis=1)



# Setup the mean shift object...
ms = MeanShift()
ms.set_data(data, 'df')

ms.set_kernel(random.choice(filter(lambda s: s!='fisher', ms.kernels())))
ms.set_spatial(random.choice(ms.spatials()))
ms.set_scale(numpy.ones(2), 1)



# Iterate and calculate the probability at every point...
sam = numpy.arange(-5.0, 5.0, 0.15)
prob = numpy.array(map(lambda v: ms.prob(numpy.array([v,1.0])), sam))



# Print out basic stats...
print 'kernel = %s; spatial = %s' % (ms.get_kernel(), ms.get_spatial())
print 'exemplars = %i; features = %i' % (ms.exemplars(), ms.features())
print



# Visualise the output...
for threshold in numpy.arange(prob.max(), 0.0, -prob.max()/15.0):
  print ''.join(map(lambda p: '|' if p>threshold else ' ', prob))
