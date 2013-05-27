#! /usr/bin/env python

# Copyright 2013 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import random
import numpy
import numpy.random

from ms import MeanShift



# Create a data set...
alpha_a1 = 5.0
beta_a1 = 10.0
alpha_a2 = 3.0
beta_a2 = 1.0
count_a = 100

a = numpy.concatenate((numpy.random.beta(alpha_a1, beta_a1, count_a).reshape((-1,1)), numpy.random.beta(alpha_a2, beta_a2, count_a).reshape((-1,1))), axis=1)

alpha_b1 = 16.0
beta_b1 = 2.0
alpha_b2 = 20.0
beta_b2 = 6.0
count_b = 75

b = numpy.concatenate((numpy.random.beta(alpha_b1, beta_b1, count_b).reshape((-1,1)), numpy.random.beta(alpha_b2, beta_b2, count_b).reshape((-1,1))), axis=1)

alpha_c1 = 4.0
beta_c1 = 1.0
alpha_c2 = 5.0
beta_c2 = 30.0
count_c = 50

c = numpy.concatenate((numpy.random.beta(alpha_c1, beta_c1, count_c).reshape((-1,1)), numpy.random.beta(alpha_c2, beta_c2, count_c).reshape((-1,1))), axis=1)

data = numpy.concatenate((a,b,c), axis=0)
scale = 6.0
data *= scale



# Setup the mean shift object...
ms = MeanShift()
ms.set_data(data, 'df')

ms.set_kernel(random.choice(filter(lambda s: s!='fisher', ms.kernels())))
ms.set_spatial(random.choice(ms.spatials()))



# Print out basic stats...
print 'kernel = %s; spatial = %s' % (ms.get_kernel(), ms.get_spatial())
print 'exemplars = %i; features = %i' % (ms.exemplars(), ms.features())
print 'quality = %.3f; epsilon = %.3f; iter_cap = %i' % (ms.quality, ms.epsilon, ms.iter_cap)
print



# Create a grid of samples...
axis = numpy.arange(0.0, scale+1e-3, 1.0)
x, y = numpy.meshgrid(axis, axis)
dm = numpy.concatenate((y.flatten().reshape((-1,1)), x.flatten().reshape((-1,1))), axis=1)

modes = ms.modes(dm)

for j in xrange(axis.shape[0]):
  for i in xrange(axis.shape[0]):
    loc = j*axis.shape[0] + i
    print '(%.1f,%.1f)' % (modes[loc,1], modes[loc,0]),
  print
