#! /usr/bin/env python

# Copyright 2013 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import random
import numpy
import numpy.random

from ms import MeanShift



# Create a data set, 2D...
means = [[1.0,1.0], [4.0,4.0], [4.0,1.0], [1.0,4.0]]
quantity = 250

a = numpy.random.multivariate_normal(means[0], 0.1*numpy.eye(2), quantity)
b = numpy.random.multivariate_normal(means[1], 0.2*numpy.eye(2), quantity)
c = numpy.random.multivariate_normal(means[2], 0.3*numpy.eye(2), quantity)
d = numpy.random.multivariate_normal(means[3], 0.4*numpy.eye(2), quantity)

data = numpy.concatenate((a,b,c,d), axis=0)



# Add weights to it...
weights = numpy.zeros(quantity*4)
weights[0:quantity] = 0.1 / 0.4
weights[quantity:2*quantity] = 0.2 / 0.4
weights[2*quantity:3*quantity] = 0.3 / 0.4
weights[3*quantity:4*quantity] = 0.4 / 0.4

data = numpy.concatenate((data, weights.reshape((-1,1))), axis=1)



# Use mean shift to cluster it...
ms = MeanShift()
ms.set_data(data, 'df', 2)

ms.set_kernel(random.choice(filter(lambda s: s!='fisher', ms.kernels())))
ms.set_spatial(random.choice(ms.spatials()))

modes, indices = ms.cluster()



# Print out basic stats...
print 'kernel = %s; spatial = %s' % (ms.get_kernel(), ms.get_spatial())
print 'exemplars = %i; features = %i' % (ms.exemplars(), ms.features())
print 'quality = %.3f; epsilon = %.3f; iter_cap = %i' % (ms.quality, ms.epsilon, ms.iter_cap)
print 'weight = %.1f' % ms.weight()
print



# Create a grid of samples...
scale = 5.0
axis = numpy.arange(0.0, scale+1e-3, 0.25)
x, y = numpy.meshgrid(axis, axis)
dm = numpy.concatenate((y.flatten().reshape((-1,1)), x.flatten().reshape((-1,1))), axis=1)

clusters = ms.assign_clusters(dm)

for j in xrange(axis.shape[0]):
  for i in xrange(axis.shape[0]):
    loc = j*axis.shape[0] + i
    print clusters[loc],
  print
