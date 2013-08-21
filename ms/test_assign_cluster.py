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



# Use mean shift to cluster it...
ms = MeanShift()
ms.set_data(data, 'df')

ms.set_kernel(random.choice(filter(lambda s: s!='fisher', ms.kernels())))
ms.set_spatial(random.choice(ms.spatials()))

modes, indices = ms.cluster()



# Print out basic stats...
print 'kernel = %s; spatial = %s' % (ms.get_kernel(), ms.get_spatial())
print 'exemplars = %i; features = %i' % (ms.exemplars(), ms.features())
print 'quality = %.3f; epsilon = %.3f; iter_cap = %i' % (ms.quality, ms.epsilon, ms.iter_cap)
print



# Print out a grid of cluster assignments...
for j in xrange(20):
  for i in xrange(20):
    fv = numpy.array([0.25*j, 0.25*i])
    c = ms.assign_cluster(fv)
    print c,
  print
