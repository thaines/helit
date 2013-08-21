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
means = [[3.0,2.0,4.0,-2.0], [2.0,-1.0,8.0,1.0], [5.0,-1.0,8.0,2.5], [0.0,0.0,4.0,0.5]]
quantity = 250

a = numpy.random.multivariate_normal(means[0], 0.1*numpy.eye(4), quantity)
b = numpy.random.multivariate_normal(means[1], 0.2*numpy.eye(4), quantity)
c = numpy.random.multivariate_normal(means[2], 0.3*numpy.eye(4), quantity)
d = numpy.random.multivariate_normal(means[3], 0.4*numpy.eye(4), quantity)

data = numpy.concatenate((a,b,c,d), axis=0)



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



# Calculate the modes for all vectors, then print out some randomly selected convergances...
res = ms.modes_data()

order = range(data.shape[0])
random.shuffle(order)

for i in order[:32]:
  print '%i:\n  mean  = %s\n  value = %s\n  mode  = %s' % (i, str(means[i//quantity]), str(data[i,:]), str(res[i,:]))
