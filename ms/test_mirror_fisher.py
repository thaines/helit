#! /usr/bin/env python

# Copyright 2014 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import random
import numpy
import numpy.random

import cv
from utils.cvarray import *

from ms import MeanShift



# Test the mirrored version of the von_mises Fisher distribution, this time in 5D...

# Create a dataset - just a bunch of points in one direction, so we can test the mirroring effect (Abuse MeanShift object to do this)...
print 'Mirrored draws:'

vec = numpy.array([1.0, 0.5, 0.0, -0.5, -1.0])
vec /= numpy.sqrt(numpy.square(vec).sum())

print 'Base dir =', vec

draw = MeanShift()
draw.set_data(vec, 'f')
draw.set_kernel('fisher(256.0)')

data = draw.draws(32)

#print 'Input:'
#print data



# Create a mean shift object from the draws, but this time with a mirror_fisher kernel...
mirror = MeanShift()
mirror.set_data(data, 'df')
mirror.set_kernel('mirror_fisher(64.0)')

resample = mirror.draws(16)

for row in resample:
  print '[%6.3f %6.3f %6.3f %6.3f %6.3f]' % tuple(row)
print



# Test probabilities by ploting them...
print 'Probability single 2D mirror Fisher:'
mirror = MeanShift()
mirror.set_data(numpy.array([1.0, 0.0]), 'f')
mirror.set_kernel('mirror_fisher(16.0)')

angles = numpy.linspace(0.0, numpy.pi*2.0, num=70, endpoint=False)
vecs = numpy.concatenate((numpy.cos(angles)[:,numpy.newaxis], numpy.sin(angles)[:,numpy.newaxis]),axis=1)
probs = mirror.probs(vecs)
probs /= probs.max()

steps = 12
for i in xrange(steps):
  threshold = 1.0 - (i+1)/float(steps)
  
  print ''.join(map(lambda p: '#' if p>threshold else ' ', probs))
print



# Test that mean shift still works; also test loo bandwidth estimation...

## Abuse the mean shift object to draw 8 uniform locations on a sphere...





# Try out multiplication, and throughroughly verify it works; for more fun include weighting...


