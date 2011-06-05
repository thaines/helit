#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import random
import numpy
from gcp import *



# Tests the weighted update of a conjugate prior - generates a bunch of samples and then generates a conjugate prior by either adding the samples,  or by splitting them up into fractional samples and adding them in a randomised order - if its correct the same answer should present for both techneques. Also tries batch adding...



# Parameters...
dims = 2
mean = numpy.array([-2.5,24.0])
covar = numpy.array([[40.0,30.0],[30.0,50.0]])
sample_count = 16



# Generate the two set of samples...
gt = Gaussian(dims)
gt.setMean(mean)
gt.setCovariance(covar)

samples = map(lambda _: gt.sample() ,xrange(sample_count))



# Break the samples up into lots of small chunks, shuffle...
shards = []
for sam in samples:
  split = random.randint(1,5)
  weight = 1.0 / float(split)
  for _ in xrange(split):
    shards.append((sam,weight))

random.shuffle(shards)



# Construct the three conjugate posteriors...
whole = GaussianPrior(dims)
whole.addPrior(mean, covar)

for sam in samples: whole.addSample(sam)


fractions = GaussianPrior(dims)
fractions.addPrior(mean, covar)

for sam, weight in shards: fractions.addSample(sam, weight)


batch = GaussianPrior(dims)
batch.addPrior(mean, covar)

blocks = map(lambda _: list(), xrange(len(shards)/3))
for sam, weight in shards:
  blocks[random.randrange(len(blocks))].append((sam,weight))

for b in blocks:
  if len(b)!=0:
    sam = numpy.empty((len(b),dims))
    weight = numpy.empty(len(b))
    for i,pair in enumerate(b):
      sam[i,:] = pair[0]
      weight[i] = pair[1]
    batch.addSamples(sam, weight)



# Output diagnostics...
print 'whole:'
print str(whole)

print 'fractions:'
print str(fractions)

print 'batch:'
print str(batch)

print
