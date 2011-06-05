#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import numpy
from gcp import *



# Tests the incrimental update of a conjugate prior - generates a bunch of samples and selects a random subset - it then generates the conjugate prior for the subset with two methods - adding the samples in the subset; and adding all samples before removing those not in the subset. The two priors are then compared, to check that they are identical, as they mathematically should be, though in reality floating point error means they will only be close.



# Parameters...
dims = 2
mean = numpy.array([5.0,-3.0])
covar = numpy.array([[100.0,30.0],[30.0,50.0]])
target_count = 8
extra_count = 64



# Generate the two sets of samples...
gt = Gaussian(dims)
gt.setMean(mean)
gt.setCovariance(covar)

target = map(lambda _: gt.sample(),xrange(target_count))
extra = map(lambda _: gt.sample(),xrange(extra_count))



# Calculate the conjugate posterior for target using the two methods...
## Addative...
gp_add = GaussianPrior(dims)
gp_add.addPrior(mean, covar)

for sam in target: gp_add.addSample(sam)


## Addative then subtractive...
gp_add_sub = GaussianPrior(dims)
gp_add_sub.addPrior(mean, covar)

for sam in extra:  gp_add_sub.addSample(sam)
for sam in target: gp_add_sub.addSample(sam)
for sam in extra:  gp_add_sub.remSample(sam)



# Output diagnostics...
print 'add:'
print str(gp_add)

print 'add-sub:'
print str(gp_add_sub)

print
