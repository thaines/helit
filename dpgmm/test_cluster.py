#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2012 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import numpy
import numpy.random
import random

from utils.prog_bar import ProgBar
from dpgmm import DPGMM



# Parameters...
trainCount = 1024
testCount = 256



# Generate a random set of Gaussians to test with...
print 'Generating model...'
count = random.randrange(2,6)
mix = numpy.random.rand(count) + 1.0
mix /= mix.sum()
mean = numpy.random.randint(-2,3,(count,3))
sd = 0.4*numpy.random.rand(count) + 0.1

for i in xrange(count):
  print '%i: mean = %s; sd = %f'%(i,str(mean[i,:]), sd[i]) 



# Draw two sets of samples from them...
print 'Generating data...'
train = []
for _ in xrange(trainCount):
  which = numpy.random.multinomial(1,mix).argmax()
  covar = sd[which]*numpy.identity(3)
  s = numpy.random.multivariate_normal(mean[which,:],covar)
  train.append(s)
  
test = []
for _ in xrange(testCount):
  which = numpy.random.multinomial(1,mix).argmax()
  covar = sd[which]*numpy.identity(3)
  s = numpy.random.multivariate_normal(mean[which,:],covar)
  test.append((s,which))



# Train a model...
print 'Trainning model...'
model = DPGMM(3)
for feat in train: model.add(feat)

model.setPrior() # This sets the models prior using the data that has already been added.
model.setConcGamma(1.0, 0.25) # Make the model a little less conservative about creating new categories..

p = ProgBar()
iters = model.solveGrow()
del p
print 'Solved model with %i iterations'%iters



# Classify the test set...
probs = model.stickProb(numpy.array(map(lambda t: t[0], test)))
catGuess = probs.argmax(axis=1)
catTruth = numpy.array(map(lambda t: t[1], test))

confusion_matrix = numpy.zeros((count, model.getStickCap()+1), dtype=numpy.int32)

for i in xrange(len(catGuess)):
  confusion_matrix[catTruth[i],catGuess[i]] += 1

print 'Confusion matrix [truth, guess], noting that class labels will not match:'
print confusion_matrix
