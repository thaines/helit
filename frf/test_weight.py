#! /usr/bin/env python

# Copyright 2014 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import frf
import numpy



# Parameters...
seed_count = 16
dims = 6
cats = 4
train = 4096
test = 1024
trees = 8
halflife = 0.1


# Draw random seeds in a n-d unit cube, assign a class to each...
seeds = []

for i in xrange(seed_count):
  loc = numpy.random.random(size=dims)
  cat = i % cats
  seeds.append((loc, cat))



# Function to generate data - draw points in cube, and assign...
def sample():
  """Returns (feature, class)"""
  # Select a feature vector...
  feat = numpy.random.random(size=dims)
  
  # Find two closest seeds...
  close = []
  for seed in seeds:
    dist = numpy.sqrt(numpy.square(seed[0] - feat).sum())
    close.append((dist, seed[1]))
  
  close.sort()
  close = close[:2]
  
  # Assign classification, depending on their relevant distances...
  if close[0][0] * 2.0 < close[1][0]:
    cat = close[0][1]
  else:
    prob_first = close[1][0] / close[0][0]
    
    if numpy.random.random()< prob_first:
      cat = close[0][1]
    else:
      cat = close[1][1]
  
  return (feat, cat)



# Create a trainning data set...
train_feat   = numpy.empty((train, dims), dtype=numpy.float32)
train_cat    = numpy.empty(train, dtype=numpy.int32)
train_weight = numpy.empty(train, dtype=numpy.float32)

for i in xrange(train):
  feat, cat = sample()
  
  train_feat[i,:] = feat
  train_cat[i] = cat
  train_weight[i] = halflife / (halflife + cat)



# Train a rf...
forest = frf.Forest()
forest.configure('C', 'C', 'S'*dims)
forest.min_exemplars = 4

oob = forest.train(train_feat, [train_cat, ('w', train_weight)], trees)

print 'oob = %f'%oob



# Create a test set...
test_feat   = numpy.empty((test, dims), dtype=numpy.float32)
test_cat    = numpy.empty(test, dtype=numpy.int32)
test_weight = numpy.empty(test, dtype=numpy.float32)

for i in xrange(test):
  feat, cat = sample()
  
  test_feat[i,:] = feat
  test_cat[i] = cat
  test_weight[i] = halflife / (halflife + cat)



# Get estimates for test set...
results = forest.predict(test_feat)
pred_cat = numpy.argmax(results[0]['prob'], axis=1)

print 'Percentage right = %.1f%%' % (100.0*(test_cat==pred_cat).sum()/float(test))
print



# Print out per-category reliability - should reflect the weighting...
correct = numpy.zeros(cats, dtype=numpy.int32)
total = numpy.zeros(cats, dtype=numpy.int32)

for i in xrange(test):
  total[test_cat[i]] += 1
  if test_cat[i]==pred_cat[i]:
    correct[test_cat[i]] += 1

for i in xrange(cats):
  if total[i]!=0:
    print 'class %i: %i of %i correct (%.1f%%)' % (i, correct[i], total[i], 100.0 * correct[i] / float(total[i]))
print



# Test the error method...
print 'Unweighted test error = %f' % forest.error(test_feat, test_cat)
print 'Weighted test error = %f' % forest.error(test_feat, (test_cat, ('w', test_weight)))
