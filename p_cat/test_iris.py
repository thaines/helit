#! /usr/bin/env python

# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import time
from collections import defaultdict
import numpy

from p_cat import *



# First load and parse the data set...
f = open('iris/iris.data', 'r')

data_set = []
counts = defaultdict(int)

for l in f.readlines():
  parts = l.split(',')
  if len(parts)!=5: continue
  fv = numpy.empty(4, dtype=numpy.float32)
  fv[:] = map(float, parts[:4])
  name = parts[-1].strip()
  data_set.append((fv,name))
  counts[name] += 1

f.close()

print 'Found %i categories:'%len(counts)
for pair in counts.iteritems(): print '  %s: %i members'%pair
print



# Generate a test set and a trainning set...
train_size = min(counts.itervalues())//3
train = []
test = []

train_counts = defaultdict(int)
for fv, cat in data_set:
  if train_counts[cat]<train_size:
    train.append((fv,cat))
    train_counts[cat] += 1
  else:
    test.append((fv,cat))

print '%i selected for trainning, %i for testing'%(len(train), len(test))
print



# Create a bunch of models...
c_g   = ClassifyGaussian(4)
c_kde = ClassifyKDE(0.3*numpy.eye(4, dtype=numpy.float32))
c_dpgmm = ClassifyDPGMM(4)
c_df = ClassifyDF(4, 8)
c_df_kde = ClassifyKDE(0.3*numpy.eye(4, dtype=numpy.float32), 32, 8)

models = [('gaussian',c_g), ('kde',c_kde), ('dpgmm',c_dpgmm), ('df',c_df), ('df_kde',c_df_kde)]



# Setup their priors...
print 'Creating priors...'
speed = defaultdict(lambda: defaultdict(float))

for name,mod in models:
  start = time.clock()
  for fv, _ in train:
    mod.priorAdd(fv)
  end = time.clock()
  speed[name]['prior'] += end-start



# Train them...
print 'Training...'

for name,mod in models:
  start = time.clock()
  for fv, cat in train:
    mod.add(fv, cat)
  end = time.clock()
  speed[name]['train'] += end-start



# Test them...
print 'Testing...'
correct = defaultdict(lambda: defaultdict(int))

for name,mod in models:
  start = time.clock()
  for fv, cat in test:
    est = mod.getCat(fv)
    if est==cat:
      correct[name][cat] += 1
  end = time.clock()
  speed[name]['test'] += end-start


# Print out the results of testing, including time consumed...
print 'Time:'
for name, res_dict in speed.iteritems():
  print '  For model %s:'%name
  for task, length in res_dict.iteritems():
    print '    For task %s took %.4f seconds'%(task,length)
print

print 'Accuracy:'
for name, res_dict in correct.iteritems():
  print '  For model %s:'%name
  avg = 0.0
  for cat, count in res_dict.iteritems():
    total = counts[cat]-train_counts[cat]
    pcent = 100.0*float(count)/float(total)
    print '    Category %s: %i out of %i correct (%.2f%%)'%(cat, count, total, pcent)
    avg += pcent
  print '    (Average percentage of %.2f%%)'%(avg/len(res_dict))
print
