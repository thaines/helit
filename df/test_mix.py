#! /usr/bin/env python

# Copyright 2012 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import time

import numpy
import numpy.random

from df import *



categories = ('minion_muscle', 'minion_scientist', 'competition', 'guinea_pig', 'victim')

train_count = {'minion_muscle':16, 'minion_scientist':16, 'competition':16, 'guinea_pig':16, 'victim':64}
train_weight = {'minion_muscle':1.0, 'minion_scientist':1.0, 'competition':1.0, 'guinea_pig':1.0, 'victim':0.25}
test_count = {'minion_muscle':256, 'minion_scientist':256, 'competition':256, 'guinea_pig':256, 'victim':256}


class Eyesight:
  cont = False
  
  Glasses = 0
  TwentyTwenty = 1
  WompRat = 2

  dist = {'minion_muscle':(0.0,0.4,0.6), 'minion_scientist':(0.5,0.3,0.2), 'competition':(0.1,0.2,0.7), 'guinea_pig':(0.4,0.4,0.2), 'victim':(0.5,0.3,0.2)}


class MindSet:
  cont = False
  
  Philosopher = 0
  Intelectual = 1
  Religious = 2
  Thoughtless = 3

  dist = {'minion_muscle':(0.0,0.0,0.5,0.5), 'minion_scientist':(0.5,0.5,0.0,0.0), 'competition':(0.6,0.2,0.2,0.0), 'guinea_pig':(0.0,0.0,0.6,0.4), 'victim':(0.1,0.2,0.3,0.4)}


class Body:
  cont = True
  
  Muscle = 0
  Height = 1
  Width = 2

  length = 3

  mean = {'minion_muscle':(1.0,2.0,0.6), 'minion_scientist':(0.1,1.6,0.3), 'competition':(1.0,2.4,0.2), 'guinea_pig':(0.5,1.3,0.4), 'victim':(0.5,1.8,0.4)}
  
  covar = {'minion_muscle':((0.1,0.2,0.2),(0.2,0.2,-0.3),(0.2,-0.3,0.8)), 'minion_scientist':((0.4,0.0,0.0),(0.0,0.4,-0.2),(0.0,-0.2,0.4)), 'competition':((0.1,0.0,0.0),(0.0,0.1,0.0),(0.0,0.0,0.1)), 'guinea_pig':((0.4,0.1,0.1),(0.1,0.4,-0.2),(0.1,-0.2,0.4)), 'victim':((0.4,0.2,0.2),(0.2,0.5,-0.3),(0.2,-0.3,0.5))}


class Brains:
  cont = True
  
  Intellect = 0
  Bravery = 1
  Submissive = 2

  length = 3

  mean = {'minion_muscle':(0.1,0.8,1.0), 'minion_scientist':(1.0,0.0,0.7), 'competition':(1.0,1.0,0.0), 'guinea_pig':(0.1,0.1,1.0), 'victim':(0.1,0.1,0.1)}

  covar = {'minion_muscle':((0.2,-0.2,-0.5),(-0.2,0.1,0.0),(-0.5,0.0,0.5)), 'minion_scientist':((0.3,0.0,-0.4),(0.0,0.1,0.0),(-0.4,0.0,0.4)), 'competition':((0.1,0.0,0.0),(0.0,0.1,0.0),(0.0,0.0,0.1)), 'guinea_pig':((0.3,-0.2,-0.5),(-0.2,0.3,0.0),(-0.5,0.0,0.6)), 'victim':((0.3,-0.2,-0.2),(-0.2,0.3,0.0),(-0.2,0.0,0.3))}


attributes = (Eyesight, MindSet, Body, Brains)
int_length = len(filter(lambda a: a.cont==False, attributes))
real_length = sum(map(lambda a: a.length if a.cont else 0, attributes))



# Function that can be given an entry from categories and returns an instance, as a tuple of (int vector, real vector)...
def generate(cat):
  int_ret = numpy.empty(int_length, dtype=numpy.int32)
  real_ret = numpy.empty(real_length, dtype=numpy.float32)
  int_offset = 0
  real_offset = 0

  for att in attributes:
    if att.cont:
      real_ret[real_offset:real_offset+att.length] = numpy.random.multivariate_normal(att.mean[cat], att.covar[cat])
      real_offset += att.length
    else:
      int_ret[int_offset] = numpy.nonzero(numpy.random.multinomial(1,att.dist[cat]))[0][0]
      int_offset += 1

  return (int_ret, real_ret)



# Generate the trainning data...
length = sum(train_count.itervalues())
int_dm = numpy.empty((length, int_length), dtype=numpy.int32)
real_dm = numpy.empty((length, real_length), dtype=numpy.float32)
cats = numpy.empty(length, dtype=numpy.int32)
weights = numpy.empty(length, dtype=numpy.float32)

pos = 0
for cat in categories:
  cat_ind = categories.index(cat)
  for _ in xrange(train_count[cat]):
    int_dm[pos,:], real_dm[pos,:] = generate(cat)
    cats[pos] = cat_ind
    weights[pos] = train_weight[cat]
    pos += 1

cats = cats.reshape((-1,1))
weights = weights.reshape((-1,1))

es = MatrixES(int_dm, real_dm, cats, weights)



# Generate the testing data...
test = dict()
for cat, count in test_count.iteritems():
  test[cat] = map(lambda _: MatrixES(*generate(cat)), xrange(count))



# Define a function to run the test on a specific generator...
def doTest(gen):
  # Train the model...
  df = DF()
  df.setGoal(Classification(5,2)) # 5 = # of classes, 2 = channel of truth for trainning.
  df.setGen(gen)
  
  start = time.time()
  df.learn(8, es, weightChannel=3) # 8 = number of trees to learn, weightChannel is where to find the weighting.
  end = time.time()
  
  # Drop some stats...
  print '%i trees containing %i nodes trained in %.3f seconds.\nAverage error is %.3f.'%(df.size(), df.nodes(), end-start, df.error())
  
  
  # Test the model...
  success = dict()
  prob = dict()
  
  for cat, examples in test.iteritems():
    success[cat] = 0
    prob[cat] = 0.0
    cat_ind = categories.index(cat)
    
    for mfs in examples:
      dist, best = df.evaluate(mfs, which = ['prob', 'best'])[0]
      if cat_ind==best: success[cat] += 1
      prob[cat] += dist[cat_ind]
    
    print 'Of %i %s %i (%.1f%%) were correctly detected, with %.1f%% of total probability.'%(len(examples), cat, success[cat], 100.0*success[cat]/float(len(examples)), 100.0*prob[cat]/len(examples))
  
  total_success = sum(success.itervalues())
  total_test = sum(map(lambda l: len(l), test.itervalues()))
  print 'Combined success is %i out of %i (%.1f%%)'%(total_success, total_test, 100.0*total_success/float(total_test))



# Run the test on a set of composite generators...
print 'RandomGen:'
doTest(RandomGen(1,DiscreteRandomGen(0,4,4),LinearRandomGen(1,2,4,8,4)))
print

print 'MergeGen:'
doTest(MergeGen(DiscreteRandomGen(0,4,4),LinearRandomGen(1,2,4,8,4)))
print
