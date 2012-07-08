#! /usr/bin/env python

# Copyright 2012 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import time

import numpy
import numpy.random

from df import *


doMP = False



categories = ('undergrad', 'masters', 'phd', 'postdoc', 'lecturer')
train_count = {'undergrad':32, 'masters':32, 'phd':32, 'postdoc':32, 'lecturer':32}
test_count = {'undergrad':256, 'masters':256, 'phd':256, 'postdoc':256, 'lecturer':256}



# Define the parameters for the drawing of the features...
class Eyes:
  cont = False
  
  Normal = 0
  Repressed = 1
  Confused = 2
  Stressed = 3
  GodComplex = 4 
  
  
  dist = {'undergrad':(0.4,0.0,0.3,0.3,0.0), 'masters':(0.2,0.0,0.4,0.4,0.0), 'phd':(0.0,0.4,0.3,0.3,0.0), 'postdoc':(0.2,0.1,0.2,0.3,0.2), 'lecturer':(0.1,0.0,0.0,0.1,0.8)}


class Transport:
  cont = False
  
  Crawling = 0
  Walking = 1
  Cycling = 2
  Car = 3
  Lectica = 4
  
  dist = {'undergrad':(0.5,0.5,0.0,0.0,0.0), 'masters':(0.0,0.5,0.3,0.2,0.0), 'phd':(0.0,0.5,0.3,0.2,0.0), 'postdoc':(0.0,0.4,0.4,0.2,0.0), 'lecturer':(0.0,0.0,0.2,0.4,0.4)}


class Clothes:
  cont = False
  
  Naked = 0
  Rags = 1
  Casual = 3
  Smart = 4
  
  dist = {'undergrad':(0.1,0.8,0.1,0.0), 'masters':(0.0,0.8,0.2,0.0), 'phd':(0.0,0.8,0.2,0.0), 'postdoc':(0.0,0.0,0.6,0.4), 'lecturer':(0.25,0.25,0.25,0.25)}


class Brain:
  cont = True
  
  Intellect = 0
  Arrogance = 1
  Fear = 2
  
  length = 3
  
  mean = {'undergrad':(0.5,1.0,1.0), 'masters':(0.6,0.8,0.2), 'phd':(2.0,0.4,2.0), 'postdoc':(3.0,0.8,2.0), 'lecturer':(3.0,2.0,0.0)}
  
  covar = {'undergrad':((0.09,0.5,-0.1),(0.5,0.1,-0.4),(-0.1,-0.4,0.2)), 'masters':((0.04,0.5,-0.1),(0.5,0.1,-0.4),(-0.1,-0.4,0.2)), 'phd':((0.01,0.5,-0.1),(0.5,0.1,-0.4),(-0.1,-0.4,0.2)), 'postdoc':((0.01,0.5,-0.1),(0.5,0.1,-0.4),(-0.1,-0.4,0.2)), 'lecturer':((0.2,0.5,-0.1),(0.5,0.01,-0.4),(-0.1,-0.4,0.01))}


class Stomach:
  cont = True
  
  Fullness = 0
  
  length = 1
  
  mean = {'undergrad':(-0.2,), 'masters':(-0.3,), 'phd':(0.0,), 'postdoc':(1.0,), 'lecturer':(1.1,)}
  
  covar = {'undergrad':((0.7,),), 'masters':((0.6,),), 'phd':((0.5,),), 'postdoc':((0.3,),), 'lecturer':((0.1,),)}


attributes = (Eyes, Transport, Clothes, Brain, Stomach)
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

pos = 0
for cat in categories:
  cat_ind = categories.index(cat)
  for _ in xrange(train_count[cat]):
    int_dm[pos,:], real_dm[pos,:] = generate(cat)
    cats[pos] = cat_ind
    pos += 1

order = numpy.random.permutation(length)
int_dm = int_dm[order,:]
real_dm = real_dm[order,:]
cats = cats[order].reshape((-1,1))



# Generate the testing data...
test = dict()
for cat, count in test_count.iteritems():
  test[cat] = map(lambda _: MatrixFS(*generate(cat)), xrange(count))



# Function to create a suitable decision forest object reday for trainning...
def createDF():
  df = DF()
  df.setGoal(Classification(len(categories),2))
  df.setGen(MergeGen(DiscreteRandomGen(0,4,4),LinearRandomGen(1,2,4,8,4)))
  return df



# Function to do the test on a decision forest and return the relevent info...
def doTest(df):
  success = 0.0
  total = 0.0
  
  for cat, examples in test.iteritems():
    cat_ind = categories.index(cat)
    
    for mfs in examples:
      dist, best = df.evaluate(mfs, which = ['prob', 'best'])[0]
      total += 1.0
      if cat_ind==best: success += 1.0
    
  return success / total



# Loop through incrimental learning, also running batch learning for comparison, and printing a line of stats at each step...
incDF = createDF()
incDF.setInc(True)

growDF = createDF()
growDF.setInc(True, True)
growDF.setPruner(PruneCap(8, 8, 1e-3, 0))

es = MatrixGrow()

for i in xrange(64):
  # Update the data at this juncture...
  es.append(int_dm[i,:], real_dm[i,:], cats[i,:])
  
  # Update the incrimental model...
  incStart = time.time()
  incDF.learn(2, es, clamp = 8, mp=doMP)
  incEnd = time.time()
  
  # Update the incrimental model that does tree growth exclusivly...
  growStart = time.time()
  growDF.learn(8 if i==0 else 0, es, clamp = 8, mp=doMP)
  growEnd = time.time()
  
  # Batch train a model from scratch...
  batchDF = createDF()
  batchStart = time.time()
  batchDF.learn(8, es, mp=doMP)
  batchEnd = time.time()
  
  # Test them both...
  incRate = doTest(incDF) if incDF.size()!=0 else 0.0
  growRate = doTest(growDF) if growDF.size()!=0 else 0.0
  batchRate = doTest(batchDF) if batchDF.size()!=0 else 0.0
  
  # Print out the stats...
  print '% 3i: {Batch,Inc,GrowInc} = {%.1f%%,%.1f%%,%.1f%%} in {%.2fs,%.2fs,%.2fs} (cat = %i)'%(i+1, 100.0*batchRate, 100.0*incRate, 100.0*growRate, batchEnd-batchStart, incEnd-incStart, growEnd-growStart, cats[i,0])
