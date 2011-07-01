# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import numpy
import numpy.random

# Defines a model for testing, that includes everything - multiple categories, discrete and continuous, weights,,,
# Toy problem revolves around classifying people from the point of view of an evil genius.



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



# Function that returns a tuple of (int data matrix, real data matrix, category vector, weight vector), to be used for trainning...
def generate_train():
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

  return (int_dm, real_dm, cats, weights)



# Given a function that takes 2 parameters - (int vector, real vector) and returns a dictionary representing the distribution for the classification prints out a report sumarising how good that function does...
def test(get_dist):
  for cat in categories:
    cat_ind = categories.index(cat)
    print 'Testing %s:'%cat
    correct = 0
    for _ in xrange(test_count[cat]):
      int_vec, real_vec = generate(cat)
      dist = get_dist(int_vec,real_vec)
      max_prob = max(dist.itervalues())
      if cat_ind in dist and dist[cat_ind]==max_prob: correct += 1
    print 'Got %i out of %i correct (%.1f%%)'%(correct, test_count[cat], 100.0*correct/float(test_count[cat]))

# Same as test, except it blocks the requests using a function that takes data matrics and returns lists of distributions instead...
def test_multi(get_dist_multi):
  for cat in categories:
    cat_ind = categories.index(cat)
    print 'Testing %s:'%cat
    
    int_dm = numpy.empty((test_count[cat],int_length), dtype=numpy.int32)
    real_dm = numpy.empty((test_count[cat],real_length), dtype=numpy.float32)
    for i in xrange(test_count[cat]):
      int_dm[i,:], real_dm[i,:] = generate(cat)

    dists = get_dist_multi(int_dm,real_dm)

    correct = 0
    for i in xrange(test_count[cat]):
      max_prob = max(dists[i].itervalues())
      if cat_ind in dists[i] and dists[i][cat_ind]==max_prob: correct += 1
    print 'Got %i out of %i correct (%.1f%%)'%(correct, test_count[cat], 100.0*correct/float(test_count[cat]))
