#! /usr/bin/env python

# Copyright 2014 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import frf

import numpy



# A toy problem based on discrete attributes only - a simulation of a zombie detector.



# Define a set of attributes and the probabilities for when generating the two classes...
class Speed:
  Shuffle = 0
  Walk = 1
  Run = 2

  zombie = [0.9,0.1,0.0]
  human = [0.2,0.4,0.4]

class Complection:
  Necrotic = 0
  Rough = 1
  Clean = 2

  zombie = [0.7,0.3,0.0]
  human = [0.0,0.4,0.6]

class Colour:
  Pale = 0
  White = 1
  Black = 2

  zombie = [0.5,0.0,0.5]
  human = [0.0,0.5,0.5]

class Vocal:
  Braaaiins = 0
  Silence = 1
  Help = 2
  Hello = 3

  zombie = [0.2,0.8,0.0,0.0]
  human = [0.0,0.7,0.1,0.2]

class Clothes:
  Naked = 0
  Rags = 1
  Casual = 2
  Suit = 3

  zombie = [0.2,0.8,0.0,0.0]
  human = [0.1,0.3,0.4,0.2]

attributes = [Speed,Complection,Colour,Vocal,Clothes]



# Functions to generate exemplars of the two classes...
def make_zombie():
  ret = numpy.empty(len(attributes), dtype=numpy.int32)
  for i,attribute in enumerate(attributes):
    ret[i] = numpy.nonzero(numpy.random.multinomial(1,attribute.zombie))[0][0]
  return ret

def make_human():
  ret = numpy.empty(len(attributes), dtype=numpy.int32)
  for i,attribute in enumerate(attributes):
    ret[i] = numpy.nonzero(numpy.random.multinomial(1,attribute.human))[0][0]
  return ret



# Generate the trainning set...
zombie_count = 32
human_count = 32
total_count = zombie_count + human_count

dm = numpy.empty((total_count, len(attributes)), dtype=numpy.int32)
cat = numpy.empty(total_count, dtype=numpy.int32)

for i in xrange(total_count):
  if i<zombie_count:
    dm[i,:] = make_zombie()
    cat[i] = 0
  else:
    dm[i,:] = make_human()
    cat[i] = 1



# Train the model...
forest = frf.Forest()
forest.configure('C', 'C', 'OOOOO')
forest.min_exemplars = 4

oob = forest.train(dm, cat, 4)

print 'Made forest (oob = %.2f%%):' % ((1.0 - oob[0]) * 100.0)
for i in xrange(min(len(forest),4)):
  print '  Tree %i: %i bytes, %i nodes' % (i, forest[i].size, forest[i].nodes())
print


# Test...
zombie_test = 256
zombie_success = 0
for i in xrange(zombie_test):
  z = make_zombie()
  dist = forest.predict(z[numpy.newaxis,:], 0)[0]
  if dist['prob'][0]>dist['prob'][1]:
    zombie_success += 1

print 'Of %i zombies %i (%.1f%%) were correctly detected.'%(zombie_test, zombie_success, 100.0*zombie_success/float(zombie_test))
  
human_test = 256
human_success = 0
for i in xrange(human_test):
  h = make_human()
  dist = forest.predict(h[numpy.newaxis,:], 0)[0]
  if dist['prob'][1]>dist['prob'][0]:
    human_success += 1

print 'Of %i humans %i (%.1f%%) were correctly detected.'%(human_test, human_success, 100.0*human_success/float(human_test))
