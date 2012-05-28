#! /usr/bin/env python

# Copyright 2012 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import time

import numpy
import numpy.random

from df import *



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
zombie_count = 16
human_count = 64
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

cat = cat.reshape((-1,1))
es = MatrixES(dm, cat)



# Generate the testing set...
zombie_test = 256
zombie = map(lambda _: make_zombie(), xrange(zombie_test))

human_test = 256
human = map (lambda _: make_human(), xrange(human_test))



# Define a function to run the test on a specific generator...
def doTest(gen):
  # Train the model...
  df = DF()
  df.getPruner().setMinTrain(2)
  df.setGoal(Classification(2,1)) # 2 = # of classes, 1 = channel of truth for trainning.
  df.setGen(gen)
  
  start = time.time()
  df.learn(8, es, mp = False) # 8 = number of trees to learn. dm is in channel 0, cat in channel 1.
  end = time.time()
  
  # Drop some stats...
  print '%i trees containing %i nodes trained in %.3f seconds.\nAverage error is %.3f.'%(df.size(), df.nodes(), end-start, df.error())


  # Test...
  zombie_success = 0
  zombie_gen_success = 0
  zombie_prob = 0.0
  zombie_gen_prob = 0.0
  for i in xrange(zombie_test):
    dist, best, gen = df.evaluate(MatrixES(zombie[i]), which = ['prob', 'best', 'gen'])[0]
    if 0==best: zombie_success += 1
    if gen.argmax()==0: zombie_gen_success += 1
    zombie_prob += dist[0]
    zombie_gen_prob += gen[0]

  print 'Of %i zombies %i (%.1f%%) were correctly detected, with %.1f%% of total probability.'%(zombie_test, zombie_success, 100.0*zombie_success/float(zombie_test), 100.0*zombie_prob/zombie_test)
  print 'Generative: Of %i zombies %i (%.1f%%) were correctly detected, with %.1f%% of total probability.'%(zombie_test, zombie_gen_success, 100.0*zombie_gen_success/float(zombie_test), 100.0*zombie_gen_prob/zombie_test)
  
  human_success = 0
  human_gen_success = 0
  human_prob = 0.0
  human_gen_prob = 0.0
  for i in xrange(human_test):
    dist, best, gen = df.evaluate(MatrixES(human[i]), which = ['prob', 'best', 'gen'])[0]
    if 1==best: human_success += 1
    if gen.argmax()==1: human_gen_success += 1
    human_prob += dist[1]
    human_gen_prob += gen[1]

  print 'Of %i humans %i (%.1f%%) were correctly detected, with %.1f%% of total probability.'%(human_test, human_success, 100.0*human_success/float(human_test), 100.0*human_prob/human_test)
  print 'Generative: Of %i humans %i (%.1f%%) were correctly detected, with %.1f%% of total probability.'%(human_test, human_gen_success, 100.0*human_gen_success/float(human_test), 100.0*human_gen_prob/human_test)
  
  total_success = zombie_success + human_success
  total_gen_success = zombie_gen_success + human_gen_success
  total_test = zombie_test + human_test
  print 'Combined success is %i out of %i (%.1f%%)'%(total_success, total_test, 100.0*total_success/float(total_test))
  print 'Generative: Combined success is %i out of %i (%.1f%%)'%(total_gen_success, total_test, 100.0*total_gen_success/float(total_test))



# Run the test on a set of generators...
print 'Discrete random generator:'
doTest(DiscreteRandomGen(0,4,4)) # 0 = channel to use to generate tests, 4 = # of features to randomly select and generate test for, 4 = # of tests to generate per feature.
print

print 'Discrete classify generator:'
doTest(DiscreteClassifyGen(0,1,2,4)) # 0 = channel to use to generate tests, 1 = channel that contains the true categories, 2 = # of features to randomly select and generate tests for, 4 = # of tests to generate per feature.
print
