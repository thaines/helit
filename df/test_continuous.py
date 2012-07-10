#! /usr/bin/env python

# Copyright 2012 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import time

import numpy
import numpy.random

from utils.prog_bar import ProgBar

from df import *

try: from svm import svm
except: svm = None



# A toy problem using continuous attributes only - a simulation of a politician/marketing/tele-sales differentiator.



# Define the parameters for the drawing of the attributes...
class Style:
  Loudness = 0
  Speed = 1

  length = 2

  politician_mean = (1.0,0.0)
  politician_covar = ((0.3,0.0),(0.0,0.3))

  marketing_mean = (0.5,0.8)
  marketing_covar = ((0.5,0.0),(0.0,0.5))

  tele_sales_mean = (0.5,0.3)
  tele_sales_covar = ((0.2,0.0),(0.0,1.0))


class Content:
  Bullshit = 0
  Lies = 1
  Exaggeration = 2

  length = 3

  politician_mean = (1.0,1.0,0.9)
  politician_covar = ((0.1,0.5,0.5),(0.5,0.1,0.5),(0.5,0.5,0.1))

  marketing_mean = (0.9,0.4,1.0)
  marketing_covar = ((0.3,0.4,-0.4),(0.4,0.2,0.4),(-0.4,0.1,0.4))

  tele_sales_mean = (0.7,0.7,0.7)
  tele_sales_covar = ((0.5,0.0,0.0),(0.0,1.0,0.0),(0.0,0.0,0.6))


class Clothes:
  Suit = 0
  Casual = 1

  length = 2

  politician_mean = (1.0,0.0)
  politician_covar = ((0.1,-1.0),(-1.0,0.2))

  marketing_mean = (0.5,0.5)
  marketing_covar = ((0.3,-0.1),(-0.1,0.2))

  tele_sales_mean = (0.0,1.0)
  tele_sales_covar = ((0.4,-0.5),(-0.5,0.4))


attributes = [Style, Content, Clothes]



# Functions to generate examples of the three classes...
def make_politician():
  length = sum(map(lambda a: a.length, attributes))
  ret = numpy.empty(length, dtype=numpy.float32)

  offset = 0
  for att in attributes:
    ret[offset:offset+att.length] = numpy.random.multivariate_normal(att.politician_mean, att.politician_covar)
    offset += att.length

  return ret

def make_marketing():
  length = sum(map(lambda a: a.length, attributes))
  ret = numpy.empty(length, dtype=numpy.float32)

  offset = 0
  for att in attributes:
    ret[offset:offset+att.length] = numpy.random.multivariate_normal(att.marketing_mean, att.marketing_covar)
    offset += att.length

  return ret

def make_tele_sales():
  length = sum(map(lambda a: a.length, attributes))
  ret = numpy.empty(length, dtype=numpy.float32)

  offset = 0
  for att in attributes:
    ret[offset:offset+att.length] = numpy.random.multivariate_normal(att.tele_sales_mean, att.tele_sales_covar)
    offset += att.length

  return ret



# Generate the trainning set...
politician_count = 32
marketing_count = 32
tele_sales_count = 32
total_count = politician_count + marketing_count + tele_sales_count

feat_length = sum(map(lambda a: a.length, attributes))
dm = numpy.empty((total_count, feat_length), dtype=numpy.float32)
cat = numpy.empty(total_count, dtype=numpy.int32)

for i in xrange(total_count):
  if i<politician_count:
    dm[i,:] = make_politician()
    cat[i] = 0
  elif i<(politician_count+marketing_count):
    dm[i,:] = make_marketing()
    cat[i] = 1
  else:
    dm[i,:] = make_tele_sales()
    cat[i] = 2

cat = cat.reshape((-1,1))
es = MatrixES(dm, cat)



# Generate the testing set...
politician_test = 256
politician = map(lambda _: make_politician(), xrange(politician_test))

marketing_test = 256
marketing = map(lambda _: make_marketing(), xrange(marketing_test))

tele_sales_test = 256
tele_sales = map(lambda _: make_tele_sales(), xrange(tele_sales_test))



# Define a function to run the test on a specific generator...
def doTest(gen):
  # Train the model...
  df = DF()
  df.setGoal(Classification(3,1)) # 3 = # of classes, 1 = channel of truth for trainning.
  df.setGen(gen)
  
  pb = ProgBar()
  df.learn(8, es, callback = pb.callback) # 8 = number of trees to learn. dm is in channel 0, cat in channel 1.
  del pb
  
  # Drop some stats...
  print '%i trees containing %i nodes.\nAverage error is %.3f.'%(df.size(), df.nodes(), df.error())


  # Test...
  politician_success = 0
  politician_prob = 0.0
  res = df.evaluate(MatrixES(numpy.asarray(politician)), which = ['prob', 'best'])
  for i in xrange(politician_test):
    dist, best = res[i]
    if 0==best: politician_success += 1
    politician_prob += dist[0]

  print 'Of %i politicians %i (%.1f%%) were correctly detected, with %.1f%% of total probability.'%(politician_test, politician_success, 100.0*politician_success/float(politician_test), 100.0*politician_prob/politician_test)
  
  marketing_success = 0
  marketing_prob = 0.0
  res = df.evaluate(MatrixES(numpy.asarray(marketing)), which = ['prob', 'best'], mp=False)
  for i in xrange(marketing_test):
    dist, best = res[i]
    if 1==best: marketing_success += 1
    marketing_prob += dist[1]

  print 'Of %i marketers %i (%.1f%%) were correctly detected, with %.1f%% of total probability.'%(marketing_test, marketing_success, 100.0*marketing_success/float(marketing_test), 100.0*marketing_prob/marketing_test)

  tele_sales_success = 0
  tele_sales_prob = 0.0
  for i in xrange(tele_sales_test):
    dist, best = df.evaluate(MatrixES(tele_sales[i]), which = ['prob', 'best'])[0]
    if 2==best: tele_sales_success += 1
    tele_sales_prob += dist[2]

  print 'Of %i tele-sellers %i (%.1f%%) were correctly detected, with %.1f%% of total probability.'%(tele_sales_test, tele_sales_success, 100.0*tele_sales_success/float(tele_sales_test), 100.0*tele_sales_prob/tele_sales_test)
  
  total_success = politician_success + marketing_success + tele_sales_success
  total_test = politician_test + marketing_test + tele_sales_test
  print 'Combined success is %i out of %i (%.1f%%)'%(total_success, total_test, 100.0*total_success/float(total_test))



# Run the test on a set of generators...
print 'Axis-aligned median generator:'
doTest(AxisMedianGen(0,2)) # 0 = channel to use to generate tests, 2 = # of tests to try.
print

print 'Linear median generator:'
doTest(LinearMedianGen(0,2,4,8)) # 0 = channel to use to generate tests, 2 = # of dimensions for each test, 4 = # of dimension possibilities to consider, 8 = # of orientations to consider.
print

print 'Axis-aligned random generator:'
doTest(AxisRandomGen(0,4,8)) # 0 = channel to generate tests for, 4 = # of dimensions to try splits for, 8 = # of splits to try per dimension.
print

print 'Linear random generator:'
doTest(LinearRandomGen(0,2,4,8,4)) # 0 = channel to generate tests for, 2 = # of dimensions used for each test, 4 = number of random dimension selections to try, 8 = number of random directions to try, 4 = number of random split points to try.
print

print 'Axis-aligned classify generator:'
doTest(AxisClassifyGen(0,1,3)) # 0 = channel to get features for tests from, 1 = channel containing the actual answer to optimise, 3 = number of tests to generate.
print

print 'Linear classify generator:'
doTest(LinearClassifyGen(0,1,2,4,8)) # 0 = channel to use to generate tests,1 = channel containing the actual answer to optimise, 2 = # of dimensions for each test, 4 = # of dimension possibilities to consider, 8 = # of orientations to consider.
print

if svm!=None:
  print 'SVM generator:'
  params = svm.ParamsSet(True)
  doTest(SVMClassifyGen(params, 8, 1, 2, 0, 3, 2))
  print
