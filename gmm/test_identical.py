#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import time

import numpy

import gmm


def testSimple(obj):
  # Two well seperated clusters are tested...
  feats = numpy.random.rand(50,2)
  feats[:25,0] -= 1.5
  feats[25:,0] += 0.5
  feats[:,1] -= 0.5

  obj.train(feats,2)

  for i in xrange(2):
    print 'centre',i,':',obj.getCentre(i)


def testSimpleNormal(obj,num=100,silent=False):
  # Test with 3 samples drawn from normal distributions, two overlapping...
  covar = numpy.array([[0.5,0.0],[0.0,0.5]])
  feats = numpy.random.multivariate_normal(numpy.zeros(2),covar,num*3)
  feats[:num,0] -= 1.0
  feats[num:num*2,0] += 1.0
  feats[num*2:,0] += 1.0
  feats[num:num*2,1] += 0.5
  feats[num*2:,1] -= 0.5

  obj.train(feats,3)

  if not silent:
    for i in xrange(3):
      print 'centre',i,':',obj.getCentre(i)



def test(obj):
  # Given a k-means object tests multiple data sets on it...
  print 'Two clusters, clean seperation... (-1,0) and (1,0)'
  testSimple(obj)
  print 'Three clusters, Gaussian drawn, two overlap... (-1,0), (1,0.5) and (1,-0.5)'
  testSimpleNormal(obj)

def speedTest(obj):
  tStart = time.clock()
  for _ in xrange(100): testSimpleNormal(obj,num=1000,silent=True)
  tEnd = time.clock()
  print 'Runtime =',(tEnd-tStart)


def testAll():
  print 'Testing k-means 1...'
  test(gmm.KMeans1())
  speedTest(gmm.KMeans1())
  print 'Testing k-means 2...'
  test(gmm.KMeans2())
  speedTest(gmm.KMeans2())
  print 'Testing k-means 3...'
  test(gmm.KMeans3())  
  speedTest(gmm.KMeans3())
  print '(Note that due to short feature lengths the runtimes do not mean much - the optimisations of 2 & 3 are based on the assumption that distance computations are expensive.)'
  print 'Testing isotropic GMM...'
  test(gmm.IsotropicGMM())


testAll()
