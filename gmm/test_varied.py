#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import numpy

import gmm



def testTwoEven(obj,num=100):
  # Two well seperated clusters are tested...
  mean1 = numpy.array([-1.0,0.0])
  covar1 = numpy.array([[1.0,0.0],[0.0,1.0]])
  mean2 = numpy.array([1.0,0.0])
  covar2 = numpy.array([[0.5,0.0],[0.0,0.5]])
  
  feats1 = numpy.random.multivariate_normal(mean1,covar1,num)
  feats2 = numpy.random.multivariate_normal(mean2,covar2,num)
  feats = numpy.vstack([feats1,feats2])

  obj.train(feats,2)

  for i in xrange(2):
    print 'centre',i,':',obj.getCentre(i),'sd :',obj.getSD(i),'mix :',obj.getMix(i)


def testThreeUnbal(obj,num=100):
  # Two well seperated clusters are tested...
  mean1 = numpy.array([-1.0,0.0])
  covar1 = numpy.array([[1.0,0.0],[0.0,1.0]])
  mean2 = numpy.array([1.0,1.0])
  covar2 = numpy.array([[0.5,0.0],[0.0,0.5]])
  mean3 = numpy.array([1.0,-1.0])
  covar3 = numpy.array([[0.5,0.0],[0.0,0.5]])

  feats1 = numpy.random.multivariate_normal(mean1,covar1,2*num)
  feats2 = numpy.random.multivariate_normal(mean2,covar2,2*num)
  feats3 = numpy.random.multivariate_normal(mean3,covar3,num)
  feats = numpy.vstack([feats1,feats2,feats3])

  obj.train(feats,3)

  for i in xrange(3):
    print 'centre',i,':',obj.getCentre(i),'sd :',obj.getSD(i),'mix :',obj.getMix(i)



def test(obj):
  # Do a sequence of tests...
  print 'Two clusters, clean seperation... (-1,0),large sd and (1,0),low sd'
  testTwoEven(obj)

  print 'Two clusters, close, variable probabilities... (-1,0),large sd,high mix; (1,1),low sd,high mix; (1,-1),low sd,low mix'
  testThreeUnbal(obj)


def testAll():
  print 'Testing isotropic GMM...'
  test(gmm.IsotropicGMM())



testAll()
