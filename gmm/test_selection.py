#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import random
import time

import numpy

import gmm



def test(model, sampCount = 1000):
  # Generate a random set of Gaussians to test with...
  count = random.randrange(2,5)
  mix = numpy.random.rand(count) + 1.0
  mix /= mix.sum()
  mean = numpy.random.randint(-1,2,(count,3))
  sd = 1.75*numpy.random.rand(count) + 0.25

  # Draw samples from them...
  samples = []
  for _ in xrange(sampCount):
    which = numpy.random.multinomial(1,mix).argmax()
    covar = sd[which]*numpy.identity(3)
    s = numpy.random.multivariate_normal(mean[which,:],covar)
    samples.append(s)

  # Model select...
  start = time.clock()
  gmm.modelSelectBIC(samples,model)
  end = time.clock()
  print 'time =',(end-start)

  # Print out the results...
  print 'Actual:'
  print 'count =',count
  for c in xrange(count):
    print 'mix = %f; mean = %s; sd = %f;'%(mix[c],str(mean[c,:]),sd[c])

  print 'Estimated:'
  print 'count =',model.clusterCount()
  for c in xrange(model.clusterCount()):
    if hasattr(model,'getMix'):
      print 'mix = %f; mean = %s; sd = %f;'%(model.getMix(c), str(model.getCentre(c)), model.getSD(c))
    else:
      print 'centre = %s;'%str(model.getCentre(c))
  print



def testAll():
  print 'Testing isotropic GMM...'
  test(gmm.IsotropicGMM())

  print 'Testing kmeans3...'
  test(gmm.KMeans3())
  
  print 'Testing kmeans2...'
  test(gmm.KMeans2())
  
  print 'Testing kmeans1...'
  test(gmm.KMeans1())
  
  print 'Testing kmeans0...'
  test(gmm.KMeans0())


testAll()
