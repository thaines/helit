# -*- coding: utf-8 -*-

# Copyright 2010 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



from smo import SMO

import copy
import numpy



def looPair(params,data):
  """Given a parameters object and a pair of data matrix and y (As returned by dataset.getTrainData.) this returns a (good) approximation of the leave one out negative log likellihood, and a model trained on *all* the data as a pair. Makes the assumption that losing a non-supporting vector does not require retraining, which is correct the vast majority of the time, and as a bonus avoids retrainning for most of the data, making this relativly fast."""
  dataMatrix,y = data

  # First train on all the data...
  smo = SMO()
  smo.setParams(params)
  smo.setData(dataMatrix,y)
  smo.solve()
  onAll = copy.deepcopy(smo.getModel())
  indices = smo.getIndices()

  # Collate statistics for all the non-supporting vectors...
  scores = onAll.multiClassify(dataMatrix)*y
  correct = (scores>0).sum() - (scores[indices]>0).sum()

  # Now iterate and retrain without each of the supporting vectors, collating the statistics...
  for i in xrange(indices.shape[0]):
    index = indices[i]
    noIndex = numpy.array(range(index)+range(index+1,y.shape[0]))
    smo.setData(dataMatrix[noIndex],y[noIndex])
    smo.solve()
    res = smo.getModel().classify(dataMatrix[index]) * y[index]
    if res>0: correct += 1

  # Return the loo and initial trainning on all the data...
  return (float(correct)/float(y.shape[0]),onAll)



def looPairRange(params, data, dist = 1.1):
  """Identical to looPair, except you specifiy a distance from the boundary and it retrains for all points in that range, but not for once outside that range. For a value of one, ignoring rounding error, it should be identical to looPair, though in practise you should never do this - dist should always be >1.0. This also has a better than optimisation - if it knows its result is going to be worse than betterThan it gives up and saves computation."""
  dataMatrix,y = data

  # First train on all the data...
  smo = SMO()
  smo.setParams(params)
  smo.setData(dataMatrix,y)
  smo.solve()
  onAll = copy.deepcopy(smo.getModel())

  # Get set of indices to retrain with, collate statistics for all the non-supporting vectors...
  scores = onAll.multiClassify(dataMatrix)*y
  indices = numpy.nonzero(scores<dist)[0]
  correct = (scores>0).sum() - (scores[indices]>0).sum()

  # Now iterate and retrain without each of the supporting vectors, collating the statistics...
  for i in xrange(indices.shape[0]):
    index = indices[i]
    noIndex = numpy.array(range(index)+range(index+1,y.shape[0]))
    smo.setData(dataMatrix[noIndex],y[noIndex])
    smo.solve()
    res = smo.getModel().classify(dataMatrix[index]) * y[index]
    if res>0: correct += 1

  # Return the loo and initial trainning on all the data...
  return (float(correct)/float(y.shape[0]),onAll)



def looPairBrute(params,data):
  """Same as looPair but does it brute force style - no approximation here."""
  dataMatrix,y = data

  # First train on all the data...
  smo = SMO()
  smo.setParams(params)
  smo.setData(dataMatrix,y)
  smo.solve()
  onAll = copy.deepcopy(smo.getModel())

  # Now iterate and retrain without each of the vectors, collating the statistics...
  correct = 0
  for i in xrange(y.shape[0]):
    noIndex = numpy.array(range(i)+range(i+1,y.shape[0]))
    smo.setData(dataMatrix[noIndex],y[noIndex])
    smo.solve()
    res = smo.getModel().classify(dataMatrix[i]) * y[i]
    if res>0: correct += 1

  # Return the loo and initial trainning on all the data...
  return (float(correct)/float(y.shape[0]),onAll)



def looPairSelect(paramsList,data):
  """Given an iterator of parameters this returns a pair of the loo score and model of the best set of parameters - just loops over looPair."""
  best = None
  for params in paramsList:
    res = looPair(params,data)
    if best==None or res[0]>best[0]:
      best = res
  return best
