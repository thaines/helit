# -*- coding: utf-8 -*-

# Copyright 2010 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



from params import *
from smo_aux import *
from model import *

import numpy
from scipy.weave import inline



class SMO:
  """Implimentation of the 'Sequential Minimal Optimization' SVM solving method of Platt, using the WSS 3 pair selection method of Fan, Chen and Lin. This just solves for the alpha values - it is upto the wrapping code to do something useful with them. Makes extensive use of scipy.weave, so you need that working."""
  def __init__(self):
    """Initalises the parameters to a suitable default, but does not fill in a dataset - at the very least you will have to provide that."""
    self.params = Params()
    self.dataMatrix = None
    self.y = None
    self.model = None
    self.alpha = None


  def setParams(self, params):
    """Sets the parameters, i.e. which model to fit."""
    self.params = params

  def getParams(self):
    """Returns the parameters object - by default this is a linear model with a C of 10.0"""
    return self.params

  def setData(self, dataMatrix, y=None):
    """Sets the data matrix and corresponding y vector of +/- 1 values. If given only one value this function assumes its a tuple of (dataMatrix,y), as returned by the Dataset getTrainData method."""
    if y==None:
      self.dataMatrix = dataMatrix[0]
      self.y = dataMatrix[1]
    else:
      self.dataMatrix = dataMatrix
      self.y = y
    assert self.y.shape[0]==self.dataMatrix.shape[0] , 'dataMatrix and feature vector lengths do not match.'

  def getDataMatrix(self):
    """Returns the current data matrix, where each row is a feature vector."""
    return self.dataMatrix

  def getY(self):
    """Returns the y vector, that is the labels for the feature vector."""
    return self.y


  def solve(self, alpha = None):
    """Solves for the current information and replaces the current model, if any. You can optionally provide an alpha vector of alpha values for each vector - this can speed up convergance if initialised better than the typical zeroed vector."""

    # Check for having no samples of one type - handle elegantly...
    dm = self.dataMatrix
    y = self.y
    
    pCount = numpy.nonzero(y>0)[0].shape[0]
    nCount = y.shape[0] - pCount
    
    if pCount==0 or nCount==0:
      self.alpha = numpy.zeros(0,dtype=numpy.float_)
      if pCount>0: b = 1.0
      else: b = -1.0
      self.model = Model(self.params, numpy.zeros((0,self.dataMatrix.shape[1]),dtype=numpy.float_), numpy.zeros(0,dtype=numpy.float_), b)
      return

    # First do the heavy weight task - calculate the alpha weights for the vectors...
    support  = self.params.getCode()
    support += cacheCode

    kernelKey = '// Kernel = '+self.params.kernelKey()+'\n'

    if alpha==None: alpha = numpy.zeros(self.y.shape[0], dtype=numpy.double)
    else: alpha = alpha.copy()
    gradient = numpy.ones(self.y.shape[0],dtype=numpy.double)
    gradient *= -1.0
    if self.params.getRebalance():
      r = self.params.getC() * y.shape[0]/(2.0*pCount*nCount)
      cp = r * nCount
      cn = r * pCount
    else:
      cp = self.params.getC()
      cn = self.params.getC()
    
    inline(kernelKey+smoCoreCode, ['dm','y','alpha','gradient','cp','cn'], support_code = support)

    # Now build the model so far, but set b to zero...
    self.alpha = alpha
    indices = numpy.nonzero(alpha>=1e-3)[0]
    self.model = Model(self.params, self.dataMatrix[indices], numpy.asfarray(self.y[indices]) * alpha[indices], 0.0)

    # Finally, calculate the b offset value and stuff it into the model - its easier this way as we can use the model to calculate the offsets...
    # (Note the below code 'handles' the scenario where all vectors are at 0 or c - this shouldn't happen, but better safe than screwed by numerical error.)
    minB = -1e100
    maxB = 1e100
    actualB = 0.0
    numActualB = 0

    dec = self.model.multiDecision(self.dataMatrix)
    
    for i in xrange(y.shape[0]):
      if self.y[i]<0: cap = cn
      else: cap = cp
      if alpha[i]<1e-3:
        if y[i]<0:
          maxB = min((maxB,self.y[i] - dec[i]))
        else:
          minB = max((minB,self.y[i] - dec[i]))
      elif alpha[i]>(cap-1e-3):
        if y[i]<0:
          minB = max((minB,self.y[i] - dec[i]))
        else:
          maxB = min((maxB,self.y[i] - dec[i]))
      else:
        numActualB += 1
        actualB += (self.y[i] - dec[i] - actualB) / float(numActualB)
    if numActualB>0:
      self.model.b = actualB
    else:
      self.model.b = 0.5*(minB + maxB)


  def getModel(self):
    """Returns the model from the last call to solve, or None if solve has never been called."""
    return self.model

  def getAlpha(self):
    """Returns the alpha vector that the algorithm converged to - can be useful for initialising a new but similar model."""
    return self.alpha

  def getIndices(self):
    """Returns an array of the indices of the vectors from the input dataset that form the support vectors of the current model, or None if solve has never been called."""
    return numpy.nonzero(self.alpha>=1e-3)[0]
