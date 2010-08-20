# -*- coding: utf-8 -*-

# Copyright 2010 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import numpy

from params import *
from scipy.weave import inline



class Model:
  """Defines a model - this will consist of a parameters object to define the kernel (C is ignored, but will be the same as the trainning parameter if needed for reference.), a list of support vectors in a dataMatrix and then a vector of weights, plus the b parameter. The weights are the multiple of the y value and alpha value. Uses weave to make evaluation of new features fast."""
  def __init__(self, params, supportVectors, supportWeights, b):
    """Sets up a model given the parameters. Note that if given a linear kernel and multiple support vectors it does the obvious optimisation."""
    self.params = params
    self.supportVectors = supportVectors
    self.supportWeights = supportWeights
    self.b = b

    # Get the kernel code ready for the weave call...
    self.kernel = self.params.getCode()
    self.kernelKey = self.params.kernelKey()

    # Optimise the linear kernel if needed...
    if self.params.getKernel()==Kernel.linear and len(self.supportWeights)>1:
      self.supportVectors = (self.supportVectors.T * self.supportWeights).sum(axis=1)
      self.supportVectors = self.supportVectors.reshape((1, self.supportVectors.shape[0]))
      self.supportWeights = numpy.array((1.0,), dtype=numpy.double)


  def getParams(self):
    """Returns the parameters the svm was trainned with."""
    return self.params

  def getSupportVectors(self):
    """Returns a 2D array where each row is a support vector."""
    return self.supportVectors

  def getSupportWeights(self):
    """Returns the vector of weights matching the support vectors."""
    return self.supportWeights

  def getB(self):
    """Returns the addative offset of the function defined by the support vectors to locate the decision boundary at 0."""
    return self.b


  def decision(self,feature):
    """Given a feature vector this returns its decision boundary evaluation, specifically the weighted sum of each of the kernel evaluations for the support vectors against the given feature vector, plus b."""
    code = '// Kernel = '+self.kernelKey+'\n' + """
    double ret = b;
    for (int v=0;v<Nsw[0];v++)
    {
     ret += SW1(v) * kernel(Nsv[1],feature,&SV2(v,0));
    }
    return_val = ret;
    """

    sv = self.supportVectors
    sw = self.supportWeights
    b = self.b
    return inline(code,['feature','sv','sw','b'], support_code=self.kernel)

  def classify(self,feature):
    """Classifies a single feature vector - returns -1 or +1 depending on its class. Just the sign of the decision method."""
    if self.decision(feature)<0.0: return -1
    else: return 1


  def multiDecision(self,features):
    """Given a matrix where every row is a feature returns the decision boundary evaluation for each feature as an array of values."""
    code = '// Kernel = '+self.kernelKey+'\n' + """
    for (int f=0;f<Nfeatures[0];f++)
    {
     RET1(f) = b;
     for (int v=0;v<Nsw[0];v++)
     {
      RET1(f) += SW1(v) * kernel(Nsv[1],&FEATURES2(f,0),&SV2(v,0));
     }
    }
    """

    sv = self.supportVectors
    sw = self.supportWeights
    b = self.b
    ret = numpy.empty(features.shape[0], dtype=numpy.float_)
    inline(code, ['features','sv','sw','b','ret'], support_code=self.kernel)

    return ret

  def multiClassify(self,features):
    """Given a matrix where every row is a feature returns - returns -1 or +1 depending on the class of each vector, as an array. Just the sign of the multiDecision method. Be warned the classification vector is returned with a type of int8."""
    dec = self.multiDecision(features)
    ret = numpy.zeros(features.shape[0],dtype=numpy.int8)
    
    code = """
    for (int i=0;i<Ndec[0];i++)
    {
     if (dec[i]<0.0) ret[i] = -1;
                else ret[i] =  1;
    }
    """
    
    inline(code,['dec','ret'])
    
    return ret
