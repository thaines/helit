# -*- coding: utf-8 -*-

# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import numpy
import numpy.linalg

from gaussian import Gaussian



class GaussianInc:
  """Allows you to incrimentally calculate a Gaussian distribution by providing lots of samples."""
  def __init__(self, dims):
    """You provide the number of dimensions - you must add at least dims samples before there is the possibility of extracting a gaussian from this. Can also act as a copy constructor."""
    if isinstance(dims, GaussianInc):
      self.n = dims.n
      self.mean = dims.mean.copy()
      self.scatter = dims.scatter.copy()
    else:
      self.n = 0
      self.mean = numpy.zeros(dims, dtype=numpy.float32)
      self.scatter = numpy.zeros((dims,dims), dtype=numpy.float32)

  def add(self, sample, weight=1.0):
    """Updates the state given a new sample - sample can have a weight, which obviously defaults to 1, but can be set to other values to indicate repetition of a single point, including fractional."""
    sample = numpy.asarray(sample)
    
    # Sample count goes up...
    self.n += weight

    # Update mean vector...
    delta = sample - self.mean
    self.mean += delta*(weight/float(self.n))

    # Update scatter matrix (Yes, there is duplicated calculation here as it is symmetric, but who cares?)...
    self.scatter += weight * numpy.outer(delta, sample - self.mean)
    

  def safe(self):
    """Returns True if it has enough data to provide an actual Gaussian, False if it does not."""
    return math.fabs(numpy.linalg.det(self.scatter)) > 1e-6

  def makeSafe(self):
    """Bodges the internal representation so it can provide a non-singular covariance matrix - obviously a total hack, but potentially useful when insufficient information exists. Works by taking the svd, nudging zero entrys away from 0 in the diagonal matrix, then multiplying the terms back together again. End result is arbitary, but won't be inconsistant with the data provided."""
    u, s, v = numpy.linalg.svd(self.scatter)
    
    epsilon = 1e-5
    for i in xrange(s.shape[0]):
      if math.fabs(s[i])<epsilon:
        s[i] = math.copysign(epsilon, s[i])
    
    self.scatter[:,:] = numpy.dot(u, numpy.dot(numpy.diag(s), v))
    

  def fetch(self):
    """Returns the Gaussian distribution calculated so far."""
    ret = Gaussian(self.mean.shape[0])
    ret.setMean(self.mean)
    ret.setCovariance(self.scatter/float(self.n))
    return ret
