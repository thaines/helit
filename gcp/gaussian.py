# -*- coding: utf-8 -*-

# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import math
import numpy
import numpy.linalg
import numpy.random



class Gaussian:
  """A basic multivariate Gaussian class. Has caching to avoid duplicate calculation."""
  def __init__(self, dims):
    """dims is the number of dimensions. Initialises with mu at the origin and the identity matrix for the precision/covariance. dims can also be another Gaussian object, in which case it acts as a copy constructor."""
    if isinstance(dims, Gaussian):
      self.mean = dims.mean.copy()
      self.precision = dims.precision.copy() if dims.precision!=None else None
      self.covariance = dims.covariance.copy() if dims.covariance!=None else None
      self.norm = dims.norm
      self.cholesky = dims.cholesky.copy() if dims.cholesky!=None else None
    else:
      self.mean = numpy.zeros(dims, dtype=numpy.float32)
      self.precision = numpy.identity(dims, dtype=numpy.float32)
      self.covariance = None
      self.norm = None
      self.cholesky = None

  def setMean(self, mean):
    """Sets the mean - you can use anything numpy will interprete as a 1D array of the correct length."""
    nm = numpy.array(mean, dtype=numpy.float32)
    assert(nm.shape==self.mean.shape)
    self.mean = nm

  def setPrecision(self, precision):
    """Sets the precision matrix. Alternativly you can use the setCovariance method."""
    np = numpy.array(precision, dtype=numpy.float32)
    assert(np.shape==(self.mean.shape[0],self.mean.shape[0]))
    self.precision = np
    self.covariance = None
    self.norm = None
    self.cholesky = None

  def setCovariance(self, covariance):
    """Sets the covariance matrix. Alternativly you can use the setPrecision method."""
    nc = numpy.array(covariance, dtype=numpy.float32)
    assert(nc.shape==(self.mean.shape[0],self.mean.shape[0]))
    self.covariance = nc
    self.precision = None
    self.norm = None
    self.cholesky = None

  def getMean(self):
    """Returns the mean."""
    return self.mean

  def getPrecision(self):
    """Returns the precision matrix."""
    if self.precision==None:
      self.precision = numpy.linalg.inv(self.covariance)
    return self.precision

  def getCovariance(self):
    """Returns the covariance matrix."""
    if self.covariance==None:
      self.covariance = numpy.linalg.inv(self.precision)
    return self.covariance


  def getNorm(self):
    """Returns the normalising constant of the distribution. Typically for internal use only."""
    if self.norm==None:
      self.norm = math.pow(2.0*math.pi,-0.5*self.mean.shape[0]) * math.sqrt(numpy.linalg.det(self.getPrecision()))
    return self.norm

  def prob(self, x):
    """Given a vector x evaluates the probability density function at that point."""
    x = numpy.asarray(x)
    offset = x - self.mean
    val = numpy.dot(offset,numpy.dot(self.getPrecision(),offset))
    return self.getNorm() * math.exp(-0.5 * val)


  def sample(self):
    """Draws and returns a sample from the distribution."""
    if self.cholesky==None:
      self.cholesky = numpy.linalg.cholesky(self.getCovariance())
    z = numpy.random.normal(size=self.mean.shape)
    return self.mean + numpy.dot(self.cholesky,z)


  def __str__(self):
    return '{mean:%s, covar:%s}'%(str(self.mean), str(self.getCovariance()))
