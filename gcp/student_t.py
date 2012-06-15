# -*- coding: utf-8 -*-

# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import math
import numpy
import numpy.linalg
import scipy.special



class StudentT:
  """A feature incomplete multivariate student-t distribution object - at this time it only supports calculating the probability of a sample, and not the ability to make a draw."""
  def __init__(self, dims):
    """dims is the number of dimensions - initalises it to default values with the degrees of freedom set to 1, the location as the zero vector and the identity matrix for the scale. Suports copy construction."""
    if isinstance(dims, StudentT):
      self.dof = dims.dof
      self.loc = dims.loc.copy()
      self.scale = dims.scale.copy() if dims.scale!=None else None
      self.invScale = dims.invScale.copy() if dims.invScale!=None else None
      self.norm = dims.norm.copy() if dims.norm!=None else None
    else:
      self.dof = 1.0
      self.loc = numpy.zeros(dims, dtype=numpy.float32)
      self.scale = numpy.identity(dims, dtype=numpy.float32)
      self.invScale = None
      self.norm = None # Actually the log of the normalising constant.

  def setDOF(self, dof):
    """Sets the degrees of freedom."""
    self.dof = dof
    self.norm = None

  def setLoc(self, loc):
    """Sets the location vector."""
    l = numpy.array(loc, dtype=numpy.float32)
    assert(l.shape==self.loc.shape)
    self.loc = l

  def setScale(self, scale):
    """Sets the scale matrix."""
    s = numpy.array(scale, dtype=numpy.float32)
    assert(s.shape==(self.loc.shape[0],self.loc.shape[0]))
    self.scale = s
    self.invScale = None
    self.norm = None

  def setInvScale(self, invScale):
    """Sets the scale matrix by providing its inverse."""
    i = numpy.array(invScale, dtype=numpy.float32)
    assert(i.shape==(self.loc.shape[0],self.loc.shape[0]))
    self.scale = None
    self.invScale = i
    self.norm = None

  def getDOF(self):
    """Returns the degrees of freedom."""
    return self.dof

  def getLoc(self):
    """Returns the location vector."""
    return self.loc

  def getScale(self):
    """Returns the scale matrix."""
    if self.scale==None:
      self.scale = numpy.linalg.inv(self.invScale)
    return self.scale

  def getInvScale(self):
    """Returns the inverse of the scale matrix."""
    if self.invScale==None:
      self.invScale = numpy.linalg.inv(self.scale)
    return self.invScale


  def getLogNorm(self):
    """Returns the logarithm of the normalising constant of the distribution. Typically for internal use only."""
    if self.norm==None:
      d = self.loc.shape[0]
      self.norm = scipy.special.gammaln(0.5*(self.dof+d))
      self.norm -= scipy.special.gammaln(0.5*self.dof)
      self.norm -= math.log(self.dof*math.pi)*(0.5*d)
      self.norm += 0.5*math.log(numpy.linalg.det(self.getInvScale()))
    return self.norm

  def prob(self, x):
    """Given a vector x evaluates the density function at that point."""
    x = numpy.asarray(x)
    d = self.loc.shape[0]
    delta = x - self.loc
    
    val = numpy.dot(delta,numpy.dot(self.getInvScale(),delta))
    val = 1.0 + val/self.dof
    return math.exp(self.getLogNorm() + math.log(val)*(-0.5*(self.dof+d)))

  def logProb(self, x):
    """Returns the logarithm of prob - faster than a straight call to prob."""
    x = numpy.asarray(x)
    d = self.loc.shape[0]
    delta = x - self.loc

    val = numpy.dot(delta,numpy.dot(self.getInvScale(),delta))
    val = 1.0 + val/self.dof
    return self.getLogNorm() + math.log(val)*(-0.5*(self.dof+d))

  def batchProb(self, dm):
    """Given a data matrix evaluates the density function for each entry and returns the resulting array of probabilities."""
    d = self.loc.shape[0]
    delta = dm - self.loc.reshape((1,d))

    if hasattr(numpy, 'einsum'): # Can go away when scipy older than 1.6 is no longer in use.
      val = numpy.einsum('kj,ij,ik->i', self.getInvScale(), delta, delta)
    else:
      val = ((self.getInvScale().reshape(1,d,d) * delta.reshape(dm.shape[0],1,d)).sum(axis=2) * delta).sum(axis=1)
      
    val = 1.0 + val/self.dof
    return numpy.exp(self.getLogNorm() + numpy.log(val)*(-0.5*(self.dof+d)))

  def batchLogProb(self, dm):
    """Same as batchProb, but returns the logarithm of the probability instead."""
    d = self.loc.shape[0]
    delta = dm - self.loc.reshape((1,d))

    if hasattr(numpy, 'einsum'): # Can go away when scipy older than 1.6 is no longer in use.
      val = numpy.einsum('kj,ij,ik->i', self.getInvScale(), delta, delta)
    else:
      val = ((self.getInvScale().reshape(1,d,d) * delta.reshape(dm.shape[0],1,d)).sum(axis=2) * delta).sum(axis=1)
      
    val = 1.0 + val/self.dof
    return self.getLogNorm() + numpy.log(val)*(-0.5*(self.dof+d))


  def __str__(self):
    return '{dof:%f, location:%s, scale:%s}'%(self.getDOF(), str(self.getLoc()), str(self.getScale()))
