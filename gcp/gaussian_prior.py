# -*- coding: utf-8 -*-

# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import math
import numpy
import numpy.linalg
import numpy.random

from wishart import Wishart
from gaussian import Gaussian
from student_t import StudentT



class GaussianPrior:
  """The conjugate prior for the multivariate Gaussian distribution. Maintains the 4 values and supports various operations of interest - initialisation of prior, Bayesian update, drawing a Gaussian and calculating the probability of a data point comming from a Gaussian drawn from the distribution. Not a particularly efficient implimentation, and it has no numerical protection against extremelly large data sets. Interface is not entirly orthogonal, due to the demands of real world usage."""
  def __init__(self, dims):
    """Initialises with everything zeroed out, such that a prior must added before anything interesting is done. Supports cloning."""
    if isinstance(dims, GaussianPrior):
      self.invShape = dims.invShape.copy()
      self.shape = dims.shape.copy() if dims.shape!=None else None
      self.mu = dims.mu.copy()
      self.n = dims.n
      self.k = dims.k
    else:
      self.invShape = numpy.zeros((dims,dims), dtype=numpy.float32) # The inverse of lambda in the equations.
      self.shape = None # Cached value - inverse is considered primary.
      self.mu = numpy.zeros(dims, dtype=numpy.float32)
      self.n = 0.0
      self.k = 0.0

  def reset(self):
    """Resets as though there is no data, other than the dimensions of course."""
    self.invShape[:] = 0.0
    self.shape = None
    self.mu[:] = 0.0
    self.n = 0.0
    self.k = 0.0

  def addPrior(self, mean, covariance, weight = None):
    """Adds a prior to the structure, as an estimate of the mean and covariance matrix, with a weight which can be interpreted as how many samples that estimate is worth. Note the use of 'add' - you can call this after adding actual samples, or repeatedly. If weight is omitted it defaults to the number of dimensions, as the total weight in the system must match or excede this value before draws etc can be done."""
    if weight==None: weight = float(self.mu.shape[0])
    delta = mean - self.mu
    
    self.invShape += weight * covariance # *weight converts to a scatter matrix.
    self.invShape += ((self.k*weight)/(self.k+weight)) * numpy.outer(delta,delta)
    self.shape = None
    self.mu += (weight/(self.k+weight)) * delta
    self.n += weight
    self.k += weight

  def addSample(self, sample, weight=1.0):
    """Updates the prior given a single sample drawn from the Gaussian being estimated. Can have a weight provided, in which case it will be equivalent to repetition of that data point, where the repetition count can be fractional."""
    sample = numpy.asarray(sample, dtype=numpy.float32)
    if len(sample.shape)==0: sample.shape = (1,)
    delta = sample - self.mu

    self.invShape += (weight*self.k/(self.k+weight)) * numpy.outer(delta,delta)
    self.shape = None
    self.mu += delta * (weight / (self.k+weight))
    self.n += weight
    self.k += weight

  def remSample(self, sample):
    """Does the inverse of addSample, to in effect remove a previously added sample. Note that the issues of floating point (in-)accuracy mean its not perfect, and removing all samples is bad if there is no prior. Does not support weighting - effectvily removes a sample of weight 1."""
    sample = numpy.asarray(sample, dtype=numpy.float32)
    if len(sample.shape)==0: sample.shape = (1,)
    delta = sample - self.mu

    self.k -= 1.0
    self.n -= 1.0
    self.mu -= delta / self.k
    self.invShape -= ((self.k+1.0)/self.k) * numpy.outer(delta,delta)
    self.shape = None

  def addSamples(self, samples, weight = None):
    """Updates the prior given multiple samples drawn from the Gaussian being estimated. Expects a data matrix ([sample, position in sample]), or an object that numpy.asarray will interpret as such. Note that if you have only a few samples it might be faster to repeatedly call addSample, as this is designed to be efficient for hundreds+ of samples. You can optionally weight the samples, by providing an array to the weight parameter."""
    samples = numpy.asarray(samples, dtype=numpy.float32)

    # Calculate the mean and scatter matrices...
    if weight==None:
      # Unweighted samples...

      # Calculate the mean and scatter matrix...
      d = self.mu.shape[0]
      num = samples.shape[0]
    
      mean = numpy.average(samples, axis=0)
      
      scatter = numpy.tensordot(delta, delta, ([0],[0]))

    else:
      # Weighted samples...

      # Calculate the mean and scatter matrix...
      d = self.mu.shape[0]
      num = weight.sum()
      
      mean = numpy.average(samples, axis=0, weights=weight)
      
      delta = samples - mean.reshape((1,-1))
      scatter = numpy.tensordot(weight.reshape((-1,1))*delta, delta, ([0],[0]))

    # Update parameters...
    delta = mean-self.mu

    self.invShape += scatter
    self.invShape += ((self.k*num)/(self.k+num)) * numpy.outer(delta,delta)
    self.shape = None
    self.mu += (num/(self.k+num)) * delta
    self.n += num
    self.k += num

  def addGP(self, gp):
    """Adds another Gaussian prior, combining the two."""
    delta = gp.mu - self.mu
    
    self.invShape += gp.invShape
    self.invShape += ((gp.k*self.k)/(gp.k+self.k)) * numpy.outer(delta,delta)
    self.shape = None
    self.mu += (gp.k/(self.k+gp.k)) * delta
    self.n += gp.n
    self.k += gp.k


  def make_safe(self):
    """Checks for a singular inverse shape matrix - if singular replaces it with the identity. Also makes sure n and k are not less than the number of dimensions, clamping them if need be. obviously the result of this is quite arbitary, but its better than getting a crash from bad data."""
    dims = self.mu.shape[0]
    det = math.fabs(numpy.linalg.det(self.invShape))
    
    if det<1e-3:
      self.invShape = numpy.identity(dims, dtype=numpy.float32)
    if self.n<dims: self.n = dims
    if self.k<1e-3: self.k = 1e-3
    
  def reweight(self, newN = None, newK = None):
    """A slightly cheaky method that reweights the gp such that it has the new values of n and k, effectivly adjusting the relevant weightings of the samples - can be useful for generating a prior for some GPs using the data stored in those GPs. If a new k is not provided it is set to n; if a new n is not provided it is set to the number of dimensions."""
    if newN==None: newN = float(self.mu.shape[0])
    if newK==None: newK = newN

    self.invShape *= newN / self.n
    self.shape = None
    self.n = newN
    self.k = newK


  def getN(self):
    """Returns n."""
    return self.n

  def getK(self):
    """Returns k."""
    return self.k

  def getMu(self):
    """Returns mu."""
    return self.mu

  def getLambda(self):
    """Returns lambda."""
    if self.shape==None:
      self.shape = numpy.linalg.inv(self.invShape)
    return self.shape

  def getInverseLambda(self):
    """Returns the inverse of lambda."""
    return self.invShape


  def safe(self):
    """Returns true if it is possible to sample the prior, work out the probability of samples or work out the probability of samples being drawn from a collapsed sample - basically a test that there is enough information."""
    return self.n>=self.mu.shape[0] and self.k>0.0


  def prob(self, gauss):
    """Returns the probability of drawing the provided Gaussian from this prior."""
    d = self.mu.shape[0]
    wishart = Wishart(d)
    gaussian = Gaussian(d)
    
    wishart.setDof(self.n)
    wishart.setScale(self.getLambda())
    gaussian.setMean(self.mu)
    gaussian.setPrecision(self.k*gauss.getPrecision())

    return wishart.prob(gauss.getPrecision()) * gaussian.prob(gauss.getMean())

  def intProb(self):
    """Returns a multivariate student-t distribution object that gives the probability of drawing a sample from a Gaussian drawn from this prior, with the Gaussian integrated out. You may then call the prob method of this object on each sample obtained."""
    d = self.mu.shape[0]
    st = StudentT(d)

    dof = self.n-d+1.0
    st.setDOF(dof)
    st.setLoc(self.mu)
    mult = self.k*dof / (self.k+1.0)
    st.setInvScale(mult * self.getLambda())

    return st

  def sample(self):
    """Returns a Gaussian, drawn from this prior."""
    d = self.mu.shape[0]
    wishart = Wishart(d)
    gaussian = Gaussian(d)
    ret = Gaussian(d)

    wishart.setDof(self.n)
    wishart.setScale(self.getLambda())
    ret.setPrecision(wishart.sample())

    gaussian.setPrecision(self.k*ret.getPrecision())
    gaussian.setMean(self.mu)
    ret.setMean(gaussian.sample())

    return ret


  def __str__(self):
    return '{n:%f, k:%f, mu:%s, lambda:%s}'%(self.n, self.k, str(self.mu), str(self.getLambda()))
