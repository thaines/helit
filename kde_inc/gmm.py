# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import math
import numpy

from scipy import weave
from utils.start_cpp import start_cpp



log_2_pi = math.log(2.0*math.pi)



class GMM:
  """Contains a Gaussian mixture model - just a list of weights, means and precision matrices. List is of fixed size, and it has functions to determine the probability of a point in space. Components with a weight of zero are often computationally ignored. Initialises empty, which is not good for normalisation of weights - don't do it until data is avaliable! Designed to be used directly by any entity that is filling it in - interface is mostly user only."""
  def __init__(self, dims, count):
    """dims is the dimension of the mixture model, count the number of mixture components it will consider using."""
    self.weight = numpy.zeros(count, dtype=numpy.float32)
    self.mean = numpy.zeros((count, dims), dtype=numpy.float32)
    self.prec = numpy.zeros((count, dims, dims), dtype=numpy.float32) # Precision, i.e. inverse covariance.
    self.log_norm = numpy.zeros(count, dtype=numpy.float32) # Logarithm of the normalising multiplicative constant.
    
    self.temp = numpy.empty((2, dims), dtype=numpy.float32) # To save memory chugging in the inline code.
  
  def clone(self):
    """Returns a clone of this object."""
    ret = GMM(self.mean.shape[1], self.mean.shape[0])
    
    ret.weight[:] = self.weight
    ret.mean[:,:] = self.mean
    ret.prec[:,:,:] = self.prec
    ret.log_norm[:] = self.log_norm
    
    return ret


  def normWeights(self):
    """Scales the weights so they sum to one, as is required for correctness."""
    self.weight /= self.weight.sum()

  def calcNorm(self, i):
    """Sets the normalising constant for a specific entry."""
    self.log_norm[i] = 0.5 * math.log(numpy.linalg.det(self.prec[i,:,:]))
    self.log_norm[i] -= 0.5 * self.mean.shape[1] * log_2_pi
    
  def calcNorms(self):
    """Fills in the normalising constants for all components with weight."""
    nzi = numpy.nonzero(self.weight)[0]
    
    for ii in xrange(nzi.shape[0]):
      self.log_norm[nzi[ii]] = 0.5 * math.log(numpy.linalg.det(self.prec[nzi[ii],:,:]))

    self.log_norm[nzi] -= 0.5*self.mean.shape[1] * log_2_pi


  def prob(self, sample):
    """Given a sample vector, as something that numpy.asarray can interpret, return the normalised probability of the sample. All values must be correct for this to work. Has inline C, but if that isn't working the implimentation is fully vectorised, so should be quite fast despite being in python."""
    global weave
    
    try:
      if weave==None: raise Exception()
      
      code = start_cpp() + """
      float ret = 0.0;
      
      for (int i=0; i<Nweight[0]; i++)
      {
       if (weight[i]>1e-6)
       {
        // Calculate the delta...
         for (int j=0; j<Nmean[1]; j++)
         {
          TEMP2(0, j) = SAMPLE1(j) - MEAN2(i, j);
          TEMP2(1, j) = 0.0;
         }
         
        // Multiply the precision with the delta and put it into TEMP2(1, ...)...
         for (int j=0; j<Nmean[1]; j++)
         {
          for (int k=0; k<Nmean[1]; k++)
          {
           TEMP2(1, j) += PREC3(i, j, k) * TEMP2(0, k);
          }
         }
         
        // Dot product TEMP2(0, ...) and TEMP2(1, ...) to get the core of the distribution...
         float core = 0.0;
         for (int j=0; j<Nmean[1]; j++)
         {
          core += TEMP2(0, j) * TEMP2(1, j);
         }
         
        // Factor in the rest, add it to the return...
         float val = weight[i] *  exp(-0.5 * core + log_norm[i]);
         if (std::isfinite(val)) ret += val;
       }
      }
      
      return_val = ret;
      """
      
      sample = numpy.asarray(sample, dtype=numpy.float32)
      weight = self.weight
      mean = self.mean
      prec = self.prec
      log_norm = self.log_norm
      temp = self.temp
      
      return weave.inline(code, ['sample', 'weight', 'mean', 'prec', 'log_norm', 'temp'])
    except Exception, e:
      if weave!=None:
        print e
        weave = None
      
      nzi = numpy.nonzero(self.weight)[0]
    
      sample = numpy.asarray(sample)
      delta = numpy.reshape(sample, (1,self.mean.shape[1])) - self.mean[nzi,:]

      nds = (nzi.shape[0], delta.shape[1], 1)
      core = (numpy.reshape(delta, nds) * self.prec[nzi,:,:]).sum(axis=1)
      core = (core * delta).sum(axis=1)
      core *= -0.5

      core += self.log_norm[nzi]
      core = numpy.exp(core)
      core *= self.weight[nzi]
      return core[numpy.isfinite(core)].sum() # Little bit of safety.


  def nll(self, sample):
    """Given a sample vector, as something that numpy.asarray can interpret, return the negative log liklihood of the sample. All values must be correct for this to work. Has inline C, but if that isn't working the implimentation is fully vectorised, so should be quite fast despite being in python."""
    global weave
    
    try:
      if weave==None: raise Exception()
      
      code = start_cpp() + """
      float ret = -1e64;
      
      for (int i=0; i<Nweight[0]; i++)
      {
       if (weight[i]>1e-6)
       {
        // Calculate the delta...
         for (int j=0; j<Nmean[1]; j++)
         {
          TEMP2(0, j) = SAMPLE1(j) - MEAN2(i, j);
          TEMP2(1, j) = 0.0;
         }
         
        // Multiply the precision with the delta and put it into TEMP2(1, ...)...
         for (int j=0; j<Nmean[1]; j++)
         {
          for (int k=0; k<Nmean[1]; k++)
          {
           TEMP2(1, j) += PREC3(i, j, k) * TEMP2(0, k);
          }
         }
         
        // Dot product TEMP2(0, ...) and TEMP2(1, ...) to get the core of the distribution...
         float core = 0.0;
         for (int j=0; j<Nmean[1]; j++)
         {
          core += TEMP2(0, j) * TEMP2(1, j);
         }
         
        // Factor in the rest, add it to the return...
         float val = log(weight[i]) + log_norm[i] - 0.5 * core;
         if (std::isfinite(val))
         {
          if (ret>val)
          {
           ret = ret + log(1.0 + exp(val - ret));
          }
          else
          {
           ret = val + log(1.0 + exp(ret - val));
          }
         }
       }
      }
      
      return_val = -ret;
      """
      
      sample = numpy.asarray(sample, dtype=numpy.float32)
      weight = self.weight
      mean = self.mean
      prec = self.prec
      log_norm = self.log_norm
      temp = self.temp
      
      return weave.inline(code, ['sample', 'weight', 'mean', 'prec', 'log_norm', 'temp'])
    except Exception, e:
      if weave!=None:
        print e
        weave = None
      
      nzi = numpy.nonzero(self.weight)[0]
    
      sample = numpy.asarray(sample)
      delta = numpy.reshape(sample, (1,self.mean.shape[1])) - self.mean[nzi,:]

      nds = (nzi.shape[0], delta.shape[1], 1)
      core = (numpy.reshape(delta, nds) * self.prec[nzi,:,:]).sum(axis=1)
      core = (core * delta).sum(axis=1)
      core *= -0.5

      core += self.log_norm[nzi]
      core += numpy.log(self.weight[nzi])
      
      high = core.max()
      ret = high + numpy.log(numpy.exp(core-high).sum())
      return -ret


  def marginalise(self, dims):
    """Given a list of dimensions this keeps those dimensions and drops the rest, i.e. marginalises them out. New version of this object will have the old indices remapped as indicated by dims."""
    dims = numpy.asarray(dims)
    self.mean = self.mean[:, dims]
    
    for i in xrange(self.prec.shape[0]):
      if self.weight[i]>1e-6:
        self.prec[i,:,:] = numpy.linalg.inv(self.prec[i,:,:])
    self.prec = self.prec[numpy.ix_(numpy.arange(self.prec.shape[0]), dims, dims)]
    for i in xrange(self.prec.shape[0]):
      if self.weight[i]>1e-6:
        self.prec[i,:,:] = numpy.linalg.inv(self.prec[i,:,:])
    
    self.calcNorms()
