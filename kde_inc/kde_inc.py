# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import math
import numpy

from loo_cov import PrecisionLOO, SubsetPrecisionLOO # Not used below, just for conveniance.
from gmm import GMM



class KDE_INC:
  """Provides an incrimental kernel density estimate system that uses Gaussians. A kernel density estimate system with Gaussian kernels that, on reaching a cap, starts merging kernels to limit the number of kernels to a constant - done in such a way as to minimise error whilst capping computation. (Computation is quite high however - this is not a very efficient implimentation.)"""
  def __init__(self, prec, cap = 32):
    """Initialise with the precision matrix to use for the kernels, which implicitly provides the number of dimensions, and the cap on the number of kernels to allow."""
    self.prec = numpy.asarray(prec, dtype=numpy.float32)
    self.gmm = GMM(prec.shape[0], cap) # Current mixture model.
    self.count = 0 # Number of samples provided so far.

    self.merge = numpy.ones((cap,cap), dtype=numpy.float32) # [i,j]; cost of merging two entrys, only valid when j<i, other values set high to avoid issues.
    self.merge *= 1e100

    # For holding the temporary merge costs calculated when adding a sample...
    self.mergeT = numpy.empty(cap, dtype=numpy.float32)

  def setPrec(self, prec):
    """Changes the precision matrix - must be called before any samples are added, and must have the same dimensions as the current one."""
    self.prec = numpy.asarray(prec, dtype=numpy.float32)


  def samples(self):
    """Returns how many samples have been added to the object."""
    return self.count

  def prob(self, sample):
    """Returns the probability of the given sample - must not be called until at least one sample has been added, though it will return a positive constant if called with no samples provided."""
    if self.count!=0: return self.gmm.prob(sample)
    else: return 1.0


  def __calcMergeCost(self, weightA, meanA, precA, weightB, meanB, precB):
    """Calculates and returns the cost of merging two Gaussians."""
    logDetA = math.log(numpy.linalg.det(precA))
    logDetB = math.log(numpy.linalg.det(precB))
    delta = meanA - meanB
    
    klAB = logDetA - logDetB
    klAB += numpy.trace(numpy.dot(precB,numpy.linalg.inv(precA)))
    klAB += numpy.dot(numpy.dot(delta,precB),delta)
    klAB -= precA.shape[0]
    klAB *= 0.5
    
    klBA = logDetB - logDetA
    klBA += numpy.trace(numpy.dot(precA,numpy.linalg.inv(precB)))
    klBA += numpy.dot(numpy.dot(delta,precA),delta)
    klBA -= precA.shape[0]
    klBA *= 0.5
    
    return weightA * klAB + weightB * klBA


  def add(self, sample):
    """Adds a sample, updating the kde accordingly."""
    if self.count<self.gmm.weight.shape[0]:
      # Pure kde phase...
      self.gmm.mean[self.count,:] = numpy.asarray(sample)
      self.gmm.prec[self.count,:,:] = self.prec
      self.gmm.calcNorm(self.count)
      
      self.count += 1
      self.gmm.weight[:self.count] = 1.0 / float(self.count)

      if self.count==self.gmm.weight.shape[0]:
        # Next sample starts merging - need to prepare by filling in the kl array...
        # (Below is grossly inefficient - calculates the same things more times than is possibly funny. I'll optimise it if I ever decide that I care enough to do so.)
        for i in xrange(self.merge.shape[0]):
          for j in xrange(i):
            self.merge[i,j] = self.__calcMergeCost(self.gmm.weight[i], self.gmm.mean[i,:], self.gmm.prec[i,:,:], self.gmm.weight[j], self.gmm.mean[j,:], self.gmm.prec[j,:,:])
    else:
      # Merging phase...
      sample = numpy.asarray(sample, dtype=numpy.float32)
      
      # Adjust weights...
      adjust = float(self.count) / float(self.count+1)
      self.gmm.weight *= adjust
      for i in xrange(self.merge.shape[0]): self.merge[i,:i] *= adjust
      
      weight = 1.0 / float(self.count+1)
      self.count += 1

      # Calculate the merging costs for the new kernel versus the old kernels...
      for i in xrange(self.merge.shape[0]):
        self.mergeT[i] = self.__calcMergeCost(weight, sample, self.prec, self.gmm.weight[i], self.gmm.mean[i,:], self.gmm.prec[i,:,:])

      # Select the best merge - it either involves the new sample or it does not...
      bestOld = numpy.unravel_index(numpy.argmin(self.merge), self.merge.shape)
      bestNew = numpy.argmin(self.mergeT)
      if self.mergeT[bestNew] < self.merge[bestOld]:
        # Easy scenario - new kernel is being merged with an existing kernel - not too much fiddling involved...

        # Do the merge...
        newWeight = weight + self.gmm.weight[bestNew]
        newMean = (weight/newWeight) * sample + (self.gmm.weight[bestNew]/newWeight) * self.gmm.mean[bestNew,:]

        delta1 = sample - newMean
        cov1 = numpy.linalg.inv(self.prec) + numpy.outer(delta1, delta1)

        delta2 = self.gmm.mean[bestNew,:] - newMean
        cov2 = numpy.linalg.inv(self.gmm.prec[bestNew,:,:]) + numpy.outer(delta2, delta2)
        
        newPrec = (weight/newWeight) * cov1 + (self.gmm.weight[bestNew]/newWeight) * cov2
        newPrec = numpy.linalg.inv(newPrec)

        # Store the result...
        self.gmm.weight[bestNew] = newWeight
        self.gmm.mean[bestNew,:] = newMean
        self.gmm.prec[bestNew,:,:] = newPrec
        self.gmm.calcNorm(bestNew)

        # Update the merge weights...
        for i in xrange(self.merge.shape[0]):
          if i!=bestNew:
            cost = self.__calcMergeCost(self.gmm.weight[i], self.gmm.mean[i,:], self.gmm.prec[i,:,:], self.gmm.weight[bestNew], self.gmm.mean[bestNew,:], self.gmm.prec[bestNew,:,:])
            if i<bestNew: self.merge[bestNew,i] = cost
            else: self.merge[i,bestNew] = cost

      else:
        # We are merging two old kernels, and then putting the new kernel into the slot freed up - this is extra fiddly...
         # Do the merge...
        newWeight = self.gmm.weight[bestOld[0]] + self.gmm.weight[bestOld[1]]
        newMean = (self.gmm.weight[bestOld[0]]/newWeight) * self.gmm.mean[bestOld[1],:] + (self.gmm.weight[bestOld[1]]/newWeight) * self.gmm.mean[bestOld[1],:]

        delta1 = self.gmm.mean[bestOld[0],:] - newMean
        cov1 = numpy.linalg.inv(self.gmm.prec[bestOld[0],:,:]) + numpy.outer(delta1, delta1)

        delta2 = self.gmm.mean[bestOld[1],:] - newMean
        cov2 = numpy.linalg.inv(self.gmm.prec[bestOld[1],:,:]) + numpy.outer(delta2, delta2)

        newPrec = (self.gmm.weight[bestOld[0]]/newWeight) * cov1 + (self.gmm.weight[bestOld[1]]/newWeight) * cov2
        newPrec = numpy.linalg.inv(newPrec)

        # Store the result, put the new component in the other slot...
        self.gmm.weight[bestOld[0]] = newWeight
        self.gmm.mean[bestOld[0],:] = newMean
        self.gmm.prec[bestOld[0],:,:] = newPrec
        self.gmm.calcNorm(bestOld[0])

        self.gmm.weight[bestOld[1]] = weight
        self.gmm.mean[bestOld[1],:] = sample
        self.gmm.prec[bestOld[1],:,:] = self.prec
        self.gmm.calcNorm(bestOld[1])

        # Update the merge weights for both the merged and new kernels...
        for i in xrange(self.merge.shape[0]):
          if i!=bestOld[0]:
            cost = self.__calcMergeCost(self.gmm.weight[i], self.gmm.mean[i,:], self.gmm.prec[i,:,:], self.gmm.weight[bestOld[0]], self.gmm.mean[bestOld[0],:], self.gmm.prec[bestOld[0],:,:])
            if i<bestOld[0]: self.merge[bestOld[0],i] = cost
            else: self.merge[i,bestOld[0]] = cost

        for i in xrange(self.merge.shape[0]):
          if i!=bestOld[0] and i!=bestOld[1]:
            cost = self.__calcMergeCost(self.gmm.weight[i], self.gmm.mean[i,:], self.gmm.prec[i,:,:], self.gmm.weight[bestOld[1]], self.gmm.mean[bestOld[1],:], self.gmm.prec[bestOld[1],:,:])
            if i<bestOld[1]: self.merge[bestOld[1],i] = cost
            else: self.merge[i,bestOld[1]] = cost
