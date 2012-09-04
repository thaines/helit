# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import math
import numpy



class PrecisionLOO:
  """Given a large number of samples this uses leave one out to calculate the optimal symetric precision matrix. Standard griding solution."""
  def __init__(self):
    """Initilises with no samples but a default grid of 10^-2 to 10 in 128 incriments that are linear in log base 10 space."""
    self.samples = []
    self.grid = []
    self.best = 1.0
    
    self.setLogGrid(-4.0, 1.0, 128)

  def setLogGrid(self, low=-4.0, high = 1.0, step = 128):
    """Sets the grid of variances to test to contain values going from 10^low to 10^high, with inclusive linear interpolation of the exponents to obtain step values."""
    self.grid = []
    for i in xrange(step):
      exponent = low + i*(high-low)/float(step-1)
      self.grid.append(math.pow(10.0,exponent))

  def addSample(self, sample):
    """Adds one or more samples to the set used for loo optimisation. Can either be a single vector or a data matrix, where the first dimension indexes the individual samples."""
    self.samples.append(numpy.asarray(sample, dtype=numpy.float32))

  def dataMatrix(self):
    """More for internal use - collates all the samples into a single data matrix, which is put in the internal samples array such that it does not break things - the data matrix is then returned."""
    if len(self.samples)==0: return None
    if len(self.samples)==1 and len(self.samples[0].shape)==2:
      return self.samples[0]

    def samSize(sample):
      if len(sample.shape)==2: return sample.shape[0]
      else: return 1
    count = sum(map(samSize,self.samples))
    dm = numpy.empty((count, self.samples[0].shape[-1]), dtype=numpy.float32)
    
    offset = 0
    for sample in self.samples:
      if len(sample.shape)==1:
        dm[offset,:] = sample
        offset += 1
      else:
        dm[offset:offset+sample.shape[0],:] = sample
        offset += sample.shape[0]

    self.samples = [dm]
    return dm


  def calcVar(self, var, subset = None):
    """Internal method really - given a variance calculates its leave one out nll. Has an optional subset parameter, which indexes a subset of data point to be used from the data matrix."""
    
    dm = self.dataMatrix()
    if subset!=None: dm = dm[subset,:]
    mask = numpy.empty(dm.shape[0], dtype=numpy.bool)
    logNorm = -0.5*dm.shape[1]*math.log(2.0*math.pi*var)

    nll = 0.0
    
    for loi in xrange(dm.shape[0]):
       mask[:] = True
       mask[loi] = False
       
       delta = numpy.reshape(dm[loi,:], (1,dm.shape[1])) - dm[mask,:]
       delta = numpy.square(delta).sum(axis=1)
       delta /= var
       delta *= -0.5
       delta += logNorm # Delta is now the log probability of the target sample in terms of the kernels emitted from all others.

       maxDelta = delta.max()
       logProb = maxDelta + math.log(numpy.exp(delta - maxDelta).sum())
       # logProb is now the log of the sum of the probabilities of the left-out sample from all other samples, basically the score for leaving this sample out.
       nll -= logProb

    return nll

  def solve(self, callback=None):
    """Trys all the options, and selects the one that provides the best nll."""
    self.best = None
    bestNLL = None
    for i, var in enumerate(self.grid):
      if callback!=None: callback(i,len(self.grid))
      nll = self.calcVar(var)
      if numpy.isfinite(nll) and (self.best==None or nll<bestNLL):
        self.best = var
        bestNLL = nll

  def getBest(self):
    """Returns the best precision matrix."""
    return numpy.identity(self.dataMatrix().shape[1], dtype=numpy.float32) / self.best



class SubsetPrecisionLOO(PrecisionLOO):
  """This class performs the same task as PrecisionLOO, except it runs on a subset of data points, and in effect tunes the precision matrix for a kernel density estimate constructed using less samples than are provided to the class. Takes the mean of multiple runs with different subsets."""

  def solve(self, runs, size, callback=None):
    """Trys all the options, and selects the one that provides the best nll. runs is the number of runs to do, with it taking the average score for each run, whilst size is how many samples to have in each run, i.e. the size to tune for."""
    # First generate all the subsets of the datamatrix...
    dm = self.dataMatrix()
    subset = []
    for _ in xrange(runs): subset.append(numpy.random.permutation(dm.shape[0])[:size])

    # Now loop and do the work...
    self.best = None
    bestNLL = None
    for i, var in enumerate(self.grid):
      if callback!=None: callback(i,len(self.grid))
      
      nll = 0.0
      for j, ss in enumerate(subset):
        if callback!=None: callback(i*runs + j, len(self.grid) * runs)
        nll += self.calcVar(var, ss)
      nll /= len(subset)

      if numpy.isfinite(nll) and (self.best==None or nll<bestNLL):
        self.best = var
        bestNLL = nll

    del callback
