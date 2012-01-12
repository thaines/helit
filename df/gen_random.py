# Copyright 2012 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import numpy
import numpy.random

from generators import Generator
from tests import *



class AxisRandomGen(Generator, AxisSplit):
  """Provides a generator for axis-aligned split planes that split the data set at random - uses a normal distribution constructed from the data. Has random selection of the dimension to split the axis on."""
  def __init__(self, channel, dimCount, splitCount, ignoreWeights=False):
    """channel is which channel to select the values from; dimCount is how many dimensions to try splits on; splitCount how many random split points to try for each selected dimension. Setting ignore weights to True means it will not consider the weights when calculating the normal distribution to draw random split points from."""
    AxisSplit.__init__(self, channel)
    self.dimCount = dimCount
    self.splitCount = splitCount
    self.ignoreWeights = ignoreWeights

  def clone(self):
    return AxisRandomGen(self.channel, self.dimCount, self.splitCount, self.ignoreWeights)
    
  def itertests(self, es, index, weights = None):
    for _ in xrange(self.dimCount):
      ind = numpy.random.randint(es.features(self.channel))
      values = es[self.channel, index, ind]
      
      if weights==None or self.ignoreWeights:
        mean = numpy.mean(values)
        std = max(numpy.std(values), 1e-6)
      else:
        w = weights[index]
        mean = numpy.average(values, weights=w)
        std = max(numpy.average(numpy.fabs(values-mean), weights=w), 1e-6)
      
      for _ in xrange(self.splitCount):
        split = numpy.random.normal(mean, std)
      
        yield numpy.asarray([ind], dtype=numpy.int32).tostring() + numpy.asarray([split], dtype=numpy.float32).tostring()



class LinearRandomGen(Generator, LinearSplit):
  """Provides a generator for split planes that it is entirly random. Randomly selects which dimensions to work with, the orientation of the split plane and then where to put the split plane, with this last bit done with a normal distribution."""
  def __init__(self, channel, dims, dimCount, dirCount, splitCount, ignoreWeights = False):
    """channel is which channel to select for and dims how many features (dimensions) to test on for any given test. dimCount is how many sets of dimensions to randomly select to generate tests from, whilst dirCount is how many random dimensions (From a uniform distribution over a hyper-sphere.) to use for selection. It actually generates the two independantly and trys every combination, as generating uniform random directions is somewhat expensive. For each of these splitCount split points are then tried, as drawn from a normal distribution. Setting ignore weights to True means it will not consider the weights when calculating the normal distribution to draw random split points from."""
    LinearSplit.__init__(self, channel, dims)
    self.dimCount = dimCount
    self.dirCount = dirCount
    self.splitCount = splitCount
    self.ignoreWeights = ignoreWeights
  
  def clone(self):
    return LinearRandomGen(self.channel, self.dims, self.dimCount, self.dirCount, self.splitCount, self.ignoreWeights)
    
  def itertests(self, es, index, weights = None):
    # Generate random points on the hyper-sphere...
    dirs = numpy.random.normal(size=(self.dirCount, self.dims))
    dirs /= numpy.sqrt(numpy.square(dirs).sum(axis=1)).reshape((-1,1))
    
    # Iterate and select a set of dimensions before trying each direction on them...
    for _ in xrange(self.dimCount):
      #dims = numpy.random.choice(es.features(self.channel), size=self.dims, replace=False) For when numpy 1.7.0 is common
      dims = numpy.zeros(self.dims, dtype=numpy.int32)
      feats = es.features(self.channel)
      for i in xrange(self.dims):
        dims[i] = numpy.random.randint(feats-i)
        dims[i] += (dims[:i]<=dims[i]).sum()
      
      for di in dirs:
        dists = (es[self.channel, index, dims] * di.reshape((1,-1))).sum(axis=1)
        
        if weights==None or self.ignoreWeights:
          mean = numpy.mean(dists)
          std = max(numpy.std(dists), 1e-6)
        else:
          w = weights[index]
          mean = numpy.average(dists, weights=w)
          std = max(numpy.average(numpy.fabs(dists-mean), weights=w), 1e-6)
        
        for _ in xrange(self.splitCount):
          split = numpy.random.normal(mean, std)
      
          yield numpy.asarray(dims, dtype=numpy.int32).tostring() + numpy.asarray(di, dtype=numpy.float32).tostring() + numpy.asarray([split], dtype=numpy.float32).tostring()



class DiscreteRandomGen(Generator, DiscreteBucket):
  """Defines a generator for discrete data. It basically takes a single discrete feature and randomly assigns just one value to pass and all others to fail the test. The selection is from the values provided by the data passed in, weighted by how many of them there are."""
  def __init__(self, channel, featCount, valueCount):
    """channel is the channel to build discrete tests for. featCount is how many different features to select to generate tests for whilst valueCount is how many values to draw and offer as tests for each feature selected."""
    DiscreteBucket.__init__(self, channel)
    self.featCount = featCount
    self.valueCount = valueCount
  
  def clone(self):
    return DiscreteRandomGen(self.channel, self.featCount, self.valueCount)
  
  def itertests(self, es, index, weights = None):
    # Iterate and yield the right number of tests...
    for _ in xrange(self.featCount):
      # Randomly select a feature...
      feat = numpy.random.randint(es.features(self.channel))
      values =  es[self.channel, index, feat]
      histo = numpy.bincount(values, weights=weights[index] if weights!=None else None)
      histo /= histo.sum()
      
      # Draw and iterate the values - do a fun trick to avoid duplicate yields,,,
      values = numpy.random.multinomial(self.valueCount, histo)
      for value in numpy.where(values!=0):      
        # Yield a discrete decision object...
        yield numpy.asarray(feat, dtype=numpy.int32).tostring() + numpy.asarray(value, dtype=numpy.int32).tostring()
