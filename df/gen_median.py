# Copyright 2012 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import numpy
import numpy.random

from generators import Generator
from tests import *



class AxisMedianGen(Generator, AxisSplit):
  """Provides a generator for axis-aligned split planes that split the data set in half, i.e. uses the median. Has random selection of the dimension to split the axis on."""
  def __init__(self, channel, count, ignoreWeights = False):
    """channel is which channel to select the values from, whilst count is how many tests it will return, where each has been constructed around a randomly selected feature from the channel. Setting ignore weights to True means it will not consider the weights when calculating the median."""
    AxisSplit.__init__(self, channel)
    self.count = count
    self.ignoreWeights = ignoreWeights

  def clone(self):
    return AxisMedianGen(self.channel, self.count, self.ignoreWeights)
    
  def itertests(self, es, index, weights = None):
    for _ in xrange(self.count):
      ind = numpy.random.randint(es.features(self.channel))
      values = es[self.channel, index, ind]
      
      if weights==None or self.ignoreWeights:
        median = numpy.median(values)
      else:
        cw = numpy.cumsum(weights[index])
        half = 0.5*cw[-1]
        pos = numpy.searchsorted(cw,half)
        t = (half - cw[pos-1])/max(cw[pos] - cw[pos-1], 1e-6)
        median = (1.0-t)*values[pos-1] + t*values[pos]
      
      yield numpy.asarray([ind], dtype=numpy.int32).tostring() + numpy.asarray([median], dtype=numpy.float32).tostring()



class LinearMedianGen(Generator, LinearSplit):
  """Provides a generator for split planes that uses the median of the features projected perpendicular to the plane direction, such that it splits the data set in half. Randomly selects which dimensions to work with and the orientation of the split plane."""
  def __init__(self, channel, dims, dimCount, dirCount, ignoreWeights = False):
    """channel is which channel to select for and dims how many features (dimensions) to test on for any given test. dimCount is how many sets of dimensions to randomly select to generate tests for, whilst dirCount is how many random dimensions (From a uniform distribution over a hyper-sphere.) to try. It actually generates the two independantly and trys every combination, as generating uniform random directions is somewhat expensive. Setting ignore weights to True means it will not consider the weights when calculating the median."""
    LinearSplit.__init__(self, channel, dims)
    self.dimCount = dimCount
    self.dirCount = dirCount
    self.ignoreWeights = ignoreWeights
  
  def clone(self):
    return LinearMedianGen(self.channel, self.dims, self.dimCount, self.dirCount, self.ignoreWeights)
    
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
          median = numpy.median(dists)
        else:
          cw = numpy.cumsum(weights[index])
          half = 0.5*cw[-1]
          pos = numpy.searchsorted(cw,half)
          t = (half - cw[pos-1])/max(cw[pos] - cw[pos-1], 1e-6)
          median = (1.0-t)*dists[pos-1] + t*dists[pos]
        
        yield numpy.asarray(dims, dtype=numpy.int32).tostring() + numpy.asarray(di, dtype=numpy.float32).tostring() + numpy.asarray([median], dtype=numpy.float32).tostring()
