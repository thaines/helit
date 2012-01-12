# Copyright 2012 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import numpy
import numpy.random

from generators import Generator
from tests import *



class AxisClassifyGen(Generator, AxisSplit):
  """Provides a generator that creates axis-aligned split planes that have their position selected to maximise the information gain with respect to the task of classification."""
  def __init__(self, channel, catChannel, count):
    """channel is which channel to select the values from; catChannel contains the true classes of the features so the split can be optimised; and count is how many tests it will return, where each has been constructed around a randomly selected feature from the channel."""
    AxisSplit.__init__(self, channel)
    self.catChannel = catChannel
    self.count = count
    
  def clone(self):
    return AxisClassifyGen(self.channel, self.catChannel, self.count)
    
  def itertests(self, es, index, weights = None):
    def entropy(histo):
      histo = histo[histo>1e-6]
      return -(histo*(numpy.log(histo) - numpy.log(histo.sum()))).sum()
      
    for _ in xrange(self.count):
      ind = numpy.random.randint(es.features(self.channel))
      values = es[self.channel, index, ind]
      cats = es[self.catChannel, index, 0]
      
      if cats.shape[0]<2: split = 0.0
      else:
        indices = numpy.argsort(values)
        values = values[indices]
        cats = cats[indices]
      
        high = numpy.bincount(cats, weights=weights[index] if weights!=None else None)
        low = numpy.zeros(high.shape[0], dtype=numpy.float32)
      
        improvement = -1e100
        for i in xrange(values.shape[0]-1):
          # Move the selected item from high to low...
          w = weights[index[indices[i]]] if weights!=None else 1
          high[cats[i]] -= w
          low[cats[i]] += w
        
          # Calculate the improvement (Within a scalar factor constant for the entire field - only care about the relative value.)...
          imp = -(entropy(low) + entropy(high))
        
          # Keep if best...
          if imp>improvement:
            split = 0.5*(values[i] + values[i+1])
            improvement = imp
      
      yield numpy.asarray([ind], dtype=numpy.int32).tostring() + numpy.asarray([split], dtype=numpy.float32).tostring()



class LinearClassifyGen(Generator, LinearSplit):
  """Provides a generator for split planes that projected the features perpendicular to a random plane direction but then optimises where to put the split plane to maximise classification performance. Randomly selects which dimensions to work with and the orientation of the split plane."""
  def __init__(self, channel, catChannel, dims, dimCount, dirCount):
    """channel is which channel to select for and catChannel the channel to get the classification answers from. dims is how many features (dimensions) to test on for any given test. dimCount is how many sets of dimensions to randomly select to generate tests for, whilst dirCount is how many random dimensions (From a uniform distribution over a hyper-sphere.) to try. It actually generates the two independantly and trys every combination, as generating uniform random directions is somewhat expensive."""
    LinearSplit.__init__(self, channel, dims)
    self.catChannel = catChannel
    self.dimCount = dimCount
    self.dirCount = dirCount
  
  def clone(self):
    return LinearClassifyGen(self.channel, self.catChannel, self.dims, self.dimCount, self.dirCount)
    
  def itertests(self, es, index, weights = None):
    def entropy(histo):
      histo = histo[histo>1e-6]
      return -(histo*(numpy.log(histo) - numpy.log(histo.sum()))).sum()
      
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
        cats = es[self.catChannel, index, 0]
      
        if cats.shape[0]<2: split = 0.0
        else:
          indices = numpy.argsort(dists)
          dists = dists[indices]
          cats = cats[indices]
      
          high = numpy.bincount(cats, weights=weights[index] if weights!=None else None)
          low = numpy.zeros(high.shape[0], dtype=numpy.float32)
      
          improvement = -1e100
          for i in xrange(dists.shape[0]-1):
            # Move the selected item from high to low...
            w = weights[index[indices[i]]] if weights!=None else 1
            high[cats[i]] -= w
            low[cats[i]] += w
        
            # Calculate the improvement (Within a scalar factor constant for the entire field - only care about the relative value.)...
            imp = -(entropy(low) + entropy(high))
        
            # Keep if best...
            if imp>improvement:
              split = 0.5*(dists[i] + dists[i+1])
              improvement = imp
        
        yield numpy.asarray(dims, dtype=numpy.int32).tostring() + numpy.asarray(di, dtype=numpy.float32).tostring() + numpy.asarray([split], dtype=numpy.float32).tostring()



class DiscreteClassifyGen(Generator, DiscreteBucket):
  """Defines a generator for discrete data. It basically takes a single discrete feature and then greedily optimises to get the best classification performance, As it won't necesarilly converge to the global optimum multiple restarts are provided. The discrete values must form a contiguous set, starting at 0 and going upwards. When splitting it only uses values it can see - unseen values will fail the test, though it always arranges for the most informative half to be the one that passes the test."""
  def __init__(self, channel, catChannel, featCount, initCount):
    """channel is the channel to build discrete tests for; featCount is how many random features to randomly select and initCount how many random initialisations to try for each feature."""
    DiscreteBucket.__init__(self, channel)
    self.catChannel = catChannel
    self.featCount = featCount
    self.initCount = initCount
  
  def clone(self):
    return DiscreteClassifyGen(self.channel, self.catChannel, self.featCount, self.initCount)
    
  def itertests(self, es, index, weights = None):
    # Helper function used below...
    def entropy(histo):
      histo = histo[histo>1e-6]
      return -(histo*(numpy.log(histo) - numpy.log(histo.sum()))).sum()
    
    # Iterate and yield the right number of tests...
    for _ in xrange(self.featCount):
      # Randomly select a feature, get the values and categories...
      feat = numpy.random.randint(es.features(self.channel))
      values = es[self.channel, index, feat]
      cats = es[self.catChannel, index, 0]
      
      # Create histograms of category counts for each value...
      histos = dict()
      maxHistoSize = 0
      for value in numpy.unique(values):
        use = values==value
        histos[value] = numpy.bincount(cats[use], weights = weights[index[use]] if weights!=None else None)
        maxHistoSize = max(maxHistoSize, histos[value].shape[0])
      
      if len(histos)<2: # Can't optimise - give up.
        yield numpy.asarray(feat, dtype=numpy.int32).tostring()
        continue
      
      # Optimise from multiple starting points...
      for _ in xrange(self.initCount):
        # Generate a ranodm greedy order...
        order = numpy.random.permutation(histos.keys())
        
        # Initialise by putting the first two entrys in different halfs...
        low = numpy.zeros(maxHistoSize, dtype=numpy.float32)
        high = numpy.zeros(maxHistoSize, dtype=numpy.float32)
        
        low[:histos[order[0]].shape[0]] += histos[order[0]]
        lowEnt = entropy(low)
        keepLow = [order[0]]
        high[:histos[order[1]].shape[0]] += histos[order[1]]
        highEnt = entropy(high)
        keepHigh = [order[1]]
        
        # Loop the rest and put each of them in the best half...
        for i in xrange(2, order.shape[0]):
          # Get the histogram...
          histo = histos[order[i]]
          
          # Calculate the options...
          lowOp = low.copy()
          lowOp[:histo.shape[0]] += histo
          lowOpEnt = entropy(lowOp)
          
          highOp = high.copy()
          highOp[:histo.shape[0]] += histo
          highOpEnt = entropy(highOp)
          
          # Choose the best...
          if (lowEnt+highOpEnt)<(lowOpEnt+highEnt):
            high = highOp
            highEnt = highOpEnt
            keepHigh.append(order[i])
          else:
            low = lowOp
            lowEnt = lowOpEnt
            keepLow.append(order[i])
        
        # Swap the halfs so the half that passes has the lowest entropy - this is because unseen values will fail, so might as well send them to the least certain side...
        if lowEnt<highEnt: keepHigh = keepLow
      
        # Yield a discrete decision object...
        yield numpy.asarray(feat, dtype=numpy.int32).tostring() + numpy.asarray(keepHigh, dtype=numpy.int32).tostring()
