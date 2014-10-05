# Copyright 2013 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



# Compile the code if need be...
try:
  from utils.make import make_mod
  import os.path

  make_mod('ms_c', os.path.dirname(__file__), ['philox.h', 'philox.c', 'bessel.h', 'bessel.c', 'eigen.h', 'eigen.c', 'mult.h', 'mult.c', 'kernels.h', 'kernels.c', 'convert.h', 'convert.c', 'data_matrix.h', 'data_matrix.c', 'spatial.h', 'spatial.c', 'balls.h', 'balls.c', 'mean_shift.h', 'mean_shift.c', 'ms_c.h', 'ms_c.c'], numpy=True)
except: pass



# Import the compiled module into this space, so we can pretend they are one and the same, just with automatic compilation...
from ms_c import MeanShift as MeanShiftC

import numpy



# Augment the class coded in C with some pure python functionality...
class MeanShift(MeanShiftC):
  __doc__ = MeanShiftC.__doc__
  
  def scale_loo_nll(self, low = 0.01, high = 2.0, steps = 64, callback = None):
    """Does a sweep of the scale, from low to high, on a logarithmic scale with the given number of steps. Sets the scale to the one with the lowest loo_nll score. If low/high are provided as multipliers then these are multipliers of the silverman scale; otherwise they can by arbitrary vectors."""
    
    # Select values for low and high as needed...
    if isinstance(low, float) or isinstance(high, float):
      _, silverman = self.stats()
      silverman[silverman<1e-6] = 1e-6
      silverman = 1.0 / (silverman * (self.weight() * (silverman.shape[0] + 2.0) / 4.0) ** (-1.0 / (silverman.shape[0] + 4.0)))
      
      if isinstance(low, float):
        low = silverman * low
      
      if isinstance(high, float):
        high = silverman * high

    # Iterate, recording the scale with the best score thus far...
    if steps<2: steps = 2
    
    log_low = numpy.log(low)
    log_step = (numpy.log(high) - log_low) / (steps-1)
    
    best_score = None
    best_scale = None
    
    for i in xrange(steps):
      if callback!=None:
        callback(i, steps)

      scale = numpy.exp(log_low + i*log_step)
      
      self.set_scale(scale)
      score = self.loo_nll()
      
      if best_score==None or score < best_score:
        best_score = score
        best_scale = scale
    
    # Set it to the best...
    self.set_scale(best_scale)
    return best_score
  
  
  def scale_loo_nll_array(self, choices, callback = None):
    """Given an array of MS objects this copies in the configuration of each object in turn into this object and finds the one that minimises the leave one out error. Quite simple really - mostly for use in cases when the kernel type doesn't support scale in the usual way, i.e. the directional kernels. For copying across it uses a call to copy_all and a call to copy_scale, which between them get near as everything. Note that the array of choices will need some dummy data set, so the system knows the number of dimensions."""
    best_choice = None
    best_score = None
    
    for i, choice in enumerate(choices):
      if callback!=None:
        callback(i, len(choices))
      
      self.copy_all(choice)
      self.copy_scale(choice)
      
      score = self.loo_nll()
      
      if best_score==None or score < best_score:
        best_choice = choice
        best_score = score
    
    self.copy_all(best_choice)
    self.copy_scale(best_choice)
    return best_score


  def hierarchy(self, low = 1.0, high = 512.0, steps = 64, callback = None):
    """Does a sweep of scale, exactly like scale_loo_nll (same behaviour for low and high with vector vs single value), except it clusters the data at each level and builds a hierarchy of clusters, noting which cluster at a lower level ends up in which cluster at the next level. Note that low and high are inverted before use so that they equate to the typical mean shift parameters. Return is a list indexed by level, with index 0 representing the original data (where every data point is its own segment). Each level is represented by a tuple: (modes - array of [segment, feature], parents - array of [segment], giving the index of its parent segment in the next level. None in the highest level, sizes - array of [segment] giving the total weight of all exemplars in that segment.)"""
    
    # Select values for low and high as needed...
    if isinstance(low, float) or isinstance(high, float):
      _, silverman = self.stats()
      silverman[silverman<1e-6] = 1e-6
      silverman = 1.0 / (silverman * (self.weight() * (silverman.shape[0] + 2.0) / 4.0) ** (-1.0 / (silverman.shape[0] + 4.0)))
      
      if isinstance(low, float):
        low = silverman * low
      
      if isinstance(high, float):
        high = silverman * high
        
    # Stuff needed for the below...
    if steps<2: steps = 2
    
    log_low = numpy.log(low)
    log_step = (numpy.log(high) - log_low) / (steps-1)
    
    ret = [[self.fetch_dm(), None, self.fetch_weight()]] # Start with level 0 only.
    
    # Iterate, recording the clustering at each scale...
    for i in xrange(steps):
      if callback!=None:
        callback(i, steps)

      scale = 1.0 / numpy.exp(log_low + i*log_step)
      self.set_scale(scale)
      
      # Different behaviour for first level...
      if i==0:
        clusters, parents = self.cluster()
        parents = parents.flatten()
        safe = parents>=0
        sizes = numpy.bincount(parents[safe], weights=self.fetch_weight()[safe], minlength=clusters.shape[0])
      else:
        #clusters, _ = self.cluster()
        #parents = self.assign_clusters(ret[-1][0])
        clusters, parents = self.cluster_on(ret[-1][0])
        safe = parents>=0
        sizes = numpy.bincount(parents[safe], weights=ret[-1][2][safe], minlength=clusters.shape[0])
      
      ret[-1][1] = parents
      ret.append([clusters, None, sizes])
    
    return ret
