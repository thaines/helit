# Copyright 2013 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



# Compile the code if need be...
try:
  from utils.make import make_mod
  import os.path

  make_mod('ms_c', os.path.dirname(__file__), ['philox.h', 'philox.c', 'bessel.h', 'bessel.c', 'eigen.h', 'eigen.c', 'kernels.h', 'kernels.c', 'data_matrix.h', 'data_matrix.c', 'spatial.h', 'spatial.c', 'balls.h', 'balls.c', 'mean_shift.h', 'mean_shift.c', 'ms_c.h', 'ms_c.c'])
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
