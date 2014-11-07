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

import re
import numpy
from collections import defaultdict



# Augment the class coded in C with some pure python functionality...
class MeanShift(MeanShiftC):
  __doc__ = MeanShiftC.__doc__
  
  def scale_loo_nll(self, low = 0.01, high = 2.0, steps = 64, callback = None):
    """Does a sweep of the scale, from low to high, on a logarithmic scale with the given number of steps. Sets the scale to the one with the lowest loo_nll score. If low/high are provided as multipliers then these are multipliers of the silverman scale; otherwise they can by arbitrary vectors."""
    
    # Select values for low and high as needed...
    if isinstance(low, float) or isinstance(high, float):
      _, silverman = self.stats()
      silverman = silverman * (self.weight() * (silverman.shape[0] + 2.0) / 4.0) ** (-1.0 / (silverman.shape[0] + 4.0))
      silverman[silverman<1e-6] = 1e-6
      silverman = 1.0 / silverman
      
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
      silverman = silverman * (self.weight() * (silverman.shape[0] + 2.0) / 4.0) ** (-1.0 / (silverman.shape[0] + 4.0))
      silverman[silverman<1e-6] = 1e-6
      silverman = 1.0 / silverman
      
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



class MeanShiftCompositeScale:
  """This optimises the scale of MeanShift objects that use the composite kernel type - designed to support all cases, including directional kernels, assuming its a single composite kernel containing a list of other kernels (child kernels can be composite, but would all be fixed to share the same scale if that is being optimised). Optimises each part of the outer composite kernel seperatly, rather than considering a single linear scale parameter as the built in methods do. After construction you call the add_param_* methods to add all parameters you want to optimise over and then the object pretends its a function - simply call on any mean shift object (no parameters) and it does a drunk gradient decent (not sure what to call this - stupid simplex sort) from the closest point in the parameter space of the object until it hits a local maximum. Note that this means that the provided object should have sensible parameters when you start."""
  def __init__(self, kernel):
    """You provide a kernel configuration that must start 'composite(' (it will extract dimension counts from this, hence the requirement). Any parameters you want to control within it should be replaced with %(key)s (yes, s for string! you can tie parameters together by giving them the same key) so they can be set via the parameter system."""
    self.kernel = kernel
    
    start = 'composite('
    assert(self.kernel.startswith(start) and self.kernel[-1]==')')
    
    # Extract the terms from the outer kernel - need to handle the possibility of commas within a kernel definition...
    parts = self.kernel[len(start):-1].split(',')
    bits = []
    
    excess = 0
    for part in parts:
      if excess==0:
        bits.append(part)
      else:
        bits[-1] = bits[-1] + part
        
      excess += part.count('(')
      excess -= part.count(')')
    assert(excess==0)
    
    # Seperate out sizes, also generate offsets...
    self.sizes = numpy.array([int(bit.split(':')[0]) for bit in bits])
    self.offsets = numpy.concatenate((numpy.array([0]), numpy.cumsum(self.sizes)))
    
    # Generate a dummy data matrix, to use when creating MS objects to copy kernels and scale from...
    self.dummy = numpy.zeros(sum(self.sizes), dtype=numpy.float32)
        
    # List of parameters which it is going to optimise - recorded as tuples of (influence, log of low, log of high, steps [, re]). influence is a number indexing a internal kernel, or a string indexing a kernel parameter. For kernel ones a compiled regular expression for extracting the value from a kernel string is provided...
    self.params = []
    
    # Create the cache, to contain each MS object considered - for Fisher kernels this avoids creating the (expensive and large) cache repeatedly...
    self.cache = dict()


  def add_param_scale(self, index, low = 1.0/512, high = 1024.0, steps = 20):
    """Allows you to add a parameter to optimise on the scale parameters - you provide the index of the sub-kernel of the composite kernel to scale, a low and high scale and the number of steps. Last three have defaults. Interpolation to get the discrete values is inclusive and logarithmic."""
    assert(index<self.sizes.shape[0] and index>=0)
    self.params.append((index, numpy.log(low), numpy.log(high), steps))
  
  def add_param_kernel(self, key, low = 4.0, high = 8192.0, steps = 12):
    """Allows you to add a parameter to optimise on the kernel construction - you provide the key for the kernel string to twiddle, a low and high scale and the number of steps. Last three have defaults. Interpolation to get the discrete values is inclusive and logarithmic."""
    assert(('%%(%s)s' % key) in self.kernel)
    
    fnum = '[0-9]+\.?[0-9]*' # Matches a floating point number.
    reps = defaultdict(lambda: fnum)
    reps[key] = '<$$$>' # This should never occur in a kernel spec!
    
    reg = (self.kernel % reps).replace('(', '\(').replace(')', '\)').replace('<$$$>', '(' + fnum + ')') # Escape brackets in string, putting code to match a float into each parameter except for the one we are after where we surround the float matching expression with brackets, so we can extract it. The repeated use of brackets is what makes this so fucking painful:-/
    reg = re.compile(reg)
    
    self.params.append((key, numpy.log(low), numpy.log(high), steps, reg))
  
  
  def __set_pos(self, loc, ms):
    """Internal method to set the parameters of a mean shift object to those at the given location in the search space."""
    kdic = None
    key = []
    
    scale = ms.get_scale()
    
    for i, param in enumerate(self.params):
      val = numpy.exp((loc[i] / float(param[3]-1)) * (param[2]-param[1]) + param[1])
      
      if len(param)<5:
        offset = self.offsets[param[0]]
        scale[offset:offset + self.sizes[param[0]]] = val
      else:
        if kdic==None:
          kdic = dict()
        kdic[param[0]] = str(val)
        key.append('%s=%i' % (param[1], loc[i]))
        
    ms.set_scale(scale)
    
    if kdic!=None:
      key = ';'.join(key)
      
      if key not in self.cache:
        source = MeanShift()
        source.set_data(self.dummy, 'f')
        source.set_kernel(self.kernel % kdic)
        self.cache[key] = source
        
      else:
        source = self.cache[key]
      
      ms.copy_kernel(source)


  def __call__(self, ms, max_tries = None):
    """Optimise the given MeanShift object; the optional max tries parameter is how many steps (step == trying both directions for a single dimension) to try before stopping - it defaults to None which means it will go until every possible step results in a higher cost. Returns the number of tries it did."""
    
    ## First figure out its current parameter coordinates - easy for scale, whilst kernel parameters get a little weird...
    scale = ms.get_scale()
    
    def value(param):
      if len(param)<5:
        # Standard scale parameter - easy...
        val = numpy.log(scale[self.offsets[param[0]]])
      else:
        # Kernel parameter - use the re...
        val = numpy.log(float(param[4].match(ms.get_kernel()).group(1)))
    
      pos = int((param[3]-1) * (val - param[1]) / (param[2] - param[1]) + 0.5)
      
      if pos<0:
        pos = 0
      elif pos>=param[3]:
        pos = param[3] - 1
      
      return pos
    
    loc = numpy.array([value(param) for param in self.params])
    
    ## Record its score at the starting location...
    self.__set_pos(loc, ms)
    current = ms.loo_nll()
    
    ## Loop and try 'random' directions, noting that it avoids duplicate attempts, which is also helpful for minima detection...
    tries = 0
    index = -1
    since_last_step = 0
    
    while tries!=max_tries:
      # 'Randomly' select the dimension to optimise...
      index = (index + 1) % len(self.params)
      tries += 1
      
      # Try both directions, making the jump if either is an improvement...
      if loc[index] > 0:
        loc[index] -= 1
        self.__set_pos(loc, ms)
        option = ms.loo_nll()
        if option<current:
          current = option
          since_last_step = 0
          continue
        else:
          loc[index] += 1
      
      if loc[index]+1 < self.params[index][3]:
        loc[index] += 1
        self.__set_pos(loc, ms)
        option = ms.loo_nll()
        if option<current:
          current = option
          since_last_step = 0
          continue
        else:
          loc[index] -= 1
      
      since_last_step += 1
      if since_last_step >= len(self.params):
        return tries
    
    return tries
