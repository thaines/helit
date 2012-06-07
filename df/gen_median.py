# Copyright 2012 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import numpy
import numpy.random

from generators import Generator
from tests import *

from utils.start_cpp import start_cpp



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
        pos = numpy.searchsorted(cw, half)
        t = (half - cw[pos-1]) / max(cw[pos] - cw[pos-1], 1e-6)
        median = (1.0-t)*values[pos-1] + t*values[pos]
      
      yield numpy.asarray([ind], dtype=numpy.int32).tostring() + numpy.asarray([median], dtype=numpy.float32).tostring()


  def genCodeC(self, name, exemplar_list):
    code = start_cpp() + """
    struct State%(name)s
    {
     void * test; // Will be the length of a 32 bit int followed by a float.
     size_t length;
     
     int countRemain;
     
     float * temp; // Temporary used for calculating the median.
    };
    
    int %(name)s_float_comp(const void * lhs, const void * rhs)
    {
     float l = (*(float*)lhs);
     float r = (*(float*)rhs);
     
     if (l<r) return -1;
     if (l>r) return 1;
     return 0;
    }
    
    void %(name)s_init(State%(name)s & state, PyObject * data, Exemplar * test_set)
    {
     assert(sizeof(int)==4);
     
     int count = 0;
     while (test_set)
     {
      count++;
      test_set = test_set->next;
     }
     
     state.length = sizeof(int) + sizeof(float);
     state.test = malloc(state.length);
     
     state.countRemain = %(count)i;
     
     state.temp = (float*)malloc(sizeof(float) * 2 * count);
    }
    
    bool %(name)s_next(State%(name)s & state, PyObject * data, Exemplar * test_set)
    {
     // Check if we are done...
      if (state.countRemain==0)
      {
       free(state.test);
       free(state.temp);
       return false;
      }
      
      state.countRemain--;
      
     // Select a random feature...
      %(channelType)s cd = (%(channelType)s)PyTuple_GetItem(data, %(channel)i);
      int feat = lrand48() %% %(channelName)s_features(cd);
     
     // Extract the values...
      int count = 0;
      while (test_set)
      {
       state.temp[count*2] = %(channelName)s_get(cd, test_set->index, feat);
       state.temp[count*2+1] = test_set->weight;
       
       count++;
       test_set = test_set->next;
      }
     
     // Sort them...
      qsort(state.temp, count, sizeof(float)*2, %(name)s_float_comp);
     
     // Pull out the median...
      float median;
      if (%(ignoreWeights)s||(count<2))
      {
       int half = count/2;
       if ((count%%2)==1) median = state.temp[half*2];
       else median = 0.5*(state.temp[(half-1)*2] + state.temp[half*2]);
      }
      else
      {
       // Convert to a cumulative sum...
        for (int i=1;i<count;i++)
        {
         state.temp[i*2+1] += state.temp[i*2-1];
        }
        
        float half = 0.5*state.temp[(count-1)*2+1];

       // Find the position just after the half way point...
        int low = 0;
        int high = count-1;
        
        while (low<high)
        {
         int middle = (low+high)/2;
         if (state.temp[middle*2+1]<half)
         {
          if (low==middle) middle++;
          low = middle;
         }
         else
         {
          if (high==middle) middle--;
          high = middle;
         }
        }
      
       // Use linear interpolation to select a value...
        float t = half - state.temp[low*2-1];
        float div = state.temp[low*2+1] - state.temp[low*2-1];
        if (div<1e-6) div = 1e-6;
        t /= div;
        median = (1.0-t) * state.temp[low*2-2] + t * state.temp[low*2];
      }
     
     // Store the test and return...
      ((int*)state.test)[0] = feat;
      ((float*)state.test)[1] = median;
     
     return true;
    }
    """%{'name':name, 'channel':self.channel, 'channelName':exemplar_list[self.channel]['name'], 'channelType':exemplar_list[self.channel]['itype'], 'count':self.count, 'ignoreWeights':('true' if self.ignoreWeights else 'false')}
    
    return (code, 'State'+name)



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

  def genCodeC(self, name, exemplar_list):
    code = start_cpp() + """
    struct State%(name)s
    {
     void * test;
     size_t length;
     
     int dimRemain;
     int dirRemain;
     
     float * dirs; // Vectors giving points uniformly distributed on the hyper-sphere.
     int * feat; // The features to index at this moment.
     
     float * temp; // Temporary used for calculating the median.
    };
    
    int %(name)s_float_comp(const void * lhs, const void * rhs)
    {
     float l = (*(float*)lhs);
     float r = (*(float*)rhs);
     
     if (l<r) return -1;
     if (l>r) return 1;
     return 0;
    }
    
    void %(name)s_init(State%(name)s & state, PyObject * data, Exemplar * test_set)
    {
     assert(sizeof(int)==4);
     
     // Count how many exemplars are in the input...
      int count = 0;
      while (test_set)
      {
       count++;
       test_set = test_set->next;
      }
     
     // Setup the output...
      state.length = sizeof(int) * %(dims)i + sizeof(float) * (%(dims)i+1);
      state.test = malloc(state.length);
     
     // Counters so we know when we are done...
      state.dimRemain = %(dimCount)i;
      state.dirRemain = 0;
     
     // Generate a bunch of random directions...
      state.dirs = (float*)malloc(sizeof(float)*%(dims)i*%(dirCount)i);
      for (int d=0;d<%(dirCount)i;d++)
      {
       float length = 0.0;
       int base = %(dims)i * d;
       for (int f=0; f<%(dims)i; f++)
       {
        double u = 1.0-drand48();
        double v = 1.0-drand48();
        float bg = sqrt(-2.0*log(u)) * cos(2.0*M_PI*v);
        length += bg*bg;
        state.dirs[base+f] = bg;
       }
       
       length = sqrt(length);
       for (int f=0; f<%(dims)i; f++)
       {
        state.dirs[base+f] /= length;
       }
      }
      
     // Which features are currently being used...
      state.feat = (int*)malloc(sizeof(int)*%(dims)i);
     
     // Temporary for median calculation...
      state.temp = (float*)malloc(sizeof(float) * 2 * count);
     
     // Safety...
      %(channelType)s cd = (%(channelType)s)PyTuple_GetItem(data, %(channel)i);
      int featCount = %(channelName)s_features(cd);
      if (%(dims)i>featCount)
      {
       state.dimRemain = 0; // Effectivly cancels work.
      }
    }
    
    bool %(name)s_next(State%(name)s & state, PyObject * data, Exemplar * test_set)
    {
     // Need access to the data...
      %(channelType)s cd = (%(channelType)s)PyTuple_GetItem(data, %(channel)i);
      
     // If we are done for this set of features select a new set...
      if (state.dirRemain==0)
      {
       if (state.dimRemain==0)
       {
        free(state.test);
        free(state.dirs);
        free(state.feat);
        free(state.temp);
        return false;
       }
       state.dimRemain--;
      
       // Select a new set of features...
        int featCount = %(channelName)s_features(cd);
        for (int f=0; f<%(dims)i; f++)
        {
         state.feat[f] = lrand48() %% (featCount-f);
         for (int j=0; j<f; j++)
         {
          if (state.feat[j]<=state.feat[f]) state.feat[f]++;
         }
        }
        
       // Reset the counter...
        state.dirRemain = %(dirCount)i;
      }
      state.dirRemain--;
         
     // Extract the values, projecting them using the current direction...
      int count = 0;
      while (test_set)
      {
       float val = 0.0;
       int base = %(dims)i * state.dirRemain;
       for (int f=0; f<%(dims)i; f++)
       {
        val += state.dirs[base+f] * %(channelName)s_get(cd, test_set->index, state.feat[f]);
       }
       
       state.temp[count*2] = val;
       state.temp[count*2+1] = test_set->weight;
       
       count++;
       test_set = test_set->next;
      }
     
     // Sort them...
      qsort(state.temp, count, sizeof(float)*2, %(name)s_float_comp);
     
     // Pull out the median...
      float median;
      if (%(ignoreWeights)s||(count<2))
      {
       int half = count/2;
       if ((count%%2)==1) median = state.temp[half*2];
       else median = 0.5*(state.temp[(half-1)*2] + state.temp[half*2]);
      }
      else
      {
       // Convert to a cumulative sum...
        for (int i=1;i<count;i++)
        {
         state.temp[i*2+1] += state.temp[i*2-1];
        }
        
        float half = 0.5*state.temp[(count-1)*2+1];

       // Find the position just after the half way point...
        int low = 0;
        int high = count-1;
        
        while (low<high)
        {
         int middle = (low+high)/2;
         if (state.temp[middle*2+1]<half)
         {
          if (low==middle) middle++;
          low = middle;
         }
         else
         {
          if (high==middle) middle--;
          high = middle;
         }
        }
      
       // Use linear interpolation to select a value...
        float t = half - state.temp[low*2-1];
        float div = state.temp[low*2+1] - state.temp[low*2-1];
        if (div<1e-6) div = 1e-6;
        t /= div;
        median = (1.0-t) * state.temp[low*2-2] + t * state.temp[low*2];
      }
     
     // Store it all in the output...
      for (int i=0; i<%(dims)i;i++)
      {
       ((int*)state.test)[i] = state.feat[i];
      }
      int base = %(dims)i * state.dirRemain;
      for (int i=0; i<%(dims)i;i++)
      {
       ((float*)state.test)[%(dims)i+i] = state.dirs[base+i];
      }
      ((float*)state.test)[2*%(dims)i] = median;
      
     return true;
    }
    """%{'name':name, 'channel':self.channel, 'channelName':exemplar_list[self.channel]['name'], 'channelType':exemplar_list[self.channel]['itype'], 'dims':self.dims, 'dimCount':self.dimCount, 'dirCount':self.dirCount, 'ignoreWeights':('true' if self.ignoreWeights else 'false')}
    
    return (code, 'State'+name)
