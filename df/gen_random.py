# Copyright 2012 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import numpy
import numpy.random

from generators import Generator
from tests import *

from utils.start_cpp import start_cpp



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


  def genCodeC(self, name, exemplar_list):
    code = start_cpp() + """
    struct State%(name)s
    {
     void * test; // Will be the length of a 32 bit int followed by a float.
     size_t length;
     
     int dimRemain;
     int splitRemain;
     
     int feat;
     float mean;
     float sd;
    };
    
    void %(name)s_init(State%(name)s & state, PyObject * data, Exemplar * test_set)
    {
     assert(sizeof(int)==4);
     
     state.length = sizeof(int) + sizeof(float);
     state.test = malloc(state.length);
     
     state.dimRemain = %(dimCount)i;
     state.splitRemain = 0;
    }
    
    bool %(name)s_next(State%(name)s & state, PyObject * data, Exemplar * test_set)
    {
     // If we have used up all splits for this feature select a new one...
     if (state.splitRemain==0)
     {
      // If we have run out of features to select we are done - return as such...
       if (state.dimRemain==0)
       {
        free(state.test);
        return false;
       }
       state.dimRemain--;
     
      // Get the relevent channels object...
       %(channelType)s cd = (%(channelType)s)PyTuple_GetItem(data, %(channel)i);
       
      // Select a new feature...
       state.feat = lrand48() %% %(channelName)s_features(cd);
      
      // Calculate the mean and standard deviation of the data set, for the selected feature...
       float sum = 0.0;
       float mean = 0.0;
       float mean2 = 0.0;
       
       while (test_set)
       {
        float x = %(channelName)s_get(cd, test_set->index, state.feat);
        if (%(ignoreWeights)s)
        {
         sum += 1.0;
         float delta = x - mean;
         mean += delta/sum;
         mean2 += delta * (x - mean);
        }
        else
        {
         float newSum = sum + test_set->weight;
         float delta = x - mean;
         float mean_delta = delta * test_set->weight / newSum;
         mean += mean_delta;
         mean2 += sum * delta * mean_delta;
         sum = newSum;
        }
       
        test_set = test_set->next;
       }
       
       state.mean = mean;
       state.sd = sqrt(mean2/sum);
       if (state.sd<1e-6) state.sd = 1e-6;
       
       state.splitRemain = %(splitCount)i;
     }
     
     // Output a split point drawn from the Gaussian...
      state.splitRemain--;
      
      double u = 1.0-drand48();
      double v = 1.0-drand48();
      float bg = sqrt(-2.0*log(u)) * cos(2.0*M_PI*v);
      float split = state.mean + state.sd * bg;
      
      ((int*)state.test)[0] = state.feat;
      ((float*)state.test)[1] = split;
     
     return true;
    }
    """%{'name':name, 'channel':self.channel, 'channelName':exemplar_list[self.channel]['name'], 'channelType':exemplar_list[self.channel]['itype'], 'dimCount':self.dimCount, 'splitCount':self.splitCount, 'ignoreWeights':('true' if self.ignoreWeights else 'false')}
    
    return (code, 'State'+name)


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
      for i in xrange(self.dims): # This loop is not quite right - could result in the same feature twice. Odds are low enough that its not really worth caring about however.
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


  def genCodeC(self, name, exemplar_list):
    code = start_cpp() + """
    struct State%(name)s
    {
     void * test; 
     size_t length;
     
     float * dirs; // Vectors giving points uniformly distributed on the hyper-sphere.
     int * feat; // The features to index at this moment.
     
     float mean;
     float sd;
     
     // Control counters - all count down...
      int featRemain;
      int dirRemain;
      int splitRemain;
    };
    
    void %(name)s_init(State%(name)s & state, PyObject * data, Exemplar * test_set)
    {
     assert(sizeof(int)==4);
     
     // Output state...
      state.length = sizeof(int) * %(dims)i + sizeof(float) * (%(dims)i+1);
      state.test = malloc(state.length);
     
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
     
     // Setup the counters so we do the required work when next is called...
      state.featRemain = %(dimCount)i;
      state.dirRemain = 0;
      state.splitRemain = 0;
      
     // Safety...
      %(channelType)s cd = (%(channelType)s)PyTuple_GetItem(data, %(channel)i);
      int featCount = %(channelName)s_features(cd);
      if (%(dims)i>featCount)
      {
       state.featRemain = 0; // Effectivly cancels work.
      }
    }
    
    bool %(name)s_next(State%(name)s & state, PyObject * data, Exemplar * test_set)
    {
     if (state.splitRemain==0)
     {
      %(channelType)s cd = (%(channelType)s)PyTuple_GetItem(data, %(channel)i);
     
      if (state.dirRemain==0)
      {
       if (state.featRemain==0)
       {
        free(state.feat);
        free(state.dirs);
        free(state.test);
        return false;
       }
       state.featRemain--;
       
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
      
      // For the new direction calculate the mean and standard deviation with the current features...
       float sum = 0.0;
       float mean = 0.0;
       float mean2 = 0.0;
       
       while (test_set)
       {
        float x = 0.0;
        int base = %(dims)i * state.dirRemain;
        for (int f=0; f<%(dims)i; f++)
        {
         x += state.dirs[base+f] * %(channelName)s_get(cd, test_set->index, state.feat[f]);
        }
        
        if (%(ignoreWeights)s)
        {
         sum += 1.0;
         float delta = x - mean;
         mean += delta/sum;
         mean2 += delta * (x - mean);
        }
        else
        {
         float newSum = sum + test_set->weight;
         float delta = x - mean;
         float mean_delta = delta * test_set->weight / newSum;
         mean += mean_delta;
         mean2 += sum * delta * mean_delta;
         sum = newSum;
        }
       
        test_set = test_set->next;
       }
       
       state.mean = mean;
       state.sd = sqrt(mean2/sum);
       if (state.sd<1e-6) state.sd = 1e-6;
       
      // Reset the counter...
       state.splitRemain = %(splitCount)i;
     }
     state.splitRemain--;
     
     // Use the mean and standard deviation to select a split point...
      double u = 1.0-drand48();
      double v = 1.0-drand48();
      float bg = sqrt(-2.0*log(u)) * cos(2.0*M_PI*v);
      float split = state.mean + state.sd * bg;
      
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
      ((float*)state.test)[2*%(dims)i] = split;
     
     return true;
    }
    """%{'name':name, 'channel':self.channel, 'channelName':exemplar_list[self.channel]['name'], 'channelType':exemplar_list[self.channel]['itype'], 'dims':self.dims, 'dimCount':self.dimCount, 'dirCount':self.dirCount, 'splitCount':self.splitCount, 'ignoreWeights':('true' if self.ignoreWeights else 'false')}
    
    return (code, 'State'+name)



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


  def genCodeC(self, name, exemplar_list):
    code = start_cpp() + """
    struct State%(name)s
    {
     void * test; // Will be the length of two 32 bit ints.
     size_t length;
     
     int feat; // Current feature.
     int featRemain; // How many more times we need to select a feature to play with.
     
     int * value; // List of values drawn from the feature - we return each in turn.
     int valueLength; // Reduced as each feature is drawn.
     float * weight; // Aligns with value; temporary used for the sampling.
    };
    
    int %(name)s_int_comp(const void * lhs, const void * rhs)
    {
     return (*(int*)lhs) - (*(int*)rhs);
    }
    
    void %(name)s_init(State%(name)s & state, PyObject * data, Exemplar * test_set)
    {
     assert(sizeof(int)==4);
     
     state.test = malloc(2*sizeof(int));
     state.length = 2*sizeof(int);
     state.feat = -1;
     state.featRemain = %(featCount)i;
     state.value = (int*)malloc(%(valueCount)i*sizeof(int));
     state.valueLength = 0;
     state.weight = (float*)malloc(%(valueCount)i*sizeof(float));
    }
    
    bool %(name)s_next(State%(name)s & state, PyObject * data, Exemplar * test_set)
    {
     // Check if we need to create a new set of values...
      if (state.valueLength==0)
      {
       // Check if we are done - if so clean up and return...
        if (state.featRemain==0)
        {
         free(state.weight);
         free(state.value);
         free(state.test);
         return false;
        }
        state.featRemain--;
       
       // Get the relevent channels object...
       %(channelType)s cd = (%(channelType)s)PyTuple_GetItem(data, %(channel)i);
       
       // Select a new feature...
        state.feat = lrand48() %% %(channelName)s_features(cd);
       
       // Generate a set of values - use a method based on a single pass through the linked list...
        float minWeight = 1.0;
        while (test_set)
        {
         if (test_set->weight>1e-6)
         {
          float w = pow(drand48(), 1.0/test_set->weight);
          
          if (state.valueLength<%(valueCount)i)
          {
           state.value[state.valueLength] = test_set->index;
           state.weight[state.valueLength] = w;
           if (minWeight>w) minWeight = w;
           state.valueLength++;
          }
          else
          {
           if (w>minWeight) // Below is not very efficient, but don't really care - valueCount tends to be low enough that optimisation is not worthwhile..
           {
            int lowest = 0;
            for (int i=1; i<%(valueCount)i; i++)
            {
             if (state.weight[lowest]>state.weight[i]) lowest = i;
            }
            
            state.value[lowest] = test_set->index;
            state.weight[lowest] = w;
            
            minWeight = 1.0;
            for (int i=0; i<%(valueCount)i; i++)
            {
             if (minWeight>state.weight[i]) minWeight = state.weight[i];
            }
           }
          }
         }
         
         test_set = test_set->next;
        }
       
       // Convert exemplar numbers to actual values...
        for (int i=0; i<state.valueLength; i++)
        {
         state.value[i] = %(channelName)s_get(cd, state.value[i], state.feat);
        }
        
       // Remove duplicates...
        qsort(state.value, state.valueLength, sizeof(int), %(name)s_int_comp);
        
        int out = 1;
        for (int i=1; i<state.valueLength; i++)
        {
         if (state.value[i]!=state.value[i-1])
         {
          state.value[out] = state.value[i];
          out++;
         }         
        }
        state.valueLength = out;
      }
    
     // Get and arrange as the output the next value...
      state.valueLength -= 1;
      ((int*)state.test)[0] = state.feat;
      ((int*)state.test)[1] = state.value[state.valueLength];
      
     return true;
    }
    """%{'name':name, 'channel':self.channel, 'channelName':exemplar_list[self.channel]['name'], 'channelType':exemplar_list[self.channel]['itype'], 'featCount':self.featCount, 'valueCount':self.valueCount}
    
    return (code, 'State'+name)
