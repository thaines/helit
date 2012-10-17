# Copyright 2012 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import numpy

from utils.start_cpp import start_cpp



class Test:
  """Interface for a test definition. This provides the concept of a test that an exemplar either passes or fails. The test is actually defined by some arbitary entity made by a matching generator, but this object is required to actually do the test - contains the relevant code and any shared parameters to keep memory consumption low, as there could be an aweful lot of tests. The seperation of test from generator is required as there are typically many methods to generate a specific test - generators inherit from the relevant test object."""
  
  def do(self, test, es, index = slice(-1)):
    """Does the test. Given the entity that defines the actual test and an ExemplarSet to run the test on. An optional index for the features can be provided (Passed directly through to the exemplar sets [] operator, so its indexing rules apply.), but if omitted it runs for all. Return value is a boolean numpy array indexed by the relative exemplar indices, that gives True if it passsed the test, False if it failed."""
    raise NotImplementedError
  
  
  def testCodeC(self, name, exemplar_list):
    """Provides C code to perform the test - provides a C function that is given the test object as a pointer to the first byte and the index of the exemplar to test; it then returns true or false dependeing on if it passes the test or not. Returned string contains a function with the calling convention `bool <name>(PyObject * data, void * test, size_t test_length, int exemplar)`. data is a python tuple indexed by channel containning the object to be fed to the exemplar access function. To construct this function it needs the return value of listCodeC for an ExemplarSet, so it can get the calling convention to access the channel. When compiled the various functions must be avaliable."""
    raise NotImplementedError



class AxisSplit(Test):
  """Possibly the simplest test you can apply to continuous data - an axis-aligned split plane. Can also be applied to discrete data if that happens to make sense. This stores which channel to apply the tests to, whilst each test entity is a 8 byte string, encoding an int32 then a float32 - the first indexes the feature to use from the channel, the second the offset, such that an input has this value subtracted and then fails the test if the result is less than zero or passes if it is greater than or equal to."""
  def __init__(self, channel):
    """Needs to know which channel this test is applied to."""
    self.channel = channel
  
  def do(self, test, es, index = slice(None)):
    value_index = numpy.fromstring(test[0:4], dtype=numpy.int32, count=1)
    offset = numpy.fromstring(test[4:8], dtype=numpy.float32, count=1)
    
    values = es[self.channel, index, value_index[0]]
    
    return (values-offset[0])>=0.0
  
  
  def testCodeC(self, name, exemplar_list):
    ret = start_cpp() + """
    bool %(name)s(PyObject * data, void * test, size_t test_length, int exemplar)
    {
     int feature = *(int*)test;
     float offset = *((float*)test + 1);
     %(channelType)s channel = (%(channelType)s)PyTuple_GetItem(data, %(channel)i);
     float value = (float)%(channelName)s_get(channel, exemplar, feature);
     return (value-offset)>=0.0;
    }
    """%{'name':name, 'channel':self.channel, 'channelName':exemplar_list[self.channel]['name'], 'channelType':exemplar_list[self.channel]['itype']}
    return ret



class LinearSplit(Test):
  """Does a linear split of data based on some small set of values. Can be applied to discrete data, though that would typically be a bit strange. This object stores both the channel to which the test is applied and how many dimensions are used, whilst the test entity is a string encoding three things in sequence. First are the int32 indices of the features from the exemplars channel to use, second are the float32 values forming the vector that is dot producted with the extracted values to project to the line perpendicular to the plane, and finally the float32 offset, subtracted from the line position to make it a negative to fail, zero or positive to pass decision."""
  def __init__(self, channel, dims):
    """Needs to know which channel it is applied to and how many dimensions are to be considered."""
    self.channel = channel
    self.dims = dims
    
  def do(self, test, es, index = slice(None)):
    ss1 = 4*self.dims
    ss2 = 2*ss1
    ss3 = ss2+4
    
    value_indices = numpy.fromstring(test[0:ss1], dtype=numpy.int32, count=self.dims)
    plane_axis = numpy.fromstring(test[ss1:ss2], dtype=numpy.float32, count=self.dims)
    offset = numpy.fromstring(test[ss2:ss3], dtype=numpy.float32, count=1)
    
    values = es[self.channel, index, value_indices]
    
    return ((values*plane_axis.reshape((1,-1))).sum(axis=1) - offset)>=0.0


  def testCodeC(self, name, exemplar_list):
    ret = start_cpp() + """
    bool %(name)s(PyObject * data, void * test, size_t test_length, int exemplar)
    {
     int * feature = (int*)test;
     float * plane_axis = (float*)test + %(dims)i;
     float offset = *((float*)test + %(dims)i*2);
     %(channelType)s channel = (%(channelType)s)PyTuple_GetItem(data, %(channel)i);
     
     float value = 0.0;
     for (int i=0;i<%(dims)i;i++)
     {
      float v = (float)%(channelName)s_get(channel, exemplar, feature[i]);
      value += v*plane_axis[i];
     }
     
     return (value-offset)>=0.0;
    }
    """%{'name':name, 'channel':self.channel, 'channelName':exemplar_list[self.channel]['name'], 'channelType':exemplar_list[self.channel]['itype'], 'dims':self.dims}
    return ret



class DiscreteBucket(Test):
  """For discrete values. The test is applied to a single value, and consists of a list of values such that if it is equal to one of them it passes, but if it is not equal to any of them it fails. Basically a binary split of categorical data. The test entity is a string encoding first a int32 of the index of which feature to use, followed by the remainder of the string forming a list of int32's that constitute the values that result in success."""
  def __init__(self, channel):
    """Needs to know which channel this test is applied to."""
    self.channel = channel
  
  def do(self, test, es, index = slice(None)):
    t = numpy.fromstring(test, dtype=numpy.int32)
    
    values = es[self.channel, index, t[0]]
    
    return numpy.in1d(values, t[1:])
  
  
  def testCodeC(self, name, exemplar_list):
    ret = start_cpp() + """
    bool %(name)s(PyObject * data, void * test, size_t test_length, int exemplar)
    {
     size_t steps = test_length>>2;
     int * accept = (int*)test;
     %(channelType)s channel = (%(channelType)s)PyTuple_GetItem(data, %(channel)i);
     int value = (int)%(channelName)s_get(channel, exemplar, accept[0]);
    
     for (size_t i=1; i<steps; i++)
     {
      if (accept[i]==value) return true;
     }
    
     return false;
    }
    """%{'name':name, 'channel':self.channel, 'channelName':exemplar_list[self.channel]['name'], 'channelType':exemplar_list[self.channel]['itype']}
    return ret
