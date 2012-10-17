# Copyright 2012 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import numpy



class ExemplarSet:
  """An interface for a set of feature vectors, referred to as exemplars - whilst a data matrix will typically be used this allows the possibility of exemplars for which that is impractical, i.e. calculating them on the fly if there is an unreasonable number of features within each exemplar. Also supports the concept of channels, which ties in with the test generation so you can have different generators for each channel. For trainning the 'answer' is stored in its own channel (Note that, because that channel will not exist for novel features it should always be the last channel, so that indexing is consistant, unless it is replaced with a dummy channel.), the type of which will depend on the problem being solved. Also allows the mixing of both continuous and discrete values."""
  
  def exemplars(self):
    """Returns how many exemplars are provided."""
    raise NotImplementedError
  
  def channels(self):
    """Returns how many channels of features are provided. They are indexed {0, ..., return-1}."""
    raise NotImplementedError
  
  def name(self, channel):
    """Returns a string as a name for a channel - this is optional and provided more for human usage. Will return None if the channel in question has no name."""
    return None

  def dtype(self, channel):
    """Returns the numpy dtype used for the given channel. numpy.float32 and numpy.int32 are considered to be the standard choices."""
    raise NotImplementedError

  def features(self, channel):
    """Returns how many features exist for each exemplar for the given channel - they can then be indexed {0, .., return-1} for the given channel."""
    raise NotImplementedError
  
  def __getitem__(self, index):
    """Actual data access is via the [] operator, with 3 entities - [channel, exemplar(s), feature(s)]. channel must index the channel, and indicates which channel to get the data from - must always be an integer. exemplars(s) indicates which examples to return and features(s) which features to return for each of the exemplars. For both of these 3 indexing types must be supported - a single integer, a slice, or a numpy array of indexing integers. For indexing integers the system is designed to work such that repetitions are never used, though that is in fact supported most of the time by actual implimentations. The return value must always have the type indicated by the dtype method for the channel in question. If both are indexed with an integer then it will return a single number (But still of the numpy dtype.); if only 1 is an integer a 1D numpy array; and if neither are integers a 2D numpy array, indexed [relative exemplar, relative feature]. Note that this last requirement is not the numpy default, which would actually continue to give a 1D array rather than the 2D subset defined by two sets of indicies."""
    raise NotImplementedError
  
  
  def codeC(self, channel, name):
    """Returns a dictionary containning all the entities needed to access the given channel of the exemplar from within C, using name to provide a unique string to avoid namespace clashes. Will raise a NotImplementedError if not avaliable. `['type'] = The C type of the channel, often 'float'. ['input'] = The input object to be passed into the C code, must be protected from any messing around that scipy.weave might do. ['itype'] = The input type in C, as a string, usually 'PyObject *' or 'PyArrayObject *'. ['get'] = Returns code for a function to get values from the channel of the exemplar; has calling convention <type> <name>_get(<itype> input, int exemplar, int feature). ['exemplars'] = Code for a function to get the number of exemplars; has calling convention int <name>_exemplars(<itype> input). Will obviously return the same value for all channels, so can be a bit redundant. ['features'] = Code for a function that returns how many features the channel has; calling convention is int <name>_features(<itype> input). ['name'] is also provided, which contains the base name handed to this method, for conveniance.`"""
    raise NotImplementedError
  
  def listCodeC(self, name):
    """Helper method - returns a tuple indexed by channel that gives the dictionary returned by codeC for each channel in this exemplar. It generates the names using the provided name by adding the number indexing the channel to the end. Happens to by the required input elsewhere."""
    return tuple(map(lambda c: self.codeC(c, name+str(c)), xrange(self.channels())))
  
  def tupleInputC(self):
    """Helper method, that can be overriden - returns a tuple containing the inputs needed for the exemplar."""
    return tuple(map(lambda c: self.codeC(c, name+str(c))['input'], xrange(self.channels())))
  
  def key(self):
    """Provides a unique string that can be used to hash the results of codeC, to avoid repeated generation. Must be implimented if codeC is implimented."""
    raise NotImplementedError
    



class MatrixES(ExemplarSet):
  """The most common exemplar set - basically what you use when all the feature vectors can be computed and then stored in memory without issue. Contains a data matrix for each channel, where these are provided by the user."""
  def __init__(self, *args):
    """Optionally allows you to provide a list of numpy data matrices to by the channels data matrices. Alternativly you can use the add method to add them, one after another, post construction, or some combination of both. All data matrices must be 2D numpy arrays, with the first dimension, indexing the exemplar, being the same size in all cases. (If there is only 1 exemplar then it will accept 1D arrays.)"""
    self.dm = list(map(lambda a: a.reshape((1,-1)) if len(a.shape)==1 else a, args))
    for dm in self.dm:
      assert(len(dm.shape)==2)
      assert(dm.shape[0]==self.dm[0].shape[0])
  
  def add(self, dm):
    """Adds a new data matrix of information as another channel. Returns its channel index. If given a 1D matrix assumes that there is only one exemplar and adjusts it accordingly."""
    if len(dm.shape)==1: dm = dm.reshape((1,-1))
    assert(len(dm.shape)==2)
    self.dm.append(dm)
    assert(dm.shape[0]==self.dm[0].shape[0])
    return len(self.dm)-1
  
  def append(self, *args):
    """Allows you to add exemplars to the structure, by providing a set of data matrices that align with those contained, which contain the new exemplars. Note that this is slow and generally ill advised. If adding a single new feature the arrays can be 1D."""
    assert(len(args)==len(self.dm))
    for i, (prev, extra) in enumerate(zip(self.dm, args)):
      if len(extra.shape)==1: extra = extra.reshape((1,-1))
      self.dm[i] = numpy.append(prev, extra, 0)
  
  def exemplars(self):
    return self.dm[0].shape[0]
  
  def channels(self):
    return len(self.dm)
  
  def dtype(self, channel):
    return self.dm[channel].dtype
  
  def features(self, channel):
    return self.dm[channel].shape[1]
  
  def __getitem__(self, index):
    a = numpy.asarray(index[1]).reshape(-1)
    b = numpy.asarray(index[2]).reshape(-1)
    if a.shape[0]==1 or b.shape[0]==1: return self.dm[index[0]][index[1],index[2]]
    else: return self.dm[index[0]][numpy.ix_(a,b)]
  
  def codeC(self, channel, name):
    ret = dict()
    
    inp = self.dm[channel]
    if inp.dtype==numpy.float32: dtype = 'float'
    elif inp.dtype==numpy.float64: dtype = 'double'
    elif inp.dtype==numpy.int32: dtype = 'long'
    elif inp.dtype==numpy.int64: dtype = 'long long'
    elif inp.dtype==numpy.uint32: dtype = 'unsigned long'
    elif inp.dtype==numpy.uint64: dtype = 'unsigned long long'
    elif inp.dtype==numpy.int16: dtype = 'short'
    elif inp.dtype==numpy.uint16: dtype = 'unsigned short'
    elif inp.dtype==numpy.int8: dtype = 'char'
    elif inp.dtype==numpy.uint8: dtype = 'unsigned char'
    else: raise NotImplementedError
    
    ret['name'] = name
    ret['type'] = dtype
    ret['input'] = inp
    ret['itype'] = 'PyArrayObject *'
    ret['get'] = 'inline %s %s_get(PyArrayObject * input, int exemplar, int feature) {return *(%s *)(input->data + exemplar*input->strides[0] + feature*input->strides[1]);}' % (ret['type'], name, ret['type'])
    ret['exemplars'] = 'inline int %s_exemplars(PyArrayObject * input) {return input->dimensions[0];}'%name
    ret['features'] = 'inline int %s_features(PyArrayObject * input) {return input->dimensions[1];}'%name

    return ret
  
  def tupleInputC(self):
    return tuple(self.dm)
  
  def key(self):
    return 'MatrixES|' + reduce(lambda a,b: a+':'+b, map(lambda d: str(d.dtype), self.dm))



MatrixFS = MatrixES # For backward compatability.



class MatrixGrow(ExemplarSet):
  """A slightly more advanced version of the basic exemplar set that has better support for incrimental learning, as it allows appends to be more efficient. It still assumes that all of the data can be fitted in memory, and makes use of numpy arrays for internal storage."""
  def __init__(self, *args):
    """Optionally allows you to provide a list of numpy data matrices to by the channels data matrices. Alternativly you can use the add method to add them, one after another, post construction, or use append to start things going. All data matrices must be 2D numpy arrays, with the first dimension, indexing the exemplar, being the same size in all cases. (If there is only 1 exemplar then it will accept 1D arrays.)"""
    
    # Internal storage is as a list, where each entry in the list is a set of exemplars. The exmplars are represented as a further list, indexed by channel, of 2D data matrices.
    if len(args)!=0:
      self.dmcList = [list(map(lambda a: a.reshape((1,-1)) if len(a.shape)==1 else a, args))]
      
      for dm in self.dmcList[0]:
        assert(len(dm.shape)==2)
        assert(dm.shape[0]==self.dm[0].shape[0])
    else:
      self.dmcList = []
  
  def add(self, dm):
    """Adds a new data matrix of information as another channel. Returns its channel index. If given a 1D matrix assumes that there is only one exemplar and adjusts it accordingly."""
    self.make_compact()
    if len(dm.shape)==1: dm = dm.reshape((1,-1))
    assert(len(dm.shape)==2)
    
    if len(dmcList)==0: dmcList.append([])
    
    self.dmcList[0].append(dm)
    assert(dm.shape[0]==self.dmcList[0][0].shape[0])
    
    return len(self.dmcList[0])-1
  
  def append(self, *args):
    """Allows you to add exemplars to the structure, by providing a set of data matrices that align with those contained, which contain the new exemplars. If adding a single new exemplar the arrays can be 1D."""
    args = map(lambda dm: dm if len(dm.shape)!=1 else dm.reshape((1,-1)), args)
    
    for dm in args:
      assert(len(dm.shape)==2)
      assert(dm.shape[0]==args[0].shape[0])

    if len(self.dmcList)!=0:
      assert(len(args)==len(self.dmcList[0]))
      for i, dm in enumerate(args):
        assert(dm.dtype==self.dmcList[0][i].dtype)
        assert(dm.shape[1]==self.dmcList[0][i].shape[1])
    
    self.dmcList.append(args)


  def exemplars(self):
    return sum(map(lambda dmc: dmc[0].shape[0], self.dmcList))
  
  def channels(self):
    return len(self.dmcList[0]) if len(self.dmcList)!=0 else 0
  
  def dtype(self, channel):
    return self.dmcList[0][channel].dtype
  
  def features(self, channel):
    return self.dmcList[0][channel].shape[1]
  
  
  def make_compact(self):
    """Internal method really - converts the data structure so that len(dmcList)==1, by concatenating arrays as needed."""
    if len(self.dmcList)>1:
      rep = []
      
      for i in xrange(len(self.dmcList[0])):
        dml = map(lambda dmc: dmc[i], self.dmcList)
        dm = numpy.concatenate(dml, axis=0)
        rep.append(dm)
      
      self.dmcList = [rep]


  def __getitem__(self, index):
    self.make_compact()
    
    a = numpy.asarray(index[1]).reshape(-1)
    b = numpy.asarray(index[2]).reshape(-1)
    if not isinstance(index[1],numpy.ndarray) or not isinstance(index[2],numpy.ndarray): return self.dmcList[0][index[0]][index[1],index[2]]
    else: return self.dmcList[0][index[0]][numpy.ix_(a,b)]

  def codeC(self, channel, name):
    self.make_compact()
    
    ret = dict()
    
    inp = self.dmcList[0][channel]
    if inp.dtype==numpy.float32: dtype = 'float'
    elif inp.dtype==numpy.float64: dtype = 'double'
    elif inp.dtype==numpy.int32: dtype = 'long'
    elif inp.dtype==numpy.int64: dtype = 'long long'
    elif inp.dtype==numpy.uint32: dtype = 'unsigned long'
    elif inp.dtype==numpy.uint64: dtype = 'unsigned long long'
    elif inp.dtype==numpy.int16: dtype = 'short'
    elif inp.dtype==numpy.uint16: dtype = 'unsigned short'
    elif inp.dtype==numpy.int8: dtype = 'char'
    elif inp.dtype==numpy.uint8: dtype = 'unsigned char'
    else: raise NotImplementedError
    
    ret['name'] = name
    ret['type'] = dtype
    ret['input'] = inp
    ret['itype'] = 'PyArrayObject *'
    ret['get'] = 'inline %s %s_get(PyArrayObject * input, int exemplar, int feature) {return *(%s *)(input->data + exemplar*input->strides[0] + feature*input->strides[1]);}' % (ret['type'], name, ret['type'])
    ret['exemplars'] = 'inline int %s_exemplars(PyArrayObject * input) {return input->dimensions[0];}'%name
    ret['features'] = 'inline int %s_features(PyArrayObject * input) {return input->dimensions[1];}'%name

    return ret

  def tupleInputC(self):
    self.make_compact()
    
    return tuple(self.dmcList[0])

  def key(self):
    self.make_compact()
    return 'MatrixGrow|' + reduce(lambda a,b: a+':'+b, map(lambda d: str(d.dtype), self.dmcList[0]))