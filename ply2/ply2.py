# Copyright (c) 2016, Tom SF Haines
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#  * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import sys
import gzip
import bz2

import numpy

import re
from collections import OrderedDict, defaultdict



def create(binary = False, compress = 0):
  """Creates and returns an 'empty' dictionary to represent a ply 2 file, with reasonable defaults filled in. Takes two parameters: If binary is False (the default) it uses ascii mode, otherwise it uses binary mode, where it matches the mode to the current computer."""
  ret = dict()
  
  ret['format'] = ('binary_little_endian' if sys.byteorder=='little' else 'binary_big_endian') if binary else 'ascii'
  ret['type'] = []
  ret['meta'] = OrderedDict()
  ret['comment'] = dict()
  ret['compress'] = None if compress==0 else ('gzip' if compress==1 else 'bzip2')
  ret['element'] = OrderedDict()
  
  return ret



def verify(data):
  """Given a dictionary that is meant to be encoded as a ply 2 file this verifies its compatible - raises an error if there is a problem. Called by the write function, but provided for if you want to verify seperately."""
  
  # Check root keys are all valid...
  if not set(data.keys()) <= set(['format', 'type', 'meta', 'comment', 'compress', 'element']):
    raise KeyError('Root dictionary includes disallowed keys.')
  
  # Make sure the format is valid...
  if 'format' in data:
    if data['format'] not in ['ascii', 'binary_little_endian', 'binary_big_endian']:
      raise ValueError('Unrecognised format.')
    
  # Check the type is sane...
  if 'type' in data:
    if isinstance(data['type'], basestring):
      raise TypeError('Type must be a list of strings, not a single string.')
      
    for item in data['type']:
      if not isinstance(item, basestring):
        raise TypeError('Type must be a string.')
      
      if len(item.split())!=1 or item.strip()!=item:
        raise ValueError('Type string contains whitespace.')

  # Make sure the meta key/value pairs are all valid...
  if 'meta' in data:
    for key, value in data['meta'].iteritems():
      if not isinstance(key, basestring):
        raise TypeError('Meta name is not a string.')
      
      if len(key.split())!=1 or key.strip()!=key:
        raise KeyError('Name of meta variable contains white space.')
      
      if not (isinstance(value, basestring) or isinstance(value, int) or isinstance(value, float)):
        raise TypeError('Unsuported meta variable type.')
  
  # Check the comments work...
  if 'comment' in data:
    for i in xrange(len(data['comment'])):
      if i not in data['comment']:
        raise KeyError('Comments not indexed with contiguous natural numbers starting at zero')
      if not isinstance(data['comment'][i], basestring):
        raise ValueError('Comment line not an instance of basestring.')
      if '\n' in data['comment'][i]:
        raise ValueError('Comment line contains new line.')
  
  # Make the compresssion mode is valid...
  if 'compress' in data:
    if data['compress'] not in [None, '', 'gzip', 'bzip2']:
      raise ValueError('Unrecognised format.')
  
  # Loop and check all elements, including all details...
  if 'element' in data:
    for key, value in data['element'].iteritems():
      if not isinstance(key, basestring):
        raise TypeError('Element name must be a string.')
      
      if len(key.split())!=1 or key.strip()!=key:
        raise KeyError('Name of element contains white space.')
      
      shape = None
      for prop, arr in value.iteritems():
        if not isinstance(prop, basestring):
          raise TypeError('Property name must be a string.')
        
        if len(prop.split())!=1 or prop.strip()!=prop:
          raise KeyError('Name of property contains white space.')
        
        if not isinstance(arr, numpy.ndarray):
          raise TypeError('Element data must be represented as a ndarray.')
        
        if shape==None:
          shape = arr.shape
        else:
          if shape!=arr.shape:
            raise RuntimeError('Shapes of all properties in an element must match')
        
        if issubclass(arr.dtype.type, numpy.signedinteger):
          if arr.dtype.itemsize not in [1, 2, 4, 8, 16]:
            raise TypeError('Element array has signed integer element with unsuported size.')
          
        elif issubclass(arr.dtype.type, numpy.unsignedinteger):
          if arr.dtype.itemsize not in [1, 2, 4, 8, 16]:
            raise TypeError('Element array has unsigned integer element with unsuported size.')
          
        elif issubclass(arr.dtype.type, numpy.floating):
          if arr.dtype.itemsize not in [2, 4, 8, 16]:
            raise TypeError('Element array has float element with unsuported size.')
          
        elif arr.dtype==numpy.object:
          base = None # None for unknown type, True for string, instance of ndarray for array mode.
          for item in arr.flat:
            if base is None:
              base = True if isinstance(item, basestring) else item
            
            elif base is True:
              if not isinstance(item, basestring):
                raise TypeError('All entrys in an element array of strings must be a string.')
            
            else:
              if base.dtype!=item.dtype or len(base.shape)!=len(item.shape):
                raise TypeError('All entrys in an array of arrays must have the same type and same number of dimensions.')
          
          if isinstance(base, numpy.ndarray):
            if (not issubclass(base.dtype.type, numpy.signedinteger)) and (not issubclass(base.dtype.type, numpy.unsignedinteger)) and (not issubclass(base.dtype.type, numpy.floating)):
              raise NotImplementedError('Unsupported array data type.')
          
        else:
          raise TypeError('Element array has unsupported type.')



def encoding_to_dtype(enc, force_int = False):
  """Given an encoding from a ply 2 file this converts it into a dtype that numpy recognises. For internal use. Returns (dtype, number of dimensions if array or None, dtype of dimensions for arrays and strings or None, stored type if array or None)"""
  
  parts = enc.split(':')
  
  if parts[0]=='int8': return (numpy.dtype(numpy.int8), None, None, None)
  if parts[0]=='int16': return (numpy.dtype(numpy.int16), None, None, None)
  if parts[0]=='int32': return (numpy.dtype(numpy.int32), None, None, None)
  if parts[0]=='int64': return (numpy.dtype(numpy.int64), None, None, None)
  if parts[0]=='int128': raise NotImplementedError('int128 is not supported by this implimentation.')
  
  if parts[0]=='nat8': return (numpy.dtype(numpy.uint8), None, None, None)
  if parts[0]=='nat16': return (numpy.dtype(numpy.uint16), None, None, None)
  if parts[0]=='nat32': return (numpy.dtype(numpy.uint32), None, None, None)
  if parts[0]=='nat64': return (numpy.dtype(numpy.uint64), None, None, None)
  if parts[0]=='nat128': raise NotImplementedError('nat128 is not supported by this implimentation.')

  if force_int:
    raise IOError('Unrecognised or unsupported encoding in ply 2 file (for array/string shape type).')
  
  if parts[0]=='real16': return (numpy.dtype(numpy.float16), None, None, None)
  if parts[0]=='real32': return (numpy.dtype(numpy.float32), None, None, None)
  if parts[0]=='real64': return (numpy.dtype(numpy.float64), None, None, None)
  if parts[0]=='real128': return (numpy.dtype(numpy.float128), None, None, None)
  
  if parts[0]=='array':
    if len(parts)!=4:
      raise KeyError('Incomplete array specification')
    
    dims = int(parts[1])
    if dims<1:
      raise ValueError('Array requires a positive dimension count.')
    
    return (numpy.object, dims, encoding_to_dtype(parts[2],True)[0], encoding_to_dtype(parts[3])[0])
  
  if parts[0]=='string':
    return (numpy.object, None, encoding_to_dtype(parts[1],True)[0], None)
  
  raise IOError('Unrecognised encoding in ply 2 file.')



def array_to_encoding(arr):
  """Given a numpy array this returns a suitable ply 2 type for the property it represents. Assumes that the array is encodable (has passed verify), and will return an answer in some error situations. For internal use."""
  if issubclass(arr.dtype.type, numpy.signedinteger):
    if arr.dtype.itemsize==1: return 'int8'
    if arr.dtype.itemsize==2: return 'int16'
    if arr.dtype.itemsize==4: return 'int32'
    if arr.dtype.itemsize==8: return 'int64'
    
  if issubclass(arr.dtype.type, numpy.unsignedinteger):
    if arr.dtype.itemsize==1: return 'nat8'
    if arr.dtype.itemsize==2: return 'nat16'
    if arr.dtype.itemsize==4: return 'nat32'
    if arr.dtype.itemsize==8: return 'nat64'
  
  if issubclass(arr.dtype.type, numpy.floating):
    if arr.dtype.itemsize==2: return 'real16'
    if arr.dtype.itemsize==4: return 'real32'
    if arr.dtype.itemsize==8: return 'real64'
    if arr.dtype.itemsize==16: return 'real128'
    
  if arr.dtype==numpy.object:
    if isinstance(arr.flat[0], numpy.ndarray):
      return 'array:%i:nat32:%s' % (len(arr.flat[0].shape), array_to_encoding(arr.flat[0]))
    else:
      return 'string:nat32'
  
  raise IOError('Failed to represent numpy type as type suitable for a ply 2 file.')



def to_meta_line(key, value):
  """Given a key and value from a dictionary of meta data this returns the requisite meta line for a ply 2 file header. For internal use only."""
  key = key.encode('utf8')
  
  if isinstance(value, basestring):
    value = value.encode('utf8')
    return 'meta string:nat32 %s %i %s\n' % (key, len(value), value)
  
  if isinstance(value, int):
    return 'meta int64 %s %i\n' % (key, value)
  
  if isinstance(value, float):
    return 'meta real64 %s %.16g\n' % (key, value)
  
  raise IOError('Unsupported type to record as meta value.')



def read_meta_line(line):
  """Given a meta line returns the tuple (key, value), or throws an error. Assumes it has already been confirmed to start with meta<white space>."""
  
  # Split - max split means the length and string will not be seperated for a string type, as that requires special care...
  parts = line.split(None, 3)
  if len(parts)!=4:
    raise KeyError('Meta is missing information')
  
  if parts[1] in ['int8', 'int16', 'int32', 'int64', 'int128', 'nat8', 'nat16', 'nat32', 'nat64', 'nat128']:
    value = int(parts[3])
  
  elif parts[1] in ['real16', 'real32', 'real64', 'real128']:
    value = float(parts[3])
  
  elif parts[1][:7]=='string:':
    if parts[1][7:] not in ['int8', 'int16', 'int32', 'int64', 'int128', 'nat8', 'nat16', 'nat32', 'nat64', 'nat128']:
      raise IOError('Unrecognised string length encoding in meta.')

    base = parts[3].index(' ')
    value = parts[3][base+1:]
    
    if len(value)!=int(parts[3][:base]):
      raise IOError('String length does not match specified.')
  
  else:
    raise IOError('Unrecognised or unsuported meta value encoding.')

  return parts[2], value



def to_element_line(key, value):
  """Given an item in the element dictionary, which represents an element, this returns the element line for the header."""
  shape = (0, )
  for prop, arr in value.iteritems():
    shape = arr.shape
    break
  
  shape = ' '.join([str(x) for x in shape])
  return 'element %s %s\n' % (key, shape)



class BZ2Comp:
  """Version provided by the Python library does not write to an already open file - this version does."""
  
  def __init__(self, f):
    self.f = f
    self.comp = bz2.BZ2Compressor()
  
  
  def write(self, data):
    self.f.write(self.comp.compress(data))
  
  
  def flush(self):
    self.f.write(self.comp.flush())



class BZ2Decomp:
  """Version provided by the Python library does not read from an already open file - this version does."""
  
  def __init__(self, f):
    self.f = f
    self.decomp = bz2.BZ2Decompressor()
    
    self.chunk_size = 1024 * 1024
    
    self.spare = []
    self.spare_total = 0
  
  
  def read(self, size = -1):
    while (size < 1) or (self.spare_total < size):
      try:
        data = self.f.read(self.chunk_size)
        data = self.decomp.decompress(data)
        self.spare.append(data)
        self.spare_total += len(data)
      except EOFError:
        # Should probably do something with unused_data, but as I don't know what self.f is I am not sure what.
        break
    
    if size>0:
      ret = ''.join(self.spare)
      self.spare = [ret[size:]]
      self.spare_total = len(self.spare[0])
      return ret[:size]
    
    else:
      ret = ''.join(self.spare)
      self.spare = []
      self.spare_total = 0
      return ret
  
  
  def readline(self):
    while len(self.spare)==0 or ('\n' not in self.spare[-1]):
      try:
        data = self.f.read(self.chunk_size)
        data = self.decomp.decompress(data)
        self.spare.append(data)
        self.spare_total += len(data)
      except EOFError:
        # Should probably do something with unused_data, but as I don't know what self.f is I am not sure what.
        break
    
    ret = ''.join(self.spare)
    new_line = ret.find('\n')
    self.spare = [ret[new_line+1:]]
    return ret[:new_line+1]
   
   
  def flush(self):
    pass



def ascii_array(arr):
  """Given a numpy array this returns it represented as an ascii array."""
  shape = ' '.join([str(x) for x in arr.shape])
  data = ' '.join([str(x) for x in arr.flat])
  return '%s %s' % (shape, data)



def write_ascii(f, element, order):
  """Write the given element, with the elements in the given order. f is the file to write to, element the entry in the dictionary (dictionary containing propertioes as numpy arrays) and order the order to write the elements (list of property names)."""
  
  # Convert each property into an array of strings...
  parts = []
  for prop in order:
    arr = element[prop]
    
    if arr.dtype==numpy.object:
      if isinstance(arr.flat[0], numpy.ndarray): # Array
        parts.append([ascii_array(x) for x in arr.flat])
      
      else: # String
        parts.append(['%i %s'%(len(x.encode('utf8')),x.encode('utf8')) for x in arr.flat])
      
    elif arr.dtype==numpy.float16:
      parts.append(['%.4g' % x for x in arr.flat])
      
    elif arr.dtype==numpy.float32:
      parts.append(['%.8g' % x for x in arr.flat])
    
    elif arr.dtype==numpy.float64:
      parts.append(['%.16g' % x for x in arr.flat])
    
    elif arr.dtype==numpy.float128:
      parts.append(['%.35g' % x for x in arr.flat])
    
    else: # All the integer types.
      parts.append([str(x) for x in arr.flat])
  
  
  # Zip them and write out, line by line...
  for line in zip(*parts):
    f.write(' '.join(line) + '\n')



def write_binary(f, element, order, little=True):
  """Identical to write ascii, but for the binary cases, with an extra flag to indicate endiness."""
  
  # Needed below...
  if little:
    order_accept = '<=' if sys.byteorder=='little' else '<'
    
  else:
    order_accept = '>=' if sys.byteorder=='big' else '>'
  
  # Convert each property into a list of binary blobs...
  parts = []
  for prop in order:
    arr = element[prop]
    
    if arr.dtype in [numpy.int8, numpy.int16, numpy.int32, numpy.int64, numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64]:
      # Handle byte order...
      if arr.dtype.byteorder not in order_accept:
        arr = arr.byteswap()
      
      # Convert to bytes...
      data = arr.tostring()
      
      # Chop up into a list ready for interleaving, and record...
      size = arr.dtype.itemsize
      parts.append([data[i:i+size] for i in xrange(0, len(data), size)])
    
    elif arr.dtype in [numpy.float16, numpy.float32, numpy.float64, numpy.float128]:
      # Convert to bytes...
      data = arr.tostring()
      
      # Chop up into a list ready for interleaving, and record...
      size = arr.dtype.itemsize
      parts.append([data[i:i+size] for i in xrange(0, len(data), size)])
      
    elif arr.dtype==numpy.object:
      # Arrays or strings...
      if isinstance(arr.flat[0], numpy.ndarray):
        def prepare(a):
          shape = numpy.array(a.shape, dtype=numpy.uint32)
          
          if shape.dtype.byteorder not in order_accept:
            shape = shape.byteswap()
          
          if a.dtype.byteorder not in order_accept and a.dtype in [numpy.int8, numpy.int16, numpy.int32, numpy.int64, numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64]:
            a = a.byteswap()
          
          return shape.tostring() + a.tostring()
        
        parts.append([prepare(a) for a in arr.flat])

      else:
        def prepare(s):
          s = s.encode('utf8')
          
          size = numpy.array([len(s)], dtype=numpy.uint32)
          if size.dtype.byteorder not in order_accept:
            size = size.byteswap()
          
          return size.tostring() + s
      
        parts.append([prepare(s) for s in arr.flat])
      
    else:
      raise RuntimeError('Bug: Unsupported array managed to sneak through safety checks.')
  
  
  # Zip and write out...
  for line in zip(*parts):
    f.write(''.join(line))



def write(f, data):
  """Given a dictionary in the required format (second parameter, see readme.txt), this writes it to the file (first variable), where file can either be the filename of a file to open or a file-like object to .write() all of the data to. Note that if a file is passed in it must have been openned in binary mode, even if using the ascii format."""
  
  # Verify the passed data...
  verify(data)
  
  # If we have been passed a string open the file...
  if isinstance(f, basestring):
    f = open(f, 'wb')
    do_close = True
  else:
    do_close = False
  
  
  # Write the header...
  f.write('ply\n'.encode('utf8'))
  
  if 'format' in data:
    format = data['format']
    f.write(('format %s 2.0\n' % data['format']).encode('utf8'))
  else:
    format  = 'ascii'
    f.write('format ascii 2.0\n'.encode('utf8'))
  
  if 'type' in data and len(data['type'])>0:
    f.write(('type %s\n' % ' '.join(data['type'])).encode('utf8'))
  
  if 'meta' in data:
    for key, value in data['meta'].iteritems():
      f.write(to_meta_line(key, value).encode('utf8'))
  
  if 'comment' in data:
    for i in xrange(len(data['comment'])):
      f.write(('comment %s\n' % data['comment'][i]).encode('utf8'))
  
  compress = None
  if 'compress' in data:
    if data['compress']=='gzip':
      compress = 'gzip'
      f.write('compress gzip\n'.encode('utf8'))
    
    if data['compress']=='bzip2':
      compress = 'bzip2'
      f.write('compress bzip2\n'.encode('utf8'))
  
  element_order = []
  property_order = dict()
  if 'element' in data:
    for key, value in data['element'].iteritems():
      element_order.append(key)
      property_order[key] = []
      
      f.write(to_element_line(key, value).encode('utf8'))

      for prop, arr in value.iteritems():
        f.write(('property %s %s\n' % (array_to_encoding(arr), prop)).encode('utf8'))
        property_order[key].append(prop)
  
  f.write('end_header\n'.encode('utf8'))
  
  
  # Prepare for compression if required...
  ff = f
  
  if compress=='gzip':
    ff = gzip.GzipFile(fileobj = f, mode = 'wb')
  
  if compress=='bzip2':
    ff = BZ2Comp(f)
  
  
  # Loop and write each element in turn, using the correct writting code...
  for elem in element_order:
    if format=='ascii':
      write_ascii(ff, data['element'][elem], property_order[elem])
    
    elif format=='binary_little_endian':
      write_binary(ff, data['element'][elem], property_order[elem], True)
    
    else: # binary_big_endian
      write_binary(ff, data['element'][elem], property_order[elem], False)

  
  # If we were compressing then we better flush the buffer...
  if compress=='gzip':
    ff.close()
  
  if compress=='bzip2':
    ff.flush()
  
  
  # If we openned the file we better close it...
  if do_close:
    f.close()



# Regular expression for doing a 'split' without throwing away white space. Relies on the fact the Python re module is greedy, and tries to make each match as long as possible...
ws_keep_split = re.compile(r'(\s*[^\s]*)')



# Functor used below...
class ReadAsciiArray:
  def __init__(self, next_token, dims, store_dtype, cast):
    self.next_token = next_token
    self.dims = dims
    self.store_dtype = store_dtype
    self.cast = cast
  
  def __call__(self):
    shape = []
    for _ in xrange(self.dims):
      shape.append(int(self.next_token()))

    ret = numpy.empty(shape, dtype=self.store_dtype)

    for index in numpy.ndindex(*shape):
      ret[index] = self.cast(self.next_token())
    
    return ret



def read_ascii(f, element, prop):
  """Given a file object (f) and an element, as an ordered dictionary of arrays ready to be filled with the data, plus the results of encoding_to_dtype for each property in a dictionary indexed by property (prop). Fills in the arrays, reading the neccessary data from f."""
  
  # Functions for token extraction. Because strings contain whitespace it keeps it at the start of each token, otherwise it would be lost. Also has to do some nastiness to preserve whitespace at the end of a line by adding it to the first token on the next, which could of course also be entirly whitespace...
  
  tokens = [] # Subtle invariant used below: All entries must contain some non-whitespace except for the last entry, which may be whitespace only.
  
  def next_token():
    while len(tokens)==0 or tokens[0].isspace():
      data = f.readline()
      if len(data)==0:
        raise IOError('Insufficient data')
      more = ws_keep_split.findall(data)[:-1]
      if len(tokens)!=0:
        more[0] = tokens.pop(0) + more[0]
      tokens.extend(more)
    return tokens.pop(0)
  
  
  # Reading functions used by below...
  def read_int():
    return int(next_token())
  
  
  def read_float():
    return float(next_token())
  
  
  def read_str():
    length = int(next_token()) + 1 # +1 so below code gets the space as well.
    ret = ''
    while len(ret)<length:
      ret += next_token()
    
    if len(ret)>length:
      excess = ret[length:]
      ret = ret[:length]
      tokens.insert(0, excess)
    
    return ret[1:].decode('utf8') # [1:] to skip space at start.

  
  # To keep the reading loop sane encode it as a list of tuples, where each tuple is an array to output to followed by a (token eatting) function to call to get the data to be written...
  shape = None
  ops = []
  for name, array in element.iteritems():
    
    if shape==None:
      shape = array.shape
    
    if array.dtype in [numpy.int8, numpy.int16, numpy.int32, numpy.int64, numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64]:
      ops.append((array, read_int))
    
    elif array.dtype in [numpy.float16, numpy.float32, numpy.float64, numpy.float128]:
      ops.append((array, read_float))
    
    elif array.dtype==numpy.object:
      # String or array...
      arr_dtype, dims, shape_dtype, store_dtype = prop[name]
      
      if dims==None:
        # String...
        ops.append((array, read_str))
        
      else:
        # Array...
        if store_dtype in [numpy.int8, numpy.int16, numpy.int32, numpy.int64, numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64]:
          ops.append((array, ReadAsciiArray(next_token, dims, store_dtype, int)))
        
        elif store_dtype in [numpy.float16, numpy.float32, numpy.float64, numpy.float128]:
          ops.append((array, ReadAsciiArray(next_token, dims, store_dtype, float)))
        
        else:
          raise RuntimeError('Trying to read an unsupported array of arrays contents type.')
    
    else:
      raise RuntimeError('Trying to read into an unsupported array type; this is a bug.')
  
  
  # If shape==None there are no properties - its a no-op...
  if shape==None:
    return
  
  
  # Loop all coordinates within the shape and read in the properties for each - after all the above prep this is actually rather elegant...
  for index in numpy.ndindex(*shape):
    for array, func in ops:
      array[index] = func()



# Functors used by the read_binary function...
class ReadBinaryInt:
  def __init__(self, f, dtype, byteswap):
    self.f = f
    self.dtype = dtype
    self.byteswap = byteswap
  
  def __call__(self):
    data = self.f.read(self.dtype.itemsize)
    arr = numpy.fromstring(data, self.dtype)
    if self.byteswap:
      arr = arr.byteswap()
    return arr[0]



class ReadBinaryFloat:
  def __init__(self, f, dtype):
    self.f = f
    self.dtype = dtype
  
  def __call__(self):
    return numpy.fromstring(self.f.read(self.dtype.itemsize), self.dtype)[0]



class ReadBinaryString:
  def __init__(self, f, shape_dtype, byteswap):
    self.f = f
    self.shape_dtype = shape_dtype
    self.byteswap = byteswap
    
  def __call__(self):
    shape_data = self.f.read(self.shape_dtype.itemsize)
    length = numpy.fromstring(shape_data, self.shape_dtype)
    if self.byteswap: length = length.byteswap()
    
    data = self.f.read(length[0])
    if len(data)!=length[0]:
      raise EOFError('Ran out data reading string.')
    
    return data.decode('utf8')



class ReadBinaryArray:
  def __init__(self, f, dims, shape_dtype, store_dtype, byteswap):
    self.f = f
    self.dims = dims
    self.shape_dtype = shape_dtype
    self.store_dtype = store_dtype
    self.byteswap = byteswap
    self.byteswap_data = byteswap and self.store_dtype in [numpy.int8, numpy.int16, numpy.int32, numpy.int64, numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64]
    
  def __call__(self):
    shape_data = self.f.read(self.shape_dtype.itemsize * self.dims)
    shape = numpy.fromstring(shape_data, self.shape_dtype)
    if self.byteswap:
      shape = shape.byteswap()
    
    len_data = int(numpy.prod(shape)) * self.store_dtype.itemsize
    data = self.f.read(len_data)
    if len(data)!=len_data:
      raise EOFError('Ran out data reading array.')
    
    ret = numpy.fromstring(data, self.store_dtype).reshape(shape)
    if self.byteswap_data:
      ret = ret.byteswap()
      
    return ret



def read_binary(f, element, prop, little = True):
  """Same interface and purpose as read_ascii, except binary data and with extra flag with True to indicate little endian, False to indicate big endian."""
  
  # Prep for endian-ness...
  byteswap = little != (sys.byteorder=='little') # Sorry


  # To keep the reading loop sane encode it as a list of tuples, where each tuple is an array to output to followed by a (data eatting) functor to call to get the data to be written...
  shape = None
  ops = []
  for name, array in element.iteritems():
    
    if shape==None:
      shape = array.shape

    if array.dtype in [numpy.int8, numpy.int16, numpy.int32, numpy.int64, numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64]:
      ops.append((array, ReadBinaryInt(f, array.dtype, byteswap)))
    
    elif array.dtype in [numpy.float16, numpy.float32, numpy.float64, numpy.float128]:
      ops.append((array, ReadBinaryFloat(f, array.dtype)))
    
    elif array.dtype==numpy.object:
      arr_dtype, dims, shape_dtype, store_dtype = prop[name]
      
      if dims==None:
        # String...
        ops.append((array, ReadBinaryString(f, shape_dtype, byteswap)))
      
      else:
        # Array...
        ops.append((array, ReadBinaryArray(f, dims, shape_dtype, store_dtype, byteswap)))
    
    else:
      raise RuntimeError('Trying to read into an unsupported array type; this is a bug.')
  
  
  # If shape==None there are no properties - its a no-op...
  if shape==None:
    return
  
  
  # Loop all coordinates within the shape and read in the properties for each - after all the above prep this is actually rather elegant...
  for index in numpy.ndindex(*shape):
    for array, func in ops:
      array[index] = func()



def read_header(f):
  """Simply reads the header, returning a list of lines and leaving the files cursor immediatly where the data starts."""
  header = []
  
  while True:
    line = f.readline()
    if len(line)==0:
      raise EOFError('Ran out of data reading header.')
    
    header.append(line[:-1])
    
    # Stop if done...
    if len(header)!=0 and header[-1]=='end_header':
      break
    
    if len(header) > 16384:
      raise BufferError('Limit on header line count exceded.')
  
  return header



def read(f):
  """This reads a ply2 file (first variable), where file can either be the filename of a file to open or a file-like object to .read()/.readline() all of the data from. Note that if a file is passed in it must have been opened in binary mode, even if using the ascii format. It tries to leave the cursor at the end of the ply2 file, and will when compression is off, but may not otherwise. Returns the dictionary representing the file."""
  
  # If we have been passed a string open the file...
  if isinstance(f, basestring):
    f = open(f, 'rb')
    do_close = True
  else:
    do_close = False
  
  
  # Read the header...
  header = read_header(f)
  
  
  # Parse the header, and build an 'empty' dictionary representing the file from it, making use of ordered dictionaries so the actual reading code knows the element/property orders...
  data = dict()
  data['meta'] = OrderedDict()
  data['comment'] = dict()
  data['element'] = OrderedDict()
  
  elem_shape = dict() # Shape of element, indexed [element] to get a tuple.
  elem_prop = defaultdict(dict) # Indexed [element][property] contains the full output tuple of encoding_to_dtype.
  
  compress = None

  if header[0]!='ply':
    raise AssertionError('Wrong magic for a ply file.')
  
  if header[1]=='format ascii 2.0':
    data['format'] = 'ascii'
  elif header[1]=='format binary_little_endian 2.0':
    data['format'] = 'binary_little_endian'
  elif header[1]=='format binary_big_endian 2.0':
    data['format'] = 'binary_big_endian'
  else:
    raise AssertionError('Second line of ply does not indicate type.')
  
  if header[-1]!='end_header':
    raise AssertionError('Incorrect end of header (should not happen - bug has survived).')
  
  for line in header[2:-1]:
    part = line.split()
    
    if part[0]=='type':
      if 'type' in data:
        raise AssertionError('Multiple type specifications in header.')
      data['type'] = part[1:]
    
    elif part[0]=='meta':
      key, value = read_meta_line(line)
      data['meta'][key] = value
    
    elif part[0]=='comment':
      data['comment'][len(data['comment'])] = line[7:].lstrip()
    
    elif part[0]=='compress':
      if 'compress' in data:
        raise AssertionError('Multiple compression specifications.')
      
      if len(part)!=2:
        raise AssertionError('Header compression specification is wrong.')
      
      if part[1]=='gzip':
        data['compress'] = 'gzip'
        compress = 'gzip'
      elif part[1]=='bzip2':
        data['compress'] = 'bzip2'
        compress = 'bzip2'
      else:
        raise AssertionError('Unrecognised compression mode.')
    
    elif part[0]=='length': # Ignored - just check its at least the right structure.
      if len(part)!=2:
        raise AssertionError('Header file length specification is wrong.')
      
      length = int(part[1]) # For the exception thrown!
    
    elif part[0]=='element':
      name = part[1]
      shape = [int (p) for p in part[2:]]
      
      if name in data['element']:
        raise AssertionError('Duplicate element.')
      
      if len(shape)==0:
        raise AssertionError('Element has no shape.')
      
      for v in shape:
        if v<0:
          raise AssertionError('Shape has negative volume.')
      
      data['element'][name] = OrderedDict()
      elem_shape[name] = tuple(shape)
    
    elif part[0]=='property':
      if len(data['element'])==0:
        raise AssertionError('Property defined before element declared in header.')
      
      elem_name = next(reversed(data['element']))
      elem = data['element'][elem_name]
      shape = elem_shape[elem_name]
      
      arr_dtype, dims, shape_dtype, store_dtype = encoding_to_dtype(part[1])
      prop_name = part[2]
      
      if prop_name in elem:
        raise AssertionError('Duplicate property name within element.')
      
      elem_prop[elem_name][prop_name] = (arr_dtype, dims, shape_dtype, store_dtype)
      elem[prop_name] = numpy.zeros(shape, dtype=arr_dtype)


  # Setup decompression if required...
  ff = f
  
  if compress=='gzip':
    ff = gzip.GzipFile(fileobj = f, mode = 'rb')
  
  if compress=='bzip2':
    ff = BZ2Decomp(f)
  
  
  # Loop and read in each element in turn...
  for elem_name in data['element'].iterkeys():
    if data['format']=='ascii':
      read_ascii(ff, data['element'][elem_name], elem_prop[elem_name])
    
    elif data['format']=='binary_little_endian':
      read_binary(ff, data['element'][elem_name], elem_prop[elem_name], True)
    
    else: # binary_big_endian
      read_binary(ff, data['element'][elem_name], elem_prop[elem_name], False)
  
  
  # If decompressing then clean that up...
  if compress=='gzip':
    ff.close()
  
  if compress=='bzip2':
    ff.flush()
  
  
  # If we openned the file we better close it...
  if do_close:
    f.close()
  
  
  # Return shiny new dictionary of data...
  return data
