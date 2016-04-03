# Copyright (c) 2016, Tom SF Haines
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#  * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import sys
import numpy



def create(binary = False, compress = 0):
  """Creates and returns an 'empty' dictionary to represent a ply 2 file, with reasonable defaults filled in. Takes two parameters: If binary is False (the default) it uses ascii mode, otherwise it uses binary mode, where it matches the mode to the current computer."""
  ret = dict()
  
  ret['format'] = ('binary_little_endian' if sys.byteorder=='little' else 'binary_big_endian') if binary else 'ascii'
  ret['meta'] = dict()
  ret['comment'] = dict()
  ret['compress'] = None if compress==0 else ('gzip' if compress==1 else 'bzip2')
  ret['element'] = dict()
  
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
          
        else:
          raise TypeError('Element array has unsupported type.')



def encoding_to_dtype(enc, store_extra = None):
  """Given an encoding from a ply 2 file this converts it into a dtype that numpy recognises. For internal use."""
  
  parts = enc.split(':')
  if store_extra!=None and len(parts)>1:
    store_extra += parts[1:]
  
  if parts[0]=='int8': return numpy.int8
  if parts[0]=='int16': return numpy.int16
  if parts[0]=='int32': return numpy.int32
  if parts[0]=='int64': return numpy.int64
  if parts[0]=='int128': raise IOError('int128 is not supported by this implimentation.')
  
  if parts[0]=='nat8': return numpy.uint8
  if parts[0]=='nat16': return numpy.uint16
  if parts[0]=='nat32': return numpy.uint32
  if parts[0]=='nat64': return numpy.uint64
  if parts[0]=='nat128': raise IOError('nat128 is not supported by this implimentation.')
  
  if parts[0]=='real16': return numpy.float16
  if parts[0]=='real32': return numpy.float32
  if parts[0]=='real64': return numpy.float64
  if parts[0]=='real128': return numpy.float128
  
  if parts[0]=='array': return numpy.object
  if parts[0]=='string': return numpy.object
  
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
