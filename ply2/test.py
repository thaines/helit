#! /usr/bin/env python

# Copyright (c) 2016, Tom SF Haines
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#  * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import tempfile
import unittest

import numpy

import ply2



class TestPly2(unittest.TestCase):
  """Checks that all functions in the ply2 work, or are at least consistaent with other."""
  
  def test_create(self):
    for binary in [False, True]:
      for compress in [0, 1, 2]:
        data = ply2.create(binary, compress)
        ply2.verify(data)
  
  
  def test_verify_keys(self):
    data = dict()
    ply2.verify(data)
    
    data['format'] = 'ascii'
    ply2.verify(data)
    
    data['meta'] = dict()
    ply2.verify(data)
    
    data['penguin'] = 'flying'
    with self.assertRaises(KeyError):
      ply2.verify(data)
  
  
  def test_verify_format(self):
    data = dict()
    data['format'] = 'ascii'
    ply2.verify(data)
    
    data['format'] = 'binary_little_endian'
    ply2.verify(data)
    
    data['format'] = 'carrier pigeon'
    with self.assertRaises(ValueError):
      ply2.verify(data)

  
  def test_verify_type(self):
    data = dict()
    
    data['type'] = 'image'
    with self.assertRaises(TypeError):
      ply2.verify(data)
    
    data['type'] = []
    ply2.verify(data)
    
    data['type'].append('image')
    ply2.verify(data)
    
    data['type'].append('cabbage\n')
    with self.assertRaises(ValueError):
      ply2.verify(data)
    
    data['type'] = ['Its dead jim']
    with self.assertRaises(ValueError):
      ply2.verify(data)
  
  
  def test_verify_meta(self):
    data = dict()
    data['meta'] = dict()
    
    data['meta']['author'] = 'Cthulhu'
    ply2.verify(data)
    
    data['meta']['scaryness'] = 5
    ply2.verify(data)
    
    data['meta']['age'] = 4.561e67
    ply2.verify(data)
    
    data['meta'][42] = 42
    with self.assertRaises(TypeError):
      ply2.verify(data)
    del data['meta'][42]
    
    data['meta']['dance moves'] = 3
    with self.assertRaises(KeyError):
      ply2.verify(data)
    del data['meta']['dance moves']
    
    data['meta']['awesomness_distribution'] = (5.3, 6.2)
    with self.assertRaises(TypeError):
      ply2.verify(data)
    del data['meta']['awesomness_distribution']
    
    ply2.verify(data) # Just to make sure that none of the above are passing because an earlier failure wasn't succesfully removed.
  
  
  def test_verify_comment(self):
    data = dict()
    
    data['comment'] = 'I am a cabbage'
    with self.assertRaises(TypeError):
      ply2.verify(data)
    
    data['comment'] = dict()
    ply2.verify(data)
    
    data['comment'][0] = 'Bring out yer dead!'
    ply2.verify(data)
    
    data['comment'][2] = 'Bring out yer dead!'
    with self.assertRaises(KeyError):
      ply2.verify(data)
    
    data['comment'][1] = 'Bring out yer dead!'
    ply2.verify(data)
    
    data['comment'][3] = 'Bring out yer dead!\n'
    with self.assertRaises(ValueError):
      ply2.verify(data)
  
  
  def test_verify_compress(self):
    data = dict()
    
    data['compress'] = ''
    ply2.verify(data)
    
    data['compress'] = None
    ply2.verify(data)
    
    data['compress'] = 'gzip'
    ply2.verify(data)
    
    data['compress'] = 'bzip2'
    ply2.verify(data)
    
    data['compress'] = 'elephant'
    with self.assertRaises(ValueError):
      ply2.verify(data)
  
  
  def test_verify_element(self):
    data = dict()
    
    data['element'] = dict()
    data['element']['pixel'] = dict()
    ply2.verify(data)
    
    data['element']['pixel']['red'] = numpy.zeros((8, 8), dtype=numpy.uint8)
    ply2.verify(data)
    
    data['element']['pixel']['green'] = 'green'
    with self.assertRaises(TypeError):
      ply2.verify(data)
    
    data['element']['pixel']['green'] = numpy.zeros((8, 8), dtype=numpy.complex)
    with self.assertRaises(TypeError):
      ply2.verify(data)
    
    data['element']['pixel']['green'] = numpy.zeros((4, 8), dtype=numpy.uint8)
    with self.assertRaises(RuntimeError):
      ply2.verify(data)
    
    data['element']['pixel']['green'] = numpy.zeros((8, 8), dtype=numpy.uint8)
    ply2.verify(data)
    
    data['element'][None] = dict()
    with self.assertRaises(TypeError):
      ply2.verify(data)
    del data['element'][None]
    
    data['element']['pet alien'] = dict()
    with self.assertRaises(KeyError):
      ply2.verify(data)
    del data['element']['pet alien']
    
    data['element']['pixel']['burnt amber'] = numpy.ones((8, 8), dtype=numpy.uint8)
    with self.assertRaises(KeyError):
      ply2.verify(data)
    del data['element']['pixel']['burnt amber']
    
    data['element']['pixel']['ochre\n'] = numpy.ones((8, 8), dtype=numpy.uint8)
    with self.assertRaises(KeyError):
      ply2.verify(data)
    del data['element']['pixel']['ochre\n']
    
    data['element']['pixel']['depth'] = numpy.ones((8, 8), dtype=numpy.float32)
    data['element']['pixel']['index'] = numpy.ones((8, 8), dtype=numpy.int32)
    ply2.verify(data)
  
  
  def test_verify_element_advanced(self):
    data = dict()
    data['element'] = dict()
    
    data['element']['people'] = dict()
    data['element']['people']['name'] = numpy.zeros(3, dtype=numpy.object)
    with self.assertRaises(AttributeError):
      ply2.verify(data)
    
    data['element']['people']['name'][0] = 'The Joker'
    data['element']['people']['name'][1] = 6
    data['element']['people']['name'][2] = u'Poison Ivy'
    with self.assertRaises(TypeError):
      ply2.verify(data)
    
    data['element']['people']['name'][1] = 'Bane'
    ply2.verify(data)
    
    data['element']['people']['image'] = numpy.zeros(3, dtype=numpy.object)
    data['element']['people']['image'][0] = numpy.random.rand(96, 64).astype(numpy.float32)
    data['element']['people']['image'][1] = numpy.random.rand(96, 64).astype(numpy.float32)
    data['element']['people']['image'][2] = numpy.random.rand(96, 64).astype(numpy.float32)
    ply2.verify(data)
    
    data['element']['people']['image'][1] = numpy.random.rand(64, 64).astype(numpy.float32)
    ply2.verify(data)
    
    data['element']['people']['image'][1] = numpy.random.rand(12).astype(numpy.float32)
    with self.assertRaises(TypeError):
      ply2.verify(data)
    
    data['element']['people']['image'][1] = numpy.random.rand(48, 32).astype(numpy.float64)
    with self.assertRaises(TypeError):
      ply2.verify(data)
    
    data['element']['people']['image'][1] = 'Insert image here'
    with self.assertRaises(AttributeError):
      ply2.verify(data)
  
  
  def test_encoding_to_dtype(self):
    self.assertTrue(ply2.encoding_to_dtype('nat8')==numpy.uint8)
    self.assertTrue(ply2.encoding_to_dtype('nat16')==numpy.uint16)
    self.assertTrue(ply2.encoding_to_dtype('nat32')==numpy.uint32)
    self.assertTrue(ply2.encoding_to_dtype('nat64')==numpy.uint64)
    with self.assertRaises(IOError):
      ply2.encoding_to_dtype('nat128')
    
    self.assertTrue(ply2.encoding_to_dtype('int8')==numpy.int8)
    self.assertTrue(ply2.encoding_to_dtype('int16')==numpy.int16)
    self.assertTrue(ply2.encoding_to_dtype('int32')==numpy.int32)
    self.assertTrue(ply2.encoding_to_dtype('int64')==numpy.int64)
    with self.assertRaises(IOError):
      ply2.encoding_to_dtype('int128')
    
    self.assertTrue(ply2.encoding_to_dtype('real16')==numpy.float16)
    self.assertTrue(ply2.encoding_to_dtype('real32')==numpy.float32)
    self.assertTrue(ply2.encoding_to_dtype('real64')==numpy.float64)
    self.assertTrue(ply2.encoding_to_dtype('real128')==numpy.float128)
    
    self.assertTrue(ply2.encoding_to_dtype('string:uint8')==numpy.object)
    self.assertTrue(ply2.encoding_to_dtype('array:2:uint32:real32')==numpy.object)
    
    with self.assertRaises(IOError):
      ply2.encoding_to_dtype('red shirt')
  
  
  def tests_array_to_encoding(self):
    self.assertTrue(ply2.array_to_encoding(numpy.zeros(8,dtype=numpy.uint8))=='nat8')
    self.assertTrue(ply2.array_to_encoding(numpy.zeros(8,dtype=numpy.uint16))=='nat16')
    self.assertTrue(ply2.array_to_encoding(numpy.zeros(8,dtype=numpy.uint32))=='nat32')
    self.assertTrue(ply2.array_to_encoding(numpy.zeros(8,dtype=numpy.uint64))=='nat64')
    
    self.assertTrue(ply2.array_to_encoding(numpy.zeros(8,dtype=numpy.int8))=='int8')
    self.assertTrue(ply2.array_to_encoding(numpy.zeros(8,dtype=numpy.int16))=='int16')
    self.assertTrue(ply2.array_to_encoding(numpy.zeros(8,dtype=numpy.int32))=='int32')
    self.assertTrue(ply2.array_to_encoding(numpy.zeros(8,dtype=numpy.int64))=='int64')
    
    self.assertTrue(ply2.array_to_encoding(numpy.zeros(8,dtype=numpy.float16))=='real16')
    self.assertTrue(ply2.array_to_encoding(numpy.zeros(8,dtype=numpy.float32))=='real32')
    self.assertTrue(ply2.array_to_encoding(numpy.zeros(8,dtype=numpy.float64))=='real64')
    self.assertTrue(ply2.array_to_encoding(numpy.zeros(8,dtype=numpy.float128))=='real128')
    
    with self.assertRaises(IOError):
      ply2.array_to_encoding(numpy.zeros(8,dtype=numpy.complex))
    
    test = numpy.zeros(8,dtype=numpy.object)
    test[0] = 'Hello world'
    self.assertTrue(ply2.array_to_encoding(test)=='string:nat32')
    
    test[0] = numpy.zeros((4,4), dtype=numpy.int16)
    self.assertTrue(ply2.array_to_encoding(test)=='array:2:nat32:int16')
    
    test[0] = numpy.zeros((4,4), dtype=numpy.complex)
    with self.assertRaises(IOError):
      ply2.array_to_encoding(test)



if __name__ == '__main__':
  unittest.main()
