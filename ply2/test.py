#! /usr/bin/env python

# Copyright (c) 2016, Tom SF Haines
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#  * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from collections import OrderedDict

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
  
  
  def test_write_empty(self):
    temp = tempfile.TemporaryFile('w+b')
    ply2.write(temp, dict())
    
    temp.seek(0)
    data = temp.read()
    
    expected = ['ply', 'format ascii 2.0', 'end_header', '']
    self.assertTrue(data=='\n'.join(expected))
    
    temp.close()

  
  def test_write_default(self):
    temp = tempfile.TemporaryFile('w+b')
    
    data = ply2.create()
    ply2.write(temp, data)
    
    temp.seek(0)
    data = temp.read()
    
    expected = ['ply', 'format ascii 2.0', 'end_header', '']
    self.assertTrue(data=='\n'.join(expected))
    
    temp.close()
  
  
  def test_write_empty_element(self):
    temp = tempfile.TemporaryFile('w+b')
    
    data = ply2.create()
    data['element']['dummy'] = dict()
    
    ply2.write(temp, data)
    
    temp.seek(0)
    data = temp.read()

    expected = ['ply', 'format ascii 2.0', 'element dummy 0', 'end_header', '']
    self.assertTrue(data=='\n'.join(expected))
    
    temp.close()


  def test_write_ints(self):
    temp = tempfile.TemporaryFile('w+b')
    
    data = ply2.create()
    data['element']['values'] = dict()
    data['element']['values']['x'] = numpy.array([1, 2, 3], dtype=numpy.int32)
    
    ply2.write(temp, data)
    
    temp.seek(0)
    data = temp.read()
    
    expected = ['ply', 'format ascii 2.0', 'element values 3', 'property int32 x', 'end_header', '1', '2', '3', '']
    self.assertTrue(data=='\n'.join(expected))
    
    temp.close()
  
  
  def test_write_floats(self):
    temp = tempfile.TemporaryFile('w+b')
    
    data = ply2.create()
    data['element']['values'] = OrderedDict()
    data['element']['values']['x'] = numpy.array([1, 2, 3], dtype=numpy.float32)
    data['element']['values']['y'] = numpy.array([1.5, 5.6, numpy.pi], dtype=numpy.float32)
    
    ply2.write(temp, data)
    
    temp.seek(0)
    data = temp.read()
    
    expected = ['ply', 'format ascii 2.0', 'element values 3', 'property real32 x', 'property real32 y', 'end_header', '1 1.5', '2 5.5999999', '3 3.1415927', '']
    self.assertTrue(data=='\n'.join(expected))
    
    temp.close()
  
  
  def test_write_image(self):
    temp = tempfile.TemporaryFile('w+b')
    
    data = ply2.create()
    data['type'].append('image.rgb')
    data['element']['pixel'] = OrderedDict()
    data['element']['pixel']['red'] = numpy.array([[0, 255], [0, 0]], dtype=numpy.uint8)
    data['element']['pixel']['green'] = numpy.array([[0, 0], [255, 0]], dtype=numpy.uint8)
    data['element']['pixel']['blue'] = numpy.array([[0, 0], [0, 255]], dtype=numpy.uint8)
    
    ply2.write(temp, data)
    
    temp.seek(0)
    data = temp.read()
    
    expected = ['ply', 'format ascii 2.0', 'type image.rgb', 'element pixel 2 2', 'property nat8 red', 'property nat8 green', 'property nat8 blue', 'end_header', '0 0 0', '255 0 0', '0 255 0', '0 0 255', '']
    self.assertTrue(data=='\n'.join(expected))
    
    temp.close()
  
  
  def test_write_strings(self):
    temp = tempfile.TemporaryFile('w+b')
    
    names = numpy.zeros(8, dtype=numpy.object)
    names[0] = 'The Alien'
    names[1] = 'Ripley'
    names[2] = 'Ash'
    names[3] = u'Bite Me'
    names[4] = '    '
    names[5] = '  Penguin'
    names[6] = 'Joker  '
    names[7] = 'Two\nFace'
    
    data = ply2.create()
    data['element']['people'] = OrderedDict()
    data['element']['people']['name'] = names
    
    ply2.write(temp, data)
    
    temp.seek(0)
    data = temp.read()

    expected = ['ply', 'format ascii 2.0', 'element people 8', 'property string:nat32 name', 'end_header', '9 The Alien', '6 Ripley', '3 Ash', '7 Bite Me', '4     ', '9   Penguin', '7 Joker  ', '8 Two\nFace', '']
    self.assertTrue(data=='\n'.join(expected))
    
    temp.close()
  
  
  def test_write_arrays(self):
    temp = tempfile.TemporaryFile('w+b')
    
    samples = numpy.zeros(5, dtype=numpy.object)
    samples[0] = numpy.array([3, 1, 4, 2], dtype=numpy.int8)
    samples[1] = numpy.array([42, 42], dtype=numpy.int8) 
    samples[2] = numpy.array([100, 101, 102, -1, -2, 0], dtype=numpy.int8)
    samples[3] = numpy.array([-12], dtype=numpy.int8)
    samples[4] = numpy.array([], dtype=numpy.int8)
    
    data = ply2.create()
    data['element']['samples'] = OrderedDict()
    data['element']['samples']['values'] = samples
    
    ply2.write(temp, data)
    
    temp.seek(0)
    data = temp.read()

    expected = ['ply', 'format ascii 2.0', 'element samples 5', 'property array:1:nat32:int8 values', 'end_header', '4 3 1 4 2', '2 42 42', '6 100 101 102 -1 -2 0', '1 -12', '0 ', '']
    self.assertTrue(data=='\n'.join(expected))
    
    temp.close()
  
  
  def test_write_meta(self):
    temp = tempfile.TemporaryFile('w+b')
    
    data = ply2.create()
    data['meta']['author'] = 'Cthulhu'
    data['meta']['tentacles'] = 42
    data['meta']['pi'] = numpy.pi
    
    ply2.write(temp, data)
    
    temp.seek(0)
    data = temp.read()

    expected = ['ply', 'format ascii 2.0', 'meta string:nat32 author 7 Cthulhu', 'meta int64 tentacles 42', 'meta float64 pi 3.141592653589793', 'end_header', '']
    self.assertTrue(data=='\n'.join(expected))
    
    temp.close()
  
  
  def test_write_map(self):
    temp = tempfile.TemporaryFile('w+b')
    
    data = ply2.create()
    data['type'].append('map')
    data['meta']['location'] = 'Middle Earth'
    
    names = numpy.zeros(9, dtype=numpy.object)
    names[0] = 'Hobbiton'
    names[1] = 'Bree'
    names[2] = 'Rivendell'
    names[3] = 'Moria'
    names[4] = u'Lothl\u00f3rien'
    names[5] = 'Edoras'
    names[6] = "Helm's Deep"
    names[7] = 'Isengard'
    names[8] = 'Minas Tirith'
    
    data['element']['city'] = OrderedDict()
    data['element']['city']['name'] = names
    data['element']['city']['x'] = numpy.array([67.4, 79.0, 99.1, 100.5, 113.0, 105.1, 99.5, 98.5, 135.6], dtype=numpy.float32)
    data['element']['city']['y'] = numpy.array([55.5, 54.1, 53.3, 69.5, 74.6, 99.6, 98.7, 94.8, 111.3], dtype=numpy.float32)
    
    names = numpy.zeros(3, dtype=numpy.object)
    names[0] = 'Orthanc'
    names[1] = u'Barad-d\u00fbr'
    names[2] = 'Cirith Ungol'
    
    data['element']['tower'] = OrderedDict()
    data['element']['tower']['name'] = names
    data['element']['tower']['x'] = numpy.array([98.5, 156.2, 145.2], dtype=numpy.float32)
    data['element']['tower']['y'] = numpy.array([94.8, 107.8, 111.0], dtype=numpy.float32)
    
    ply2.write(temp, data)
    
    temp.seek(0)
    data = temp.read()

    expected = ['ply', 'format ascii 2.0', 'type map', 'meta string:nat32 location 12 Middle Earth', 'element city 9', 'property string:nat32 name', 'property real32 x', 'property real32 y', 'element tower 3', 'property string:nat32 name', 'property real32 x', 'property real32 y', 'end_header', '8 Hobbiton 67.400002 55.5', '4 Bree 79 54.099998', '9 Rivendell 99.099998 53.299999', '5 Moria 100.5 69.5', u'11 Lothl\u00f3rien 113 74.599998', '6 Edoras 105.1 99.599998', "11 Helm's Deep 99.5 98.699997", '8 Isengard 98.5 94.800003', '12 Minas Tirith 135.60001 111.3', '7 Orthanc 98.5 94.800003', u'10 Barad-d\u00fbr 156.2 107.8', '12 Cirith Ungol 145.2 111', '']
    self.assertTrue(data.decode('utf8')=='\n'.join(expected))
    
    temp.close()



if __name__ == '__main__':
  unittest.main()
