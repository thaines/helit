#! /usr/bin/env python

# Copyright (c) 2016, Tom SF Haines
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#  * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import sys

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
    self.assertTrue(ply2.encoding_to_dtype('nat8')[0]==numpy.uint8)
    self.assertTrue(ply2.encoding_to_dtype('nat16')[0]==numpy.uint16)
    self.assertTrue(ply2.encoding_to_dtype('nat32')[0]==numpy.uint32)
    self.assertTrue(ply2.encoding_to_dtype('nat64')[0]==numpy.uint64)
    with self.assertRaises(NotImplementedError):
      ply2.encoding_to_dtype('nat128')
    
    self.assertTrue(ply2.encoding_to_dtype('int8')[0]==numpy.int8)
    self.assertTrue(ply2.encoding_to_dtype('int16')[0]==numpy.int16)
    self.assertTrue(ply2.encoding_to_dtype('int32')[0]==numpy.int32)
    self.assertTrue(ply2.encoding_to_dtype('int64')[0]==numpy.int64)
    with self.assertRaises(NotImplementedError):
      ply2.encoding_to_dtype('int128')
    
    self.assertTrue(ply2.encoding_to_dtype('real16')[0]==numpy.float16)
    self.assertTrue(ply2.encoding_to_dtype('real32')[0]==numpy.float32)
    self.assertTrue(ply2.encoding_to_dtype('real64')[0]==numpy.float64)
    self.assertTrue(ply2.encoding_to_dtype('real128')[0]==numpy.float128)
    
    self.assertTrue(ply2.encoding_to_dtype('string:nat8')==(numpy.object,None,numpy.uint8,None))
    self.assertTrue(ply2.encoding_to_dtype('array:2:nat32:real32')==(numpy.object,2,numpy.uint32,numpy.float32))
    
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
  
  
  
  def ds_empty(self):
    return dict()
  
  
  def ds_default(self):
    return ply2.create()
  
  
  def ds_empty_element(self):
    data = ply2.create()
    data['element']['dummy'] = dict()
    return data
  
  
  def ds_ints(self):
    data = ply2.create()
    data['element']['values'] = dict()
    data['element']['values']['x'] = numpy.array([1, 2, 3], dtype=numpy.int32)
    return data
  
  
  def ds_floats(self):
    data = ply2.create()
    data['element']['values'] = OrderedDict()
    data['element']['values']['x'] = numpy.array([1, 2, 3], dtype=numpy.float32)
    data['element']['values']['y'] = numpy.array([1.5, 5.6, numpy.pi], dtype=numpy.float32)
    return data
  
  
  def ds_image(self):
    data = ply2.create()
    data['type'].append('image.rgb')
    data['element']['pixel'] = OrderedDict()
    data['element']['pixel']['red'] = numpy.array([[0, 255], [0, 0]], dtype=numpy.uint8)
    data['element']['pixel']['green'] = numpy.array([[0, 0], [255, 0]], dtype=numpy.uint8)
    data['element']['pixel']['blue'] = numpy.array([[0, 0], [0, 255]], dtype=numpy.uint8)
    return data
  
  
  def ds_strings(self):
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
    
    return data
  
  
  def ds_arrays(self):
    samples = numpy.zeros(5, dtype=numpy.object)
    samples[0] = numpy.array([3, 1, 4, 2], dtype=numpy.int8)
    samples[1] = numpy.array([42, 42], dtype=numpy.int8) 
    samples[2] = numpy.array([100, 101, 102, -1, -2, 0], dtype=numpy.int8)
    samples[3] = numpy.array([-12], dtype=numpy.int8)
    samples[4] = numpy.array([], dtype=numpy.int8)
    
    data = ply2.create()
    data['element']['samples'] = OrderedDict()
    data['element']['samples']['values'] = samples
    
    return data
  
  
  def ds_meta(self):
    data = ply2.create()
    data['meta']['author'] = 'Cthulhu'
    data['meta']['tentacles'] = 42
    data['meta']['pi'] = numpy.pi
    return data
  
  
  def ds_map(self):
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
    
    return data
  
  
  def ds_mesh(self):
    data = ply2.create()
    data['type'].append('mesh')
    
    verts = numpy.empty((8, 3), dtype=numpy.float32)
    verts[0,:] = ( 1.0,  1.0, -1.0)
    verts[1,:] = ( 1.0, -1.0, -1.0)
    verts[2,:] = (-1.0, -1.0, -1.0)
    verts[3,:] = (-1.0,  1.0, -1.0)
    verts[4,:] = ( 1.0,  1.0,  1.0)
    verts[5,:] = (-1.0,  1.0,  1.0)
    verts[6,:] = (-1.0, -1.0,  1.0)
    verts[7,:] = ( 1.0, -1.0,  1.0)
    
    faces = numpy.empty(6, dtype=numpy.object)
    faces[0] = numpy.array([0, 1, 2, 3], dtype=numpy.int32)
    faces[1] = numpy.array([4, 5, 6, 7], dtype=numpy.int32)
    faces[2] = numpy.array([0, 4, 7, 1], dtype=numpy.int32)
    faces[3] = numpy.array([1, 7, 6, 2], dtype=numpy.int32)
    faces[4] = numpy.array([2, 6, 5, 3], dtype=numpy.int32)
    faces[5] = numpy.array([4, 0, 3, 5], dtype=numpy.int32)
    
    data['element']['vertex'] = OrderedDict()
    data['element']['vertex']['x'] = verts[:,0]
    data['element']['vertex']['y'] = verts[:,1]
    data['element']['vertex']['z'] = verts[:,2]
    
    data['element']['face'] = OrderedDict()
    data['element']['face']['vertex_indices'] = faces
    
    return data
  
  
  def ds_colour_map(self):
    data = ply2.create()
    data['type'].append('colour_map.rgb')
    
    samples = numpy.empty((3, 6), dtype=numpy.float32)
    samples[0,:] = [0, 0, 0, 0, 0, 0]
    samples[1,:] = [1, 1, 1, 1, 1, 1]
    samples[2,:] = [0.5, 0.5, 0.5, 0.25, 0.25, 0.25]
    
    data['element']['sample'] = OrderedDict()
    data['element']['sample']['in.r'] = samples[:,0]
    data['element']['sample']['in.g'] = samples[:,1]
    data['element']['sample']['in.b'] = samples[:,2]
    data['element']['sample']['out.r'] = samples[:,3]
    data['element']['sample']['out.g'] = samples[:,4]
    data['element']['sample']['out.b'] = samples[:,5]
    
    return data
  
  
  def ds_graph(self):
    data = ply2.create()
    
    name = numpy.zeros(2, dtype=numpy.object)
    name[0] = 'x'
    name[1] = 'y'
    
    value = numpy.zeros(2, dtype=numpy.object)
    value[0] = numpy.sin(numpy.linspace(0.0, numpy.pi, 32))
    value[1] = numpy.cos(numpy.linspace(0.0, numpy.pi, 32))
    
    data['element']['variable'] = OrderedDict()
    data['element']['variable']['name'] = name
    data['element']['variable']['value'] = value
    
    return data


  
  def test_write_empty(self):
    before = self.ds_empty()
    
    temp = tempfile.TemporaryFile('w+b')
    ply2.write(temp, before)
    
    temp.seek(0)
    data = temp.read()
    
    expected = ['ply', 'format ascii 2.0', 'end_header', '']
    self.assertTrue(data.decode('utf8')=='\n'.join(expected))
    
    temp.close()

  
  def test_write_default(self):
    before = self.ds_default()
    
    temp = tempfile.TemporaryFile('w+b')
    ply2.write(temp, before)
    
    temp.seek(0)
    data = temp.read()
    
    expected = ['ply', 'format ascii 2.0', 'end_header', '']
    self.assertTrue(data=='\n'.join(expected).encode('utf8'))
    
    temp.close()
  
  
  def test_write_empty_element(self):
    before = self.ds_empty_element()
    
    temp = tempfile.TemporaryFile('w+b')
    ply2.write(temp, before)
    
    temp.seek(0)
    data = temp.read()

    expected = ['ply', 'format ascii 2.0', 'element dummy 0', 'end_header', '']
    self.assertTrue(data=='\n'.join(expected).encode('utf8'))
    
    temp.close()


  def test_write_ints(self):
    before = self.ds_ints()
    
    temp = tempfile.TemporaryFile('w+b')
    ply2.write(temp, before)
    
    temp.seek(0)
    data = temp.read()
    
    expected = ['ply', 'format ascii 2.0', 'element values 3', 'property int32 x', 'end_header', '1', '2', '3', '']
    self.assertTrue(data.decode('utf8')=='\n'.join(expected))
    
    temp.close()
  
  
  def test_write_floats(self):
    before = self.ds_floats()
    
    temp = tempfile.TemporaryFile('w+b')
    ply2.write(temp, before)
    
    temp.seek(0)
    data = temp.read()
    
    expected = ['ply', 'format ascii 2.0', 'element values 3', 'property real32 x', 'property real32 y', 'end_header', '1 1.5', '2 5.5999999', '3 3.1415927', '']
    self.assertTrue(data.decode('utf8')=='\n'.join(expected))
    
    temp.close()
  
  
  def test_write_image(self):
    before = self.ds_image()
    
    temp = tempfile.TemporaryFile('w+b')
    ply2.write(temp, before)
    
    temp.seek(0)
    data = temp.read()
    
    expected = ['ply', 'format ascii 2.0', 'type image.rgb', 'element pixel 2 2', 'property nat8 red', 'property nat8 green', 'property nat8 blue', 'end_header', '0 0 0', '255 0 0', '0 255 0', '0 0 255', '']
    self.assertTrue(data=='\n'.join(expected).encode('utf8'))
    
    temp.close()


  def test_write_strings(self):
    before = self.ds_strings()
    
    temp = tempfile.TemporaryFile('w+b')
    ply2.write(temp, before)
    
    temp.seek(0)
    data = temp.read()

    expected = ['ply', 'format ascii 2.0', 'element people 8', 'property string:nat32 name', 'end_header', '9 The Alien', '6 Ripley', '3 Ash', '7 Bite Me', '4     ', '9   Penguin', '7 Joker  ', '8 Two\nFace', '']
    self.assertTrue(data=='\n'.join(expected).encode('utf8'))
    
    temp.close()
  
  
  def test_write_arrays(self):
    before = self.ds_arrays()
    
    temp = tempfile.TemporaryFile('w+b') 
    ply2.write(temp, before)
    
    temp.seek(0)
    data = temp.read()

    expected = ['ply', 'format ascii 2.0', 'element samples 5', 'property array:1:nat32:int8 values', 'end_header', '4 3 1 4 2', '2 42 42', '6 100 101 102 -1 -2 0', '1 -12', '0 ', '']
    self.assertTrue(data.decode('utf8')=='\n'.join(expected))
    
    temp.close()
  
  
  def test_write_meta(self):
    before = self.ds_meta()
    
    temp = tempfile.TemporaryFile('w+b')
    ply2.write(temp, before)
    
    temp.seek(0)
    data = temp.read()

    expected = ['ply', 'format ascii 2.0', 'meta string:nat32 author 7 Cthulhu', 'meta int64 tentacles 42', 'meta real64 pi 3.141592653589793', 'end_header', '']
    self.assertTrue(data.decode('utf8')=='\n'.join(expected))
    
    temp.close()
  
  
  def test_write_map(self):
    before = self.ds_map()
    
    temp = tempfile.TemporaryFile('w+b')
    ply2.write(temp, before)
    
    temp.seek(0)
    data = temp.read()

    expected = ['ply', 'format ascii 2.0', 'type map', 'meta string:nat32 location 12 Middle Earth', 'element city 9', 'property string:nat32 name', 'property real32 x', 'property real32 y', 'element tower 3', 'property string:nat32 name', 'property real32 x', 'property real32 y', 'end_header', '8 Hobbiton 67.400002 55.5', '4 Bree 79 54.099998', '9 Rivendell 99.099998 53.299999', '5 Moria 100.5 69.5', u'11 Lothl\u00f3rien 113 74.599998', '6 Edoras 105.1 99.599998', "11 Helm's Deep 99.5 98.699997", '8 Isengard 98.5 94.800003', '12 Minas Tirith 135.60001 111.3', '7 Orthanc 98.5 94.800003', u'10 Barad-d\u00fbr 156.2 107.8', '12 Cirith Ungol 145.2 111', '']
    self.assertTrue(data.decode('utf8')=='\n'.join(expected))
    
    temp.close()
  
  
  def test_write_mesh(self):
    before = self.ds_mesh()
    
    temp = tempfile.TemporaryFile('w+b')
    ply2.write(temp, before)
    
    temp.seek(0)
    data = temp.read()

    expected = ['ply', 'format ascii 2.0', 'type mesh', 'element vertex 8', 'property real32 x', 'property real32 y', 'property real32 z', 'element face 6', 'property array:1:nat32:int32 vertex_indices', 'end_header', '1 1 -1',
'1 -1 -1', '-1 -1 -1', '-1 1 -1', '1 1 1', '-1 1 1', '-1 -1 1', '1 -1 1', '4 0 1 2 3', '4 4 5 6 7', '4 0 4 7 1', '4 1 7 6 2', '4 2 6 5 3', '4 4 0 3 5', '']
    self.assertTrue(data.decode('utf8')=='\n'.join(expected))
    
    temp.close()


  def test_write_colour_map(self):
    before = self.ds_colour_map()
    
    temp = tempfile.TemporaryFile('w+b')
    ply2.write(temp, before)
    
    temp.seek(0)
    data = temp.read()
    
    expected = ['ply', 'format ascii 2.0', 'type colour_map.rgb', 'element sample 3', 'property real32 in.r', 'property real32 in.g', 'property real32 in.b', 'property real32 out.r', 'property real32 out.g', 'property real32 out.b', 'end_header', '0 0 0 0 0 0', '1 1 1 1 1 1', '0.5 0.5 0.5 0.25 0.25 0.25', '']
    self.assertTrue(data.decode('utf8')=='\n'.join(expected))
    
    temp.close()



  def equal(self, a, b):
    """Internal method that compares two ply files in the dictionary representation, to see if they are identical - will fail the test if not."""
    
    # Format...
    self.assertTrue((a['format'] if 'format' in a else 'ascii')==(b['format'] if 'format' in b else 'ascii'))
    
    # Type...
    self.assertTrue(set(a['type'] if 'type' in a else [])==set(b['type'] if 'type' in b else []))
   
    # Meta...
    a_meta = a['meta'] if 'meta' in a else dict()
    b_meta = b['meta'] if 'meta' in b else dict()
    
    self.assertTrue(set(a_meta.keys())==set(b_meta.keys()))
    
    for key, va in a_meta.items():
      vb = b_meta[key]
      if sys.version_info > (3, 0): # Problem in Python 2 due to str and unicode types.
        self.assertTrue(type(va)==type(vb))
      self.assertTrue(va==vb)
    
    # Comments...
    a_com = a['comment'] if 'comment' in a else dict()
    b_com = b['comment'] if 'comment' in b else dict()
    
    self.assertTrue(len(a_com)==len(b_com))
    
    for i in range(len(a_com)):
      self.assertTrue(i in a_com)
      self.assertTrue(i in b_com)
      self.assertTrue(a_com[i]==b_com[i])
    
    # Compress...
    a_comp = a['compress'] if 'compress' in a else ''
    b_comp = b['compress'] if 'compress' in b else ''
    
    if a_comp==None: a_comp = ''
    if b_comp==None: b_comp = ''
    self.assertTrue(a_comp==b_comp)
    
    # Element structure...
    a_elems = a['element'] if 'element' in a else dict()
    b_elems = b['element'] if 'element' in b else dict()
    
    self.assertTrue(set(a_elems.keys())==set(b_elems.keys()))
    
    for elem in a_elems:
      a_props = a_elems[elem]
      b_props = b_elems[elem]
      
      self.assertTrue(set(a_props.keys())==set(b_props.keys()))
      
      for prop in a_props:
        a_array = a_props[prop]
        b_array = b_props[prop]
        
        self.assertTrue(a_array.shape==b_array.shape)
        self.assertTrue(a_array.dtype==b_array.dtype)
        
        if a_array.dtype!=numpy.object:
          self.assertTrue((a_array==b_array).all())
        
        else:
          for index in numpy.ndindex(*a_array.shape):
            if isinstance(a_array[index], numpy.ndarray):
              error = numpy.fabs(a_array[index]-b_array[index])
              self.assertTrue((error<1e-6).all())
            else:
              if a_array[index]!=b_array[index]:
                print('a', type(a_array[index]), len(a_array[index]), a_array[index])
                print('b', type(b_array[index]), len(b_array[index]), b_array[index])
              self.assertTrue(a_array[index]==b_array[index])


  
  def test_write_read_empty(self):
    before = self.ds_empty()
    
    temp = tempfile.TemporaryFile('w+b')
    ply2.write(temp, before)

    temp.seek(0)
    after = ply2.read(temp)
    temp.close()
    
    self.equal(before, after)


  def test_write_read_default(self):
    before = self.ds_default()
    
    temp = tempfile.TemporaryFile('w+b')
    ply2.write(temp, before)

    temp.seek(0)
    after = ply2.read(temp)
    temp.close()
    
    self.equal(before, after)


  def test_write_read_empty_element(self):
    before = self.ds_empty_element()
    
    temp = tempfile.TemporaryFile('w+b')
    ply2.write(temp, before)

    temp.seek(0)
    after = ply2.read(temp)
    temp.close()
    
    self.equal(before, after)


  def test_write_read_ints(self):
    before = self.ds_ints()
    
    temp = tempfile.TemporaryFile('w+b')
    ply2.write(temp, before)

    temp.seek(0)
    after = ply2.read(temp)
    temp.close()
    
    self.equal(before, after)


  def test_write_read_floats(self):
    before = self.ds_floats()
    
    temp = tempfile.TemporaryFile('w+b')
    ply2.write(temp, before)

    temp.seek(0)
    after = ply2.read(temp)
    temp.close()
    
    self.equal(before, after)


  def test_write_read_image(self):
    before = self.ds_image()
    
    temp = tempfile.TemporaryFile('w+b')
    ply2.write(temp, before)

    temp.seek(0)
    after = ply2.read(temp)
    temp.close()
    
    self.equal(before, after)


  def test_write_read_strings(self):
    before = self.ds_strings()
    
    temp = tempfile.TemporaryFile('w+b')
    ply2.write(temp, before)

    temp.seek(0)
    after = ply2.read(temp)
    temp.close()
    
    self.equal(before, after)


  def test_write_read_arrays(self):
    before = self.ds_arrays()
    
    temp = tempfile.TemporaryFile('w+b')
    ply2.write(temp, before)

    temp.seek(0)
    after = ply2.read(temp)
    temp.close()
    
    self.equal(before, after)
    

  def test_write_read_meta(self):
    before = self.ds_meta()
    
    temp = tempfile.TemporaryFile('w+b')
    ply2.write(temp, before)

    temp.seek(0)
    after = ply2.read(temp)
    temp.close()
    
    self.equal(before, after)
    
    
  def test_write_read_map(self):
    before = self.ds_map()
    
    temp = tempfile.TemporaryFile('w+b')
    ply2.write(temp, before)

    temp.seek(0)
    after = ply2.read(temp)
    temp.close()
    
    self.equal(before, after)


  def test_write_read_mesh(self):
    before = self.ds_mesh()
    
    temp = tempfile.TemporaryFile('w+b')
    ply2.write(temp, before)

    temp.seek(0)
    after = ply2.read(temp)
    temp.close()
    
    self.equal(before, after)


  def test_write_read_colour_map(self):
    before = self.ds_colour_map()
    
    temp = tempfile.TemporaryFile('w+b')
    ply2.write(temp, before)

    temp.seek(0)
    after = ply2.read(temp)
    temp.close()
    
    self.equal(before, after)


  def test_write_read_graph(self):
    before = self.ds_graph()
    
    temp = tempfile.TemporaryFile('w+b')
    ply2.write(temp, before)

    temp.seek(0)
    after = ply2.read(temp)
    temp.close()
    
    self.equal(before, after)
  
  
  
  def test_read_minimal(self):
    lines = ['ply', 'format ascii 2.0', 'end_header', '']
    
    temp = tempfile.TemporaryFile('w+b')
    temp.write('\n'.join(lines).encode('utf8'))
    
    temp.seek(0)
    ply2.read(temp)
    temp.close()
  
  
  def test_read_no_end(self):
    lines = ['ply', 'format ascii 2.0']
    
    temp = tempfile.TemporaryFile('w+b')
    temp.write('\n'.join(lines).encode('utf8'))
    
    temp.seek(0)
    with self.assertRaises(EOFError):
      ply2.read(temp)
    temp.close()
  
  
  def test_read_past_limit(self):
    lines = ['ply', 'format ascii 2.0'] + ['comment I am a fish'] * 16384 + ['end_header', '']
    
    temp = tempfile.TemporaryFile('w+b')
    temp.write('\n'.join(lines).encode('utf8'))
    
    temp.seek(0)
    with self.assertRaises(BufferError):
      ply2.read(temp)
    temp.close()
  
  
  def test_read_no_magic(self):
    lines = ['format ascii 2.0', 'end_header', '']
    
    temp = tempfile.TemporaryFile('w+b')
    temp.write('\n'.join(lines).encode('utf8'))
    
    temp.seek(0)
    with self.assertRaises(AssertionError):
      ply2.read(temp)
    temp.close()


  def test_read_no_format(self):
    lines = ['ply', 'end_header', '']
    
    temp = tempfile.TemporaryFile('w+b')
    temp.write('\n'.join(lines).encode('utf8'))
    
    temp.seek(0)
    with self.assertRaises(AssertionError):
      ply2.read(temp)
    temp.close()
  
  
  def test_read_type(self):
    lines = ['ply', 'format ascii 2.0', 'type nothing', 'end_header', '']
    
    temp = tempfile.TemporaryFile('w+b')
    temp.write('\n'.join(lines).encode('utf8'))
    
    temp.seek(0)
    ply2.read(temp)
    temp.close()
  
  
  def test_read_multiple_type(self):
    lines = ['ply', 'format ascii 2.0', 'type nothing', 'type nothingness', 'end_header', '']
    
    temp = tempfile.TemporaryFile('w+b')
    temp.write('\n'.join(lines).encode('utf8'))
    
    temp.seek(0)
    with self.assertRaises(AssertionError):
      ply2.read(temp)
    temp.close()
  
  
  def test_read_multiple_compress(self):
    lines = ['ply', 'format ascii 2.0', 'compress gzip2', 'compress gzip2', 'end_header', '']
    
    temp = tempfile.TemporaryFile('w+b')
    temp.write('\n'.join(lines).encode('utf8'))
    
    temp.seek(0)
    with self.assertRaises(AssertionError):
      ply2.read(temp)
    temp.close()
    
    
  def test_read_bad_compress(self):
    lines = ['ply', 'format ascii 2.0', 'compress dancing gzip2', 'end_header', '']
    
    temp = tempfile.TemporaryFile('w+b')
    temp.write('\n'.join(lines).encode('utf8'))
    
    temp.seek(0)
    with self.assertRaises(AssertionError):
      ply2.read(temp)
    temp.close()


  def test_read_unknown_compress(self):
    lines = ['ply', 'format ascii 2.0', 'compress elephant', 'end_header', '']
    
    temp = tempfile.TemporaryFile('w+b')
    temp.write('\n'.join(lines).encode('utf8'))
    
    temp.seek(0)
    with self.assertRaises(AssertionError):
      ply2.read(temp)
    temp.close()
    
    
  def test_read_bad_length(self):
    lines = ['ply', 'format ascii 2.0', 'length 5 blue wahles', 'end_header', '']
    
    temp = tempfile.TemporaryFile('w+b')
    temp.write('\n'.join(lines).encode('utf8'))
    
    temp.seek(0)
    with self.assertRaises(AssertionError):
      ply2.read(temp)
    temp.close()
    
    lines = ['ply', 'format ascii 2.0', 'length nine', 'end_header', '']
    
    temp = tempfile.TemporaryFile('w+b')
    temp.write('\n'.join(lines).encode('utf8'))
    
    temp.seek(0)
    with self.assertRaises(ValueError):
      ply2.read(temp)
    temp.close()
    
  
  def test_read_null_element(self):
    lines = ['ply', 'format ascii 2.0', 'element penguin 0', 'end_header', '']
    
    temp = tempfile.TemporaryFile('w+b')
    temp.write('\n'.join(lines).encode('utf8'))
    
    temp.seek(0)
    ply2.read(temp)
    temp.close()


  def test_read_duplicate_element(self):
    lines = ['ply', 'format ascii 2.0', 'element penguin 0', 'element penguin 0', 'end_header', '']
    
    temp = tempfile.TemporaryFile('w+b')
    temp.write('\n'.join(lines).encode('utf8'))
    
    temp.seek(0)
    with self.assertRaises(AssertionError):
      ply2.read(temp)
    temp.close()
    
    
  def test_read_no_shape(self):
    lines = ['ply', 'format ascii 2.0', 'element penguin', 'end_header', '']
    
    temp = tempfile.TemporaryFile('w+b')
    temp.write('\n'.join(lines).encode('utf8'))
    
    temp.seek(0)
    with self.assertRaises(AssertionError):
      ply2.read(temp)
    temp.close()
 

  def test_read_antidata(self):
    lines = ['ply', 'format ascii 2.0', 'element penguin -4', 'end_header', '']
    
    temp = tempfile.TemporaryFile('w+b')
    temp.write('\n'.join(lines).encode('utf8'))
    
    temp.seek(0)
    with self.assertRaises(AssertionError):
      ply2.read(temp)
    temp.close()
  
  
  def test_read_naked_property(self):
    lines = ['ply', 'format ascii 2.0', 'property real32 nose_size', 'end_header', '']
    
    temp = tempfile.TemporaryFile('w+b')
    temp.write('\n'.join(lines).encode('utf8'))
    
    temp.seek(0)
    with self.assertRaises(AssertionError):
      ply2.read(temp)
    temp.close()
    
    
  def test_read_duplicate_property(self):
    lines = ['ply', 'format ascii 2.0', 'element penguin 0', 'property real32 nose_size', 'property real32 nose_size', 'end_header', '']
    
    temp = tempfile.TemporaryFile('w+b')
    temp.write('\n'.join(lines).encode('utf8'))
    
    temp.seek(0)
    with self.assertRaises(AssertionError):
      ply2.read(temp)
    temp.close()
  
  
  def test_read_bad_meta(self):
    # Missing...
    lines0 = ['ply', 'format ascii 2.0', 'meta real32', 'end_header', '']
    
    # Not a float...
    lines1 = ['ply', 'format ascii 2.0', 'meta real32 author bunny', 'end_header', '']
    
    # Not an int...
    lines2 = ['ply', 'format ascii 2.0', 'meta int32 time 3.5', 'end_header', '']
    
    # Excess...
    lines3 = ['ply', 'format ascii 2.0', 'meta real32 time 3.5 seconds', 'end_header', '']
    
    # Bad string length type...
    lines4 = ['ply', 'format ascii 2.0', 'meta string:rodent owner 4 Rupert', 'end_header', '']
    
    # Bad string length...
    lines5 = ['ply', 'format ascii 2.0', 'meta string:nat32 owner rat Rupert', 'end_header', '']
    
    # Wrong string length...
    lines6 = ['ply', 'format ascii 2.0', 'meta string:nat32 owner 8 Rupert', 'end_header', '']
    
    # No string length...
    lines7 = ['ply', 'format ascii 2.0', 'meta string:nat32 owner Rupert', 'end_header', '']
    
    # Unknown type...
    lines8 = ['ply', 'format ascii 2.0', 'meta quaternion angle 3.2 1.2 4.5 -0.3', 'end_header', '']

    # Do all of the above...
    for lines, error in [(lines0,KeyError), (lines1,ValueError), (lines2,ValueError), (lines3,ValueError), (lines4,IOError), (lines5,ValueError), (lines6,IOError), (lines7,ValueError), (lines8,IOError)]:
      temp = tempfile.TemporaryFile('w+b')
      temp.write('\n'.join(lines).encode('utf8'))
    
      temp.seek(0)
      with self.assertRaises(error):
        ply2.read(temp)
      temp.close()


  def test_read_line_endings_dos(self):
    lines = ['ply', 'format ascii 2.0', 'end_header', '']
    
    temp = tempfile.TemporaryFile('w+b')
    temp.write('\n\r'.join(lines).encode('utf8'))
    
    temp.seek(0)
    with self.assertRaises(EOFError):
      ply2.read(temp)
    temp.close()


  def test_read_line_endings_mac(self):
    lines = ['ply', 'format ascii 2.0', 'end_header', '']
    
    temp = tempfile.TemporaryFile('w+b')
    temp.write('\r\n'.join(lines).encode('utf8'))
    
    temp.seek(0)
    with self.assertRaises(EOFError):
      ply2.read(temp)
    temp.close()


  def test_read_small(self):
    lines = ['ply', 'format ascii 2.0', 'element values 2', 'property int32 size', 'end_header', '4', '5']
    
    temp = tempfile.TemporaryFile('w+b')
    temp.write('\n'.join(lines).encode('utf8'))
    
    temp.seek(0)
    ply2.read(temp)
    temp.close()


  def test_read_incomplete(self):
    lines = ['ply', 'format ascii 2.0', 'element values 2', 'property int32 size', 'end_header', '4']
    
    temp = tempfile.TemporaryFile('w+b')
    temp.write('\n'.join(lines).encode('utf8'))
    
    temp.seek(0)
    with self.assertRaises(IOError):
      ply2.read(temp)
    temp.close()
    
    
  def test_read_bad_properties(self):
    # Unknown type...
    lines0 = ['ply', 'format ascii 2.0', 'element values 0', 'property elephants size', 'end_header', '']
    
    # Incomplete string...
    lines1 = ['ply', 'format ascii 2.0', 'element values 0', 'property string size', 'end_header', '']
    
    # Wrong string...
    lines2 = ['ply', 'format ascii 2.0', 'element values 0', 'property string:bytes size', 'end_header', '']
    
    # Crazy string...
    lines3 = ['ply', 'format ascii 2.0', 'element values 0', 'property string:real32 size', 'end_header', '']
    
    # Incomplete array spec...
    lines4 = ['ply', 'format ascii 2.0', 'element values 0', 'property array:2: size', 'end_header', '']
    
    # Negative dimensionality array...
    lines5 = ['ply', 'format ascii 2.0', 'element values 0', 'property array:-2:nat32:real32 size', 'end_header', '']
    
    # Silly shape measure...
    lines6 = ['ply', 'format ascii 2.0', 'element values 0', 'property array:1:real32:real32 size', 'end_header', '']
    
    # Complex nesting...
    lines7 = ['ply', 'format ascii 2.0', 'element values 0', 'property array:1:real32:string:nat32 size', 'end_header', '']
    
    # Do all of the above...
    for lines, error in [(lines0,IOError), (lines1,IndexError), (lines2,IOError), (lines3,IOError), (lines4,KeyError), (lines5,ValueError), (lines6,IOError), (lines7,KeyError)]:
      temp = tempfile.TemporaryFile('w+b')
      temp.write('\n'.join(lines).encode('utf8'))
    
      temp.seek(0)
      with self.assertRaises(error):
        ply2.read(temp)
      temp.close()
  
  
  def test_adv_write_read_empty(self):
    for compress in ['', 'gzip', 'bzip2']:
      for format in ['ascii', 'binary_little_endian', 'binary_big_endian']:
        before = self.ds_empty()
        before['compress'] = compress
        before['format'] = format
    
        temp = tempfile.TemporaryFile('w+b')
        ply2.write(temp, before)

        temp.seek(0)
        after = ply2.read(temp)
        temp.close()
    
        self.equal(before, after)


  def test_adv_write_read_default(self):
    for compress in ['', 'gzip', 'bzip2']:
      for format in ['ascii', 'binary_little_endian', 'binary_big_endian']:
        before = self.ds_default()
        before['compress'] = compress
        before['format'] = format
    
        temp = tempfile.TemporaryFile('w+b')
        ply2.write(temp, before)

        temp.seek(0)
        after = ply2.read(temp)
        temp.close()
    
        self.equal(before, after)


  def test_adv_write_read_empty_element(self):
    for compress in ['', 'gzip', 'bzip2']:
      for format in ['ascii', 'binary_little_endian', 'binary_big_endian']:
        before = self.ds_empty_element()
        before['compress'] = compress
        before['format'] = format
    
        temp = tempfile.TemporaryFile('w+b')
        ply2.write(temp, before)

        temp.seek(0)
        after = ply2.read(temp)
        temp.close()
    
        self.equal(before, after)


  def test_adv_write_read_ints(self):
    for compress in ['', 'gzip', 'bzip2']:
      for format in ['ascii', 'binary_little_endian', 'binary_big_endian']:
        before = self.ds_ints()
        before['compress'] = compress
        before['format'] = format
    
        temp = tempfile.TemporaryFile('w+b')
        ply2.write(temp, before)

        temp.seek(0)
        after = ply2.read(temp)
        temp.close()
    
        self.equal(before, after)


  def test_adv_write_read_floats(self):
    for compress in ['', 'gzip', 'bzip2']:
      for format in ['ascii', 'binary_little_endian', 'binary_big_endian']:
        before = self.ds_floats()
        before['compress'] = compress
        before['format'] = format
    
        temp = tempfile.TemporaryFile('w+b')
        ply2.write(temp, before)

        temp.seek(0)
        after = ply2.read(temp)
        temp.close()
    
        self.equal(before, after)


  def test_adv_write_read_image(self):
    for compress in ['', 'gzip', 'bzip2']:
      for format in ['ascii', 'binary_little_endian', 'binary_big_endian']:
        before = self.ds_image()
        before['compress'] = compress
        before['format'] = format
    
        temp = tempfile.TemporaryFile('w+b')
        ply2.write(temp, before)

        temp.seek(0)
        after = ply2.read(temp)
        temp.close()
    
        self.equal(before, after)


  def test_adv_write_read_strings(self):
    for compress in ['', 'gzip', 'bzip2']:
      for format in ['ascii', 'binary_little_endian', 'binary_big_endian']:
        before = self.ds_strings()
        before['compress'] = compress
        before['format'] = format
    
        temp = tempfile.TemporaryFile('w+b')
        ply2.write(temp, before)

        temp.seek(0)
        after = ply2.read(temp)
        temp.close()
    
        self.equal(before, after)


  def test_adv_write_read_arrays(self):
    for compress in ['', 'gzip', 'bzip2']:
      for format in ['ascii', 'binary_little_endian', 'binary_big_endian']:
        before = self.ds_arrays()
        before['compress'] = compress
        before['format'] = format
    
        temp = tempfile.TemporaryFile('w+b')
        ply2.write(temp, before)

        temp.seek(0)
        after = ply2.read(temp)
        temp.close()
    
        self.equal(before, after)


  def test_adv_write_read_meta(self):
    for compress in ['', 'gzip', 'bzip2']:
      for format in ['ascii', 'binary_little_endian', 'binary_big_endian']:
        before = self.ds_meta()
        before['compress'] = compress
        before['format'] = format
    
        temp = tempfile.TemporaryFile('w+b')
        ply2.write(temp, before)

        temp.seek(0)
        after = ply2.read(temp)
        temp.close()
    
        self.equal(before, after)


  def test_adv_write_read_map(self):
    for compress in ['', 'gzip', 'bzip2']:
      for format in ['ascii', 'binary_little_endian', 'binary_big_endian']:
        before = self.ds_map()
        before['compress'] = compress
        before['format'] = format
    
        temp = tempfile.TemporaryFile('w+b')
        ply2.write(temp, before)

        temp.seek(0)
        after = ply2.read(temp)
        temp.close()
    
        self.equal(before, after)


  def test_adv_write_read_mesh(self):
    for compress in ['', 'gzip', 'bzip2']:
      for format in ['ascii', 'binary_little_endian', 'binary_big_endian']:
        before = self.ds_mesh()
        before['compress'] = compress
        before['format'] = format
    
        temp = tempfile.TemporaryFile('w+b')
        ply2.write(temp, before)

        temp.seek(0)
        after = ply2.read(temp)
        temp.close()
    
        self.equal(before, after)


  def test_adv_write_read_colour_map(self):
    for compress in ['', 'gzip', 'bzip2']:
      for format in ['ascii', 'binary_little_endian', 'binary_big_endian']:
        before = self.ds_colour_map()
        before['compress'] = compress
        before['format'] = format
    
        temp = tempfile.TemporaryFile('w+b')
        ply2.write(temp, before)

        temp.seek(0)
        after = ply2.read(temp)
        temp.close()
    
        self.equal(before, after)


  def test_adv_write_read_graph(self):
    for compress in ['', 'gzip', 'bzip2']:
      for format in ['ascii', 'binary_little_endian', 'binary_big_endian']:
        before = self.ds_graph()
        before['compress'] = compress
        before['format'] = format
    
        temp = tempfile.TemporaryFile('w+b')
        ply2.write(temp, before)

        temp.seek(0)
        after = ply2.read(temp)
        temp.close()
    
        self.equal(before, after)



if __name__ == '__main__':
  unittest.main()
