#! /usr/bin/env python

# Copyright 2014 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import numpy
import numpy.random

import cv
from utils.cvarray import *
from utils.prog_bar import ProgBar

from ms import MeanShift



# Does a full 3D camera multiplication simulation...



# Function that, given a position, returns the angle axis to make a camera at that position point at the origin (ignores roll!)...
def to_origin(pos):
  start = numpy.array([1,0,0], dtype=numpy.float32) # Starting direction of camera.
  end = numpy.array(pos, dtype=numpy.float32)
  end /= numpy.sqrt(numpy.square(end).sum())
  
  # Direction is cross product - calclate and normalise, taking care of the whole zero length vector issue...
  aa = numpy.cross(start, end)
  
  # Angle from dot product...
  ang = numpy.arccos(start.dot(end))
  
  # Make sure the length of aa is right...
  if ang>1e-6:
    aa /= numpy.sqrt(numpy.square(aa).sum())
    aa *= ang
  
  return aa



# Create two camera distributions - a great circle at radius 4 pointing at the center in both cases...
samples = 1024


data_a = numpy.empty((samples, 6), dtype=numpy.float32)
data_a[:,0] = numpy.random.normal(size=samples)
data_a[:,1] = 0
data_a[:,2] = numpy.random.normal(size=samples)

data_a[:,:3] /= numpy.sqrt(numpy.square(data_a[:,:3]).sum(axis=1))[:,numpy.newaxis]
data_a[:,:3] *= 4.0

for i in xrange(samples):
  data_a[i,3:] = to_origin(data_a[i,:3])


data_b = numpy.empty((samples, 6), dtype=numpy.float32)
data_b[:,0] = numpy.random.normal(size=samples)
data_b[:,1] = numpy.random.normal(size=samples)
data_b[:,2] = 0

data_b[:,:3] /= numpy.sqrt(numpy.square(data_b[:,:3]).sum(axis=1))[:,numpy.newaxis]
data_b[:,:3] *= 4.0

for i in xrange(samples):
  data_b[i,3:] = to_origin(data_b[i,:3])



# Add a bit of noise...
data_a += 0.1 * numpy.random.normal(size=(samples,6))
data_b += 0.1 * numpy.random.normal(size=(samples,6))



# Create two distributions...
spatial_scale = 8.0
scale = numpy.array([spatial_scale, spatial_scale, spatial_scale, 1.0, 1.0, 1.0, 1.0])

mult_a = MeanShift()
mult_a.set_data(data_a, 'df', None, '...V')
mult_a.set_kernel('composite(3:gaussian,4:mirror_fisher(512.0))')
mult_a.set_spatial('kd_tree')
mult_a.set_scale(scale)

mult_b = MeanShift()
mult_b.set_data(data_b, 'df', None, '...V')
mult_b.copy_all(mult_a)
mult_b.set_scale(scale)



# A function for converting a distribution into a ply file...
def to_ply(fn, samples):
  # Open and header...
  f = open(fn, 'w')
  f.write('ply\n')
  f.write('format ascii 1.0\n');
  
  f.write('element vertex %i\n' % (samples.shape[0]*5))
  f.write('property float x\n')
  f.write('property float y\n')
  f.write('property float z\n')
  
  f.write('element edge %i\n' % (samples.shape[0]*8))
  f.write('property int vertex1\n')
  f.write('property int vertex2\n')
  
  f.write('end_header\n')
  
  # Add the vertices...
  for i in xrange(samples.shape[0]):
    base = numpy.array([1,0,0], dtype=numpy.float32)
    ang = numpy.sqrt(numpy.square(samples[i,3:]).sum())
    axis = samples[i,3:] / ang if ang>1e-6 else base
    
    offset_forward = numpy.cos(ang) * base + numpy.sin(ang) * numpy.cross(axis, base) + (1 - numpy.cos(ang)) * axis.dot(base) * axis
    
    base = numpy.array([0,1,0], dtype=numpy.float32)
    offset_left = numpy.cos(ang) * base + numpy.sin(ang) * numpy.cross(axis, base) + (1 - numpy.cos(ang)) * axis.dot(base) * axis
    
    base = numpy.array([0,0,1], dtype=numpy.float32)
    offset_up = numpy.cos(ang) * base + numpy.sin(ang) * numpy.cross(axis, base) + (1 - numpy.cos(ang)) * axis.dot(base) * axis
    
    cor_a = samples[i,:3] + offset_left * 0.1 + offset_up * 0.1
    cor_b = samples[i,:3] + offset_left * 0.1 - offset_up * 0.1
    cor_c = samples[i,:3] - offset_left * 0.1 - offset_up * 0.1
    cor_d = samples[i,:3] - offset_left * 0.1 + offset_up * 0.1
    tail = samples[i,:3] + offset_forward * 0.3 # Wrong way - something is wrong somewhere, I guess.
    
    f.write('%f %f %f\n' % tuple(cor_a))
    f.write('%f %f %f\n' % tuple(cor_b))
    f.write('%f %f %f\n' % tuple(cor_c))
    f.write('%f %f %f\n' % tuple(cor_d))
    f.write('%f %f %f\n' % tuple(tail))
  
  # Add the edges...
  for i in xrange(samples.shape[0]):
    f.write('%i %i\n' % (i*5, i*5+1))
    f.write('%i %i\n' % (i*5+1, i*5+2))
    f.write('%i %i\n' % (i*5+2, i*5+3))
    f.write('%i %i\n' % (i*5+3, i*5))
    
    f.write('%i %i\n' % (i*5+4, i*5))
    f.write('%i %i\n' % (i*5+4, i*5+1))
    f.write('%i %i\n' % (i*5+4, i*5+2))
    f.write('%i %i\n' % (i*5+4, i*5+3))
  
  # Close...
  f.close()



# Save the two distributions...
to_ply('camera_mult_a.ply', data_a)
to_ply('camera_mult_b.ply', data_b)



# Multiply them together...
final = 64

data_ab = numpy.empty((final, 6), dtype=numpy.float32)

MeanShift.mult((mult_a, mult_b), data_ab)



# Save out the multiplication...
to_ply('camera_mult_ab.ply', data_ab)
