#! /usr/bin/env python

# Copyright 2014 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import sys
import os.path

import cv
import numpy

from utils.cvarray import *
from utils.prog_bar import ProgBar

from ms import MeanShift



# Check an image filename has been provided on the command line...
if len(sys.argv)<2:
  print "Needs an image filename. An arbitrary second parameter causes it to store its results in a hsf5 file rather than dump them as images."
  sys.exit(1)

fn = sys.argv[1]



# Load the image into a numpy array...
image = cv.LoadImage(fn)
image = cv2array(image)



# Setup the mean shift object...
ms = MeanShift()
ms.set_data(image, 'bbf')

ms.set_kernel('uniform')
ms.set_spatial('iter_dual')
ms.set_balls('hash')



# Calculate the hierarchy of segments...
low_scale = 8.0
low_colour = 6.0
high_scale = 256.0
high_colour = 96.0

low  = numpy.array([low_scale, low_scale, low_colour, low_colour, low_colour])
high = numpy.array([high_scale, high_scale, high_colour, high_colour, high_colour])

steps =  32

pb = ProgBar()
hier = ms.hierarchy(low = low, high = high, steps = steps, callback = pb.callback)
del pb

if len(sys.argv)==2:
  for l, level in enumerate(hier):
    print '%i clusters at level %i' % (level[0].shape[0], l)
  print



# Assign colours to the segments, all levels...
def int_to_col(val):
  ret = [0, 0, 0]
  val += 1 # Skip black
  
  amount = 1
  for bit in xrange(32):
    if bit & val:
      ret[bit%3] += amount

    if (bit%3)==2:
      amount *= 2
  
  seq = [0.0, 1.0, 0.5, 0.25, 0.75, 0.125, 0.375, 0.625, 0.875, 0.0625, 0.1875, 0.3125, 0.4375, 0.5625, 0.6875, 0.8125, 0.9375]
  ret = map(lambda i: seq[i%len(seq)], ret)
  return tuple(ret)

colours = numpy.empty((hier[0][0].shape[0],3), dtype=numpy.float32)
for i in xrange(colours.shape[0]):
  colours[i,:] = int_to_col(i)
colours *= 255.0



if len(sys.argv)==2:
  # Render the scales of the hierarchy, saving them out to files...
  out_fn = os.path.splitext(fn)[0] + '_%i.png'

  index = hier[0][1].copy()

  for l, level in enumerate(hier):
    if l==0: # Bottom level is just the image - skip
      continue 
  
    # Generate image from current index...
    out = colours[index].reshape((image.shape[0], image.shape[1], 3))
  
    # Save image...
    out = array2cv(out)
    cv.SaveImage(out_fn % l, out)
  
    # Apply this levels transformation...
    if l!=(len(hier)-1):
      index = level[1][index]

else:
  # Save the data to a hdf5 file...
  import h5py
  
  # Open file...
  out_fn = os.path.splitext(fn)[0] + '.hms'
  f = h5py.File(out_fn, 'w')
  
  # Store meta data...
  f.attrs['low.scale'] = low_scale
  f.attrs['low.colour'] = low_colour
  f.attrs['high.scale'] = high_scale
  f.attrs['high.colour'] = high_colour
  f.attrs['steps'] = steps
  
  # Store data...
  for l, level in enumerate(hier):
    if l!=0: # Original image contains this, and much better compressed - save space.
      f.create_dataset('%i.clusters'%l, data=level[0], compression='gzip')
    if level[1]!=None:
      f.create_dataset('%i.parents'%l, data=level[1], compression='gzip')
    f.create_dataset('%i.sizes'%l, data=level[2], compression='gzip')
  
  # Close...
  f.close()


print 'Done.'
