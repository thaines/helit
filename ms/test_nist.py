#! /usr/bin/env python

# Copyright 2013 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import sys
import os.path
import random

import cv
from utils.cvarray import *
import numpy

from ms import MeanShift



# NOTE: This test assumes you have a directory called nist containing the nist images in it, with black for background, white for foreground.



# Get a list of files to process...
fn_list = os.listdir('nist')
fn_list = filter(lambda fn: fn[-9:]!='_line.png', fn_list)



# Setup a mean shift object for use...
ms = MeanShift()
ms.set_spatial('iter_dual')



# Process each in turn...
for fn in fn_list:
  print 'Doing %s...' % fn
  fn = os.path.join('nist',fn)
  
  # Load image and binarise...
  image = cv.LoadImage(fn)
  image = cv2array(image)
  image = image[:,:,0] > 128

  # Finish setup of meanshift object...
  ms.set_data(image, 'bb', 2)
  ms.set_scale(numpy.array([0.65, 0.65]))
  
  # Find the manifold...
  loc = ms.manifolds_data(1)
  
  # Render an image of the line...
  output = numpy.zeros((image.shape[0], image.shape[1], 3), dtype=numpy.float32)
  output[:,:,:] = 0.0

  for y in xrange(loc.shape[0]):
    for x in xrange(loc.shape[1]):
      if image[y,x]:
        oy = int(loc[y,x,0]+0.5)
        ox = int(loc[y,x,1]+0.5)
        if oy>=0 and oy<output.shape[0] and ox>=0 and ox<output.shape[1]:
          output[oy,ox,:] = 255.0
  
  # Save the rendered image...
  root, ext = os.path.splitext(fn)
  ofn = root + '_line.png'
  output = array2cv(output)
  cv.SaveImage(ofn, output)
