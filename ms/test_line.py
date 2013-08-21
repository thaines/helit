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



# Check an image filename has been provided on the command line...
if len(sys.argv)<2:
  print "Need an image filename"
  sys.exit(1)

fn = sys.argv[1]



# Load the image into a numpy array...
image = cv.LoadImage(fn)
image = cv2array(image)



# Perform mean shift, with full clustering...
ms = MeanShift()
ms.set_data(image, 'bbf')

ms.set_spatial('iter_dual')

spatial_scale = 4.0
colour_scale = 32.0
ms.set_scale(numpy.array([1.0/spatial_scale, 1.0/spatial_scale, 1.0/colour_scale, 1.0/colour_scale, 1.0/colour_scale]))

ms.quality = 0.0

print 'exemplars = %i; features = %i' % (ms.exemplars(), ms.features())



# Calculate the locations for every pixel...
loc = ms.manifolds_data(1)



# Draw the resulting lines...
output = numpy.zeros(image.shape, dtype=numpy.float32)
output[:,:,:] = 128.0

for y in xrange(loc.shape[0]):
  for x in xrange(loc.shape[1]):
    oy = int(loc[y,x,0]+0.5)
    ox = int(loc[y,x,1]+0.5)
    if oy>=0 and oy<output.shape[0] and ox>=0 and ox<output.shape[1]:
      output[oy,ox,:] = loc[y,x,2:]


    
# Save to disk...
root, ext = os.path.splitext(fn)
ofn = root + '_line' + ext
output = array2cv(output)
cv.SaveImage(ofn, output)
