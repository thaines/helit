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

ms.set_kernel(random.choice(filter(lambda s: s!='fisher', ms.kernels())))
ms.set_spatial('iter_dual')
ms.set_balls('hash')

spatial_scale = 16.0
colour_scale = 32.0
ms.set_scale(numpy.array([1.0/spatial_scale, 1.0/spatial_scale, 1.0/colour_scale, 1.0/colour_scale, 1.0/colour_scale]))

ms.quality = 0.0
ms.ident_dist = 0.3
ms.merge_range = 0.6
ms.merge_check_step = 1



# Print out basic stats...
print 'kernel = %s; spatial = %s' % (ms.get_kernel(), ms.get_spatial())
print 'exemplars = %i; features = %i' % (ms.exemplars(), ms.features())
print 'quality = %.3f; epsilon = %.3f; iter_cap = %i' % (ms.quality, ms.epsilon, ms.iter_cap)
print 'ident_dist = %.3f; merge_range = %.3f; merge_check_step = %i' % (ms.ident_dist, ms.merge_range, ms.merge_check_step)
print



# Generate a segmentation image...
modes, indices = ms.cluster()
image = modes[indices.flatten(),2:].reshape(image.shape)

print 'Found %i modes' % modes.shape[0]

#image = ms.modes_data()[:,:,2:] # Gets to the same result (Ignoring floating point variations), but crazy slow.



# Save the segmentation image...
root, ext = os.path.splitext(fn)
ofn = root + '_seg' + ext
image = array2cv(image)
cv.SaveImage(ofn, image)
