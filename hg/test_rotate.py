#! /usr/bin/env python

# Copyright 2016 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import sys

import numpy
from scipy.misc import imread, imsave

from homography import *
from transform import *



# Simply does the required rotation, as a test it works...

# Load a file, make into a nice floaty image...
if len(sys.argv)<3:
  print('Usage:')
  print('  ./test_rotate.py <image fn> <angle>')
  sys.exit(1)

fn = sys.argv[1]

image = imread(fn).astype(numpy.float32)
shape = image.shape
image = {'r' : image[:,:,0], 'g' : image[:,:,1], 'b' : image[:,:,2]}

angle = float(sys.argv[2]) * numpy.pi / 180.0



# Calculate homography...
hg = translate([-0.5*shape[1], -0.5*shape[0]])
hg = rotate(angle).dot(hg)
hg = translate([0.5*shape[1], 0.5*shape[0]]).dot(hg)

hg, out_shape = fit(hg, shape)



# Apply...
image = transform(numpy.linalg.inv(hg), image, out_shape[0], out_shape[1])



# Save resulting file...
image = numpy.concatenate((image['r'][:,:,numpy.newaxis], image['g'][:,:,numpy.newaxis], image['b'][:,:,numpy.newaxis]), axis=2)
image = (image+0.5).astype(numpy.uint8)

image[0,0,:] = (255, 0, 0)

out_fn = os.path.splitext(fn)[0] + '_' + sys.argv[2] + '.png'
imsave(out_fn, image)
