#! /usr/bin/env python

# Copyright 2016 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import sys
from collections import OrderedDict

import numpy
from scipy.misc import imread, imsave

from homography import *
from transform import *



# Does a 1 degree rotation 32 times to a single image, then saves the file out, for several different interpolation methods to see how they fail...

# Load a file, make into a nice floaty image...
if len(sys.argv)<2:
  print('Usage:')
  print('  ./test_spin.py <image fn>')
  sys.exit(1)

fn = sys.argv[1]

image = imread(fn).astype(numpy.float32)
shape = image.shape
image = {'r' : image[:,:,0], 'g' : image[:,:,1], 'b' : image[:,:,2]}



# Totally pointless, but the purpose of this test (!)...
def rotate_noop(image, steps = 32, transform = transform):
  rot = translate([-0.5*shape[1], -0.5*shape[0]])
  rot = rotate(numpy.pi*2.0 / steps).dot(rot)
  rot = translate([0.5*shape[1], 0.5*shape[0]]).dot(rot)
  
  for _ in xrange(steps):
    image = transform(rot, image)
  
  fillmasked(image)
  
  return image



# Create dictionary of algorithms to try...
algs = OrderedDict()

algs['B-Spline 0 (nearest neighbour)'] = lambda hg, image: transform(hg, image, -1, -1, 0)
algs['B-Spline 1 (linear)'] = lambda hg, image: transform(hg, image, -1, -1, 1)
algs['B-Spline 2 (quadratic)'] = lambda hg, image: transform(hg, image, -1, -1, 2)
algs['B-Spline 3 (cubic)'] = lambda hg, image: transform(hg, image, -1, -1, 3)
algs['B-Spline 4'] = lambda hg, image: transform(hg, image, -1, -1, 4)
algs['B-Spline 5'] = lambda hg, image: transform(hg, image, -1, -1, 5)




for name, alg in algs.iteritems():
  print(name)
  
  rot_image = rotate_noop(image, transform = alg)
  rot_image = numpy.concatenate((rot_image['r'][:,:,numpy.newaxis], rot_image['g'][:,:,numpy.newaxis], rot_image['b'][:,:,numpy.newaxis]), axis=2)
  
  rot_image[rot_image<0.0] = 0.0
  rot_image[rot_image>255.0] = 255.0
  
  rot_image = (rot_image+0.5).astype(numpy.uint8)
  
  
  out_fn = os.path.splitext(fn)[0] + ' ' + name + '.png'
  imsave(out_fn, rot_image)
  