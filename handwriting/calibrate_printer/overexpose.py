#! /usr/bin/env python

# Copyright 2016 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import argparse
import os.path

import numpy
from scipy.misc import imread, toimage



# Parse the command line arguments to work out what we are doing...
parser = argparse.ArgumentParser(description='Scale the brightness of an image so at least 50% of the pixels are pure white. A useful little tool for quickly visualising the darker areas of an image.')

parser.add_argument('-q', '--quiet', help='Makes the program shut up about what it is doing.', action='store_true')

parser.add_argument('in_file', help='Input image file, to be overexposed.')
parser.add_argument('out_file', help='Output image file; if omitted defaults to <in_file (no extension)>_overexposed.<in_file extension>.', default='', nargs='?')


args = parser.parse_args()


if args.out_file=='':
  in_fn, in_ext = os.path.splitext(args.in_file)
  args.out_file = '%s_overexposed%s'%(in_fn, in_ext)



# Load the input image...
if not args.quiet:
  print 'Loading input image... (%s)' % args.in_file

image = imread(args.in_file).astype(numpy.float32)
if image.shape[2]>3:
  image = image_in[:,:,:3] # No alpha please



# Extract the image brightness values, and sort them from brightest to dimmest...
if not args.quiet:
  print 'Calculating function...'

lum = image.reshape((-1, 3)).sum(axis=1) / 3.0
lum = numpy.sort(lum)

value = lum[lum.shape[0]//2]
mult = 256.0 / value # 256 just to bias it a bit.

if not args.quiet:
  print 'Value to scale to 255: %.2f '% value



# Apply the function...
image *= mult
image[image>255] = 255



# Save the output image...
if not args.quiet:
  print 'Saving output image... (%s)' % args.out_file

toimage(image, cmin=0.0, cmax=255.0).save(args.out_file)
