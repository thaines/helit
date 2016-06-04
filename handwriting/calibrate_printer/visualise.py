#! /usr/bin/env python

# Copyright 2016 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

# 

import argparse
import os.path

import numpy
from scipy.misc import toimage

from ply2.ply2 import read



# Parse the command line arguments to work out what we are doing...
parser = argparse.ArgumentParser(description='Visualises a colour map - just a very tall image with swatches of the input colour on the left, output on the right.')

parser.add_argument('in_file', help='Colour map file.')
parser.add_argument('out_file', help='Output image file; if omitted defaults to <in_file (no extension)>.png.', default='', nargs='?')


args = parser.parse_args()


if args.out_file=='':
  args.out_file = '%s.png' % os.path.splitext(args.in_file)[0]



# Load the colour map...
f = open(args.in_file, 'r')
cm = read(f)
f.close()

elem = cm['element']['sample']



# Build image...
image = numpy.empty((elem['in.r'].shape[0], 2, 3), dtype=numpy.float32)

image[:,0,0] = elem['in.r']
image[:,0,1] = elem['in.g']
image[:,0,2] = elem['in.b']
image[:,1,0] = elem['out.r']
image[:,1,1] = elem['out.g']
image[:,1,2] = elem['out.b']



# Save image...
image = numpy.repeat(image, 32, axis=0)
image = numpy.repeat(image, 32, axis=1)

toimage(image, cmin=0, cmax=255).save(args.out_file)
