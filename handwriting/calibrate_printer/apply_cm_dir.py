#! /usr/bin/env python

# Copyright 2016 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import argparse
import os.path
import errno

import numpy
from scipy.misc import imread, toimage

from ply2.ply2 import read

from misc.tps import TPS
from utils.prog_bar import ProgBar



# Parse the command line arguments to work out what we are doing...
parser = argparse.ArgumentParser(description='Apply a colour map to every image in a  directory.')

parser.add_argument('-r', '--reverse', help='Reverses the direction of the colour map.', action='store_true')
parser.add_argument('-s', '--smooth', help='Smoothing parameter for thin plate splines.', type=float, default=1e-3)
parser.add_argument('-b', '--block_size', help='Size of the block sent in for each step, in megabytes.', type=int, default=256)

parser.add_argument('-p', '--pad', help='Makes the program pad all images to have the same height.', action='store_true')

parser.add_argument('-q', '--quiet', help='Makes the program shut up about what it is doing.', action='store_true')

parser.add_argument('map_file', help='File that contains the colour map.')
parser.add_argument('directory', help='Directory of images to be remapped - creates a subdirectory with the name of the colour map to contain the output files.')
args = parser.parse_args()



# Load the colour map...
if not args.quiet:
  print 'Loading colour map... (%s)' % args.map_file

f = open(args.map_file, 'r')
cm = read(f)
f.close()

col_in = numpy.empty((cm['element']['sample']['in.r'].shape[0],3), dtype=numpy.float32)
col_out = numpy.empty((cm['element']['sample']['out.r'].shape[0],3), dtype=numpy.float32)

col_in[:,0] = cm['element']['sample']['in.r']
col_in[:,1] = cm['element']['sample']['in.g']
col_in[:,2] = cm['element']['sample']['in.b']

col_out[:,0] = cm['element']['sample']['out.r']
col_out[:,1] = cm['element']['sample']['out.g']
col_out[:,2] = cm['element']['sample']['out.b']


if args.reverse:
  temp = col_in
  col_in = col_out
  col_out = temp



# Build a model from the colour map using a thin plate spline...
model = [None] * 3

for cc in xrange(3):
  if not args.quiet: print 'Building thin plate spline model - %s...'%(['red', 'green', 'blue'][cc])
  
  m = TPS(3, args.smooth)
  m.learn(col_in, col_out[:,cc])
  model[cc] = m



# Create the output directory...
out_dir = os.path.join(args.directory, args.map_file.rsplit('.', 1)[0])
try:
  os.makedirs(out_dir)
except OSError as e:
  if e.errno != errno.EEXIST:
    raise



# If we are padding all images to have the same height then need to find out what the maximum height is...
if args.pad:
  pad_height = 1
  
  for fn in os.listdir(args.directory):
    full_fn = os.path.join(args.directory, fn)
    if not os.path.isfile(full_fn):
      continue
    if not fn.endswith('.png'):
      continue
    
    image = imread(args.in_file)
    
    if image.shape[0]>pad_height:
      pad_height = image.shape[0]
  
  if not args.quiet:
    print 'Padding all images to %i pixels high' % pad_height



# Find all images in the input directory - loop them...
for fn in os.listdir(args.directory):
  full_fn = os.path.join(args.directory, fn)
  if not os.path.isfile(full_fn):
    continue
  if not fn.endswith('.png'):
    continue
  
  if not args.quiet:
    print 'Processing:', fn
  
  # Load the file...
  image = imread(fn).astype(numpy.float32)
  if image.shape[2]>3:
    image = image[:,:,:3] # No alpha please
  
  # Apply the conversion...
  ## Convert image to data matrix...
  data = image.reshape((-1, 3))

  ## Compress the data matrix down, to remove duplicates...
  index = numpy.lexsort(data.T)
  data = data[index,:]
  keep = numpy.ones(data.shape[0], dtype=numpy.bool)
  keep[1:] = (numpy.diff(data, axis=0)!=0).any(axis=1)
  data = data[keep]

  ## Block the system off, so we are not trying to create gigabyte sized buffers...
  step = ((args.block_size*1024*1024) // (col_in.shape[0] * 4)) + 1
  slices = map(lambda x: slice(x*step, (x+1)*step), xrange(data.shape[0]//step + 1))
  if slices[-1].stop<data.shape[0]:
    slices.append(slice(slices[-1].stop, data.shape[0]))

  ## Calculate each channel in turn...
  out = data.copy()

  for cc in xrange(3):
    for i,s in enumerate(slices):
      out[s,cc] = model[cc](data[s,:].astype(numpy.float32))

  ## Expand the data matrix back up to the order and length of the image, by expanding duplicates...
  source = numpy.cumsum(keep) - 1
  out = out[source,:]
  out = out[numpy.argsort(index),:]
  
  ## Convert back from data matrix to image...
  out = out.reshape(image.shape)
  
  ## Clamp out of range values...
  out = numpy.clip(out, 0.0, 255.0)
  
  # Apply padding as required...
  if args.pad:
    amount = pad_height - out.shape[0]
    if amount!=0:
      extra = 255.0 * numpy.ones((amount, out.shape[1], 3), dtype=numpy.float32)
      out = numpy.concatenate((out, extra), axis=0)
  
  # Save it out...
  out_fn = os.path.join(out_dir, fn)
  toimage(out, cmin=0, cmax=255).save(out_fn)
