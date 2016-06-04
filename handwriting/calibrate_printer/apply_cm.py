#! /usr/bin/env python

# Copyright 2016 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import argparse
import os.path

import numpy
from scipy.misc import imread, toimage

from ply2.ply2 import read

from misc.tps import TPS
from utils.prog_bar import ProgBar



# Parse the command line arguments to work out what we are doing...
parser = argparse.ArgumentParser(description='Apply a colour map to an image.')

parser.add_argument('-m', '--mask', help='If included save a map file (Same file name as the output, with _mask inserted) that indicates where the colour map has exceded the dynamic range of the output.', action='store_true')
parser.add_argument('-r', '--reverse', help='Reverses the direction of the colour map.', action='store_true')
parser.add_argument('-s', '--smooth', help='Smoothing parameter for thin plate splines.', type=float, default=1e-3)
parser.add_argument('-b', '--block_size', help='Size of the block sent in for each step, in megabytes. Compromise between speed and reporting progress.', type=int, default=256)

parser.add_argument('-t', '--test', help='Does some basic verification that its working, printing out queries and known answers.', action='store_true')
parser.add_argument('-q', '--quiet', help='Makes the program shut up about what it is doing.', action='store_true')

parser.add_argument('in_file', help='Input image file, to be remapped.')
parser.add_argument('map_file', help='File that contains the colour map.')
parser.add_argument('out_file', help='Output image file; if omitted defaults to <in_file (no extension)>_<map_file (no extension)>.<in_file extension>.', default='', nargs='?')


args = parser.parse_args()


in_fn, in_ext = os.path.splitext(args.in_file)
cm_fn, cm_ext = os.path.splitext(os.path.basename(args.map_file))
  
if args.out_file=='':
  args.out_file = '%s_%s%s'%(in_fn, cm_fn, in_ext)

args.mask_file = '%s_%s_mask%s'%(in_fn, cm_fn, in_ext)



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



# Load the input image...
if not args.quiet:
  print 'Loading input image... (%s)'%args.in_file


image = imread(args.in_file).astype(numpy.float32)
if image.shape[2]>3:
  image = image[:,:,:3] # No alpha please



# Build a model from the colour map using a thin plate spline...
model = [None] * 3

for cc in xrange(3):
  if not args.quiet: print 'Building thin plate spline model - %s...'%(['red', 'green', 'blue'][cc])
  
  m = TPS(3, args.smooth)
  m.learn(col_in, col_out[:,cc])
  model[cc] = m



# Testing code...
if args.test:
  print 'Testing...'
  diff = numpy.empty(col_in.shape[0], dtype=numpy.float32)
  
  for i in xrange(col_in.shape[0]):
    test_out = numpy.empty(3, dtype=numpy.float32)
    for cc in xrange(3):
      test_out[cc] = model[cc](col_in[i,:])
    diff[i] = numpy.sqrt(numpy.square(test_out - col_out[i,:]).sum())
  
  print 'Error: min = %.3f; mean = %.3f; max = %.3f'%(diff.min(), numpy.mean(diff), diff.max())



# Convert the entire image - vectorise as much as possible...
## Convert image to data matrix...
data = image.reshape((-1, 3))

## Compress the data matrix down, to remove duplicates...
index = numpy.lexsort(data.T)
data = data[index,:]
keep = numpy.ones(data.shape[0], dtype=numpy.bool)
keep[1:] = (numpy.diff(data, axis=0)!=0).any(axis=1)
data = data[keep]

# Block the system off, so we are not trying to create gigabyte sized buffers...
step = ((args.block_size*1024*1024) // (col_in.shape[0] * 4)) + 1
if not args.quiet:
  print 'Converting... (%i pixels at a time)'%step

slices = map(lambda x: slice(x*step, (x+1)*step), xrange(data.shape[0]//step + 1))

if slices[-1].stop<data.shape[0]:
  slices.append(slice(slices[-1].stop, data.shape[0]))

## Calculate each channel in turn...
out = data.copy()

for cc in xrange(3):
  if not args.quiet:
    print 'Converting - %s...'%(['red', 'green', 'blue'][cc])
  p = ProgBar()
  for i,s in enumerate(slices):
    p.callback(i, len(slices))
    out[s,cc] = model[cc](data[s,:].astype(numpy.float32))
  del p

## Expand the data matrix back up to the order and length of the image, by expanding duplicates...
source = numpy.cumsum(keep) - 1
out = out[source,:]
out = out[numpy.argsort(index),:]
  
## Convert back from data matrix to image...
out = out.reshape(image.shape)



# Clamp unreasonable values, record where clamping occurs...
if not args.quiet:
  print 'Clamping...'
mask = numpy.zeros((out.shape[0], out.shape[1]), dtype=numpy.bool)

low = out < 0.0
out[low] = 0.0
mask[:,:] = numpy.logical_or(mask, low.any(axis=2))

high = out > 255.0
out[high] = 255.0
mask[:,:] = numpy.logical_or(mask, high.any(axis=2))

if not args.quiet:
  print '%.1f%% of image has been clamped'%(100.0*mask.sum()/float(out.shape[0]*out.shape[1]))


# Save the output image to disk...
if not args.quiet:
  print 'Saving output image... (%s)'%args.out_file

toimage(out, cmin=0, cmax=255).save(args.out_file)



# If requested save the error mask, to indicate where the printer can not do its job...
if args.mask:
  if not args.quiet:
    print 'Saving clamping mask... (%s)'%args.mask_file
  
  mask_col = numpy.empty((mask.shape[0], mask.shape[1], 3), dtype=numpy.uint8)
  mask_col[:,:,:] = 255
  mask_col[mask,:] = 0
  
  toimage(mask_col, cmin=0, cmax=255).save(args.mask_file)
