#! /usr/bin/env python

# Copyright 2016 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

# Simple script for generating a calibration target, to be printed then scanned, to calibrate the printer relative to the scanner.

import numpy
from scipy.misc import toimage



# First create the matrix of colours as a single pixel for each one...
steps = 8
pixels = 83*30 # 8.3 inches by 300dpi
square_size = pixels // (steps*3+4)

cols = numpy.ones((steps*3+2 ,steps*3+2, 3), dtype=numpy.float32)

for ri, r in map(lambda ri: (ri, ri / float(steps-1)), xrange(steps)):
  for gi, g in map(lambda gi: (gi, gi / float(steps-1)), xrange(steps)):
    for bi, b in map(lambda bi: (bi, bi/ float(steps-1)), xrange(steps)):
      ci = ri if ri<4 else ri+1
      xi = (steps+1) * (ci%3)
      yi = (steps+1) * (ci//3)
      
      cols[yi+bi,xi+gi,:] = [b,g,r]

for wi, w in map(lambda wi: (wi, wi / float(steps*steps-1)), xrange(steps*steps)):
  xi = steps + 1 + (wi//steps)
  yi = steps + 1 + (wi%steps)
  
  cols[xi,yi,:] = w



# Convert the colour matrix into an actual grid, with a decent structure - aim for 300dpi A4 size...
edges = numpy.zeros((steps*3+2, 3), dtype=numpy.float32)
edges[steps,:] = 1.0
edges[steps*2+1,:] = 1.0

edgesBig = numpy.zeros(((steps*3+2)*(square_size+1) + 1, 3), dtype=numpy.float32)
for i in xrange(square_size):
  edgesBig[(steps)*(square_size+1)+1+i,:] = 1.0
  edgesBig[(steps*2+1)*(square_size+1)+1+i,:] = 1.0


def add_edges(arr, big = False):
  yield edges if not big else edgesBig
  for a in arr:
    for _ in xrange(square_size): yield a
    yield edges if not big else edgesBig


cols_array = map(lambda i: cols[:,i,:], xrange(cols.shape[1]))
cols_array = [x.reshape(1,x.shape[0], 3) for x in add_edges(cols_array)]
cols = numpy.concatenate(cols_array, axis=0)

cols_array = map(lambda i: cols[:,i,:], xrange(cols.shape[1]))
cols_array = [x.reshape(x.shape[0], 1, 3) for x in add_edges(cols_array, True)]
cols = numpy.concatenate(cols_array, axis=1)



# Save as a png file...
toimage(cols*255.0, cmin=0.0, cmax=255.0).save('calibration_target.png')

