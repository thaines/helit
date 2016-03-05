#! /usr/bin/env python
# Copyright 2015 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import os.path
import argparse
import time

import numpy
import scipy.misc

from gbp import GBP



# Command line options...
parser = argparse.ArgumentParser(description='Removes the curl from a normal map using GBP.')

parser.add_argument('-p', '--plane', help='Precision of the unary term of each pixel that is pulling the answer towards a plane with height 0.', type=float, default=1.0/32.0)
parser.add_argument('-n', '--normal', help='Precision of the pairwise term between pixels that is pulling the answer to match the normals.', type=float, default=1.0/0.1)

parser.add_argument('-e', '--epsilon', help='Stopping condition is when the biggest absolute model parameter change is less than this. Defaults to 1e-4, which is more than enough for a normal map encoded as an 8 bot image.', type=float, default=1e-4)
parser.add_argument('-r', '--report', help='How often to report progress, in iterations. Defaults to 16.', type=int, default=16)

parser.add_argument('input', help='The input normal map to correct.')
parser.add_argument('output', help='The output normal map after correction - if not provided it defaults to <input>_uncurl.png', default='', nargs='?')

args = parser.parse_args()

if args.output=='':
  args.output = os.path.splitext(args.input)[0] + '_uncurl.png'



# Open the image...
image = scipy.misc.imread(args.input)



# Make floaty and normalise to handle quantisation error from the image encoding...
image = image.astype(numpy.float32) - 127.0

div = numpy.sqrt(numpy.square(image).sum(axis=2))
image /= div[:,:,numpy.newaxis]



# Setup a GBP object with a weak desire for each pixel to be on the plane...
solver = GBP(image.shape[0] * image.shape[1])
solver.unary(slice(None), 0.0, args.plane)



# Convert normals to gradients...
image[:,:,:2] /= image[:,:,2,numpy.newaxis]
grad_x = -image[:,:,0]
grad_y = image[:,:,1]



# Add pairwise terms to GBP solver, using the averages of adjacent gradients...
for y in xrange(image.shape[0]):
  g = 0.5 * (grad_x[y,:-1] + grad_x[y,1:])
  w = args.normal
  
  base = y * image.shape[1]
  solver.pairwise(slice(base, base+image.shape[1]-1), slice(base+1, base+image.shape[1]), g, w)

for x in xrange(image.shape[1]):
  g = 0.5 * (grad_y[:-1,x] + grad_y[1:,x])
  w = args.normal
  
  solver.pairwise(slice(x, x+(image.shape[0]-2)*image.shape[1], image.shape[1]), slice(x+image.shape[1], x+(image.shape[0]-1)*image.shape[1], image.shape[1]), g, w)



# Solve...
print 'Solving...'
iters = 0
start = time.clock()

while True:
  it = solver.solve_trws(args.report, args.epsilon)
  iters += it
  print('       %i iters, delta = %f (target = %f)' % (iters, solver.last_delta, args.epsilon))
  if it!=args.report:
    break

end = time.clock()
print('...solved in %.1f seconds' % (end - start))



# Convert back to a normal field...
height, _ = solver.result()
height = height.reshape((image.shape[0], image.shape[1]))

grad_x = height.copy()
grad_x[:,0] = height[:,1] - height[:,0]
grad_x[:,1:-1] = 0.5 * (height[:,2:] - height[:,:-2])
grad_x[:,-1] = height[:,-1] - height[:,-2]
grad_x *= -1.0

grad_y = height.copy()
grad_y[0,:] = height[1,:] - height[0,:]
grad_y[1:-1,:] = 0.5 * (height[2:,:] - height[:-2,:])
grad_y[-1,:] = height[-1,:] - height[-2,:]

norm = numpy.concatenate((grad_x[:,:,numpy.newaxis], grad_y[:,:,numpy.newaxis], numpy.ones((height.shape[0], height.shape[1], 1), dtype=numpy.float32)), axis=2)

div = numpy.sqrt(numpy.square(norm).sum(axis=2))
norm /= div[:,:,numpy.newaxis]



# Convert to an image and save...
scipy.misc.toimage(127.0*norm + 127.0, cmin=0, cmax=255).save(args.output)
