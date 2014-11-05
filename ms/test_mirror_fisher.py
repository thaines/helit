#! /usr/bin/env python

# Copyright 2014 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import random
import numpy
import numpy.random

import cv
from utils.cvarray import *
from utils.prog_bar import ProgBar

from ms import MeanShift



# Test the mirrored version of the von_mises Fisher distribution, this time in 5D...

# Create a dataset - just a bunch of points in one direction, so we can test the mirroring effect (Abuse MeanShift object to do this)...
print 'Mirrored draws:'

vec = numpy.array([1.0, 0.5, 0.0, -0.5, -1.0])
vec /= numpy.sqrt(numpy.square(vec).sum())

print 'Base dir =', vec

draw = MeanShift()
draw.set_data(vec, 'f')
draw.set_kernel('fisher(256.0)')

data = draw.draws(32)

#print 'Input:'
#print data



# Create a mean shift object from the draws, but this time with a mirror_fisher kernel...
mirror = MeanShift()
mirror.set_data(data, 'df')
mirror.set_kernel('mirror_fisher(64.0)')

resample = mirror.draws(16)

for row in resample:
  print '[%6.3f %6.3f %6.3f %6.3f %6.3f]' % tuple(row)
print



# Test probabilities by ploting them...
print 'Probability single 2D mirror Fisher:'
mirror = MeanShift()
mirror.set_data(numpy.array([1.0, 0.0]), 'f')
mirror.set_kernel('mirror_fisher(16.0)')

angles = numpy.linspace(0.0, numpy.pi*2.0, num=70, endpoint=False)
vecs = numpy.concatenate((numpy.cos(angles)[:,numpy.newaxis], numpy.sin(angles)[:,numpy.newaxis]),axis=1)
probs = mirror.probs(vecs)
probs /= probs.max()

steps = 12
for i in xrange(steps):
  threshold = 1.0 - (i+1)/float(steps)
  
  print ''.join(map(lambda p: '#' if p>threshold else ' ', probs))
print



# Test that mean shift still works; also test loo bandwidth estimation...
## Abuse the mean shift object to draw 8 uniform locations on a sphere...
usphere = MeanShift()
usphere.set_data(numpy.array([1.0, 0.0, 0.0]), 'f')
usphere.set_kernel('fisher(1e-6)') # Should be zero, but I don't support that.

centers = usphere.draws(8)
print('Distribution modes:')
print(centers)

## Use those modes to create a weighted mirror-fisher object, from which to draw lots of data...
weights = map(lambda x: 2.0 / (2.0+x), xrange(centers.shape[0]))
cext = numpy.concatenate((centers, numpy.array(weights)[:,numpy.newaxis]), axis=1)

wmf = MeanShift()
wmf.set_data(cext, 'df', 3)
wmf.set_kernel('mirror_fisher(24.0)')

## Create lots of data...
data = wmf.draws(1024)

swmf = MeanShift()
swmf.set_data(data, 'df')
swmf.set_kernel('mirror_fisher(128.0)')

## Get the indices of pixels into directions for a Mercator projection...
scale = 128
height = scale * 2
width = int(2.0 * numpy.pi * scale)

x_to_nx = numpy.cos(numpy.linspace(0.0, 2.0 * numpy.pi, width, False))
x_to_ny = numpy.sin(numpy.linspace(0.0, 2.0 * numpy.pi, width, False))
y_to_nz = numpy.linspace(-0.99, 0.99, height)

nx = x_to_nx.reshape((1,-1,1)).repeat(height, axis=0)
ny = x_to_ny.reshape((1,-1,1)).repeat(height, axis=0)
nz = y_to_nz.reshape((-1,1,1)).repeat(width, axis=1)

block = numpy.concatenate((nx, ny, nz), axis=2)
block[:,:,:2] *= numpy.sqrt(1.0 - numpy.square(y_to_nz)).reshape((-1,1,1))

## Visualise the probability of the data...
#print 'Calculating Mercator probability:'
#p = ProgBar()
#locs = block.reshape((-1,3))
#prob = numpy.empty(locs.shape[0], dtype=numpy.float32)
#step = locs.shape[0] / scale

#for i in xrange(scale):
  #p.callback(i, scale)
  #prob[i*step:(i+1)*step] = swmf.probs(locs[i*step:(i+1)*step,:])
#del p

#prob = prob.reshape((height, width))
#image = array2cv(255.0 * prob / prob.max())
#cv.SaveImage('mirror_fisher_mercator_kde.png', image)

### Apply mean shift and visualise the clustering...
#swmf.merge_range = 0.1
#modes, indices = swmf.cluster()

#print 'Meanshift clustering:'
#p = ProgBar()
#clusters = numpy.empty(locs.shape[0], dtype=numpy.int32)

#for i in xrange(scale):
  #p.callback(i, scale)
  #clusters[i*step:(i+1)*step] = swmf.assign_clusters(locs[i*step:(i+1)*step,:])
#del p

#clusters = clusters.reshape((height, width))
#image = numpy.zeros((height, width, 3), dtype=numpy.float32)

#for i in xrange(clusters.max()+1):
  #colour = numpy.random.random(3)
  #image[clusters==i,:] = colour.reshape((1,3))

#image = array2cv(255.0 * image)
#cv.SaveImage('mirror_fisher_mercator_ms.png', image)



# Helper function for below - visualises a distribution...
def visualise(fn, ms, pixels = 512):
  ang = numpy.linspace(-numpy.pi, numpy.pi, pixels*4)
  cos_ang = numpy.cos(ang)
  sin_ang = numpy.sin(ang)

  prob = ms.probs(numpy.concatenate((cos_ang[:,numpy.newaxis], sin_ang[:,numpy.newaxis]), axis=1))
  prob_max = prob.max()
  
  image = numpy.zeros((pixels, pixels, 3), dtype=numpy.float32)
  for y in xrange(pixels):
    ny = y / float(pixels-1)
    for x in xrange(pixels):
      nx = x / float(pixels-1)
    
      dist = numpy.sqrt((nx-0.5)**2 + (ny-0.5)**2) * 4.0
    
      if dist<0.2:
        image[y,x,:] = (0, 0, 0)
      
      elif dist<1:
        ang = numpy.arctan2(ny-0.5, nx-0.5)
        loc = prob.shape[0] * (ang + numpy.pi) / (2*numpy.pi)
      
        t = loc - int(loc)
        p = prob[int(loc)] * (1.0-t) + prob[(int(loc)+1)%prob.shape[0]] * t
      
        image[y,x,:] = (1, p / prob_max, 0.2)
      
      else:
        ang = numpy.arctan2(ny-0.5, nx-0.5)
        loc = prob.shape[0] * (ang + numpy.pi) / (2*numpy.pi)
      
        t = loc - int(loc)
        p = prob[int(loc)] * (1.0-t) + prob[(int(loc)+1)%prob.shape[0]] * t
      
        dp = prob_max * (dist - 1)
      
        if p<dp:
          image[y,x,:] = (1, 0, 0)

  image = array2cv(255.0 * image)
  cv.SaveImage(fn, image)



# Try out multiplication, and throughroughly verify it works; for more fun include weighting...

## Use weighting to create a mirrored multi-modal distribution on angles...
ang = numpy.linspace(-numpy.pi*0.5, numpy.pi*0.5, 128)
cos_ang = numpy.cos(ang)
sin_ang = numpy.sin(ang)
weight = numpy.fabs(numpy.sin(ang*8.0) * (1.0 - (numpy.fabs(ang) / (0.5*numpy.pi))))

mult_a = MeanShift()
mult_a.quality = 1.0
mult_a.set_data(numpy.concatenate((cos_ang[:,numpy.newaxis], sin_ang[:,numpy.newaxis], weight[:,numpy.newaxis]), axis=1), 'df', 2)
mult_a.set_kernel('mirror_fisher(128.0)')

visualise('mirror_fisher_mult_a.png', mult_a)
print 'Prepared and visualised distribution A'


## Create another distribution, designed to create a funky effect when combined with the first...
ang = numpy.linspace(-numpy.pi, 0.0, 64)
cos_ang = numpy.cos(ang)
sin_ang = numpy.sin(ang)
weight = numpy.fabs(numpy.sin(ang*3.0) * (1.0 - ((numpy.fabs(ang)+0.5*numpy.pi) / (0.5*numpy.pi))))

mult_b = MeanShift()
mult_b.set_data(numpy.concatenate((cos_ang[:,numpy.newaxis], sin_ang[:,numpy.newaxis], weight[:,numpy.newaxis]), axis=1), 'df', 2)
mult_b.set_kernel('mirror_fisher(2048.0)')

visualise('mirror_fisher_mult_b.png', mult_b)
print 'Prepared and visualised distribution B'


## A borring distribution, without weighting to test that works...
mult_c = MeanShift()
mult_c.set_data(numpy.array([[1.0,0.0], [0.0,1.0]]), 'df')
mult_c.set_kernel('mirror_fisher(32.0)')

visualise('mirror_fisher_mult_c.png', mult_c)
print 'Prepared and visualised distribution C'


## Multiply a and b distributions and visualise...
count = 256
draws = numpy.empty((count,2), dtype=numpy.float32)
MeanShift.mult([mult_a, mult_b], draws)

prod_a_b = MeanShift()
prod_a_b.set_data(draws, 'df')
prod_a_b.set_kernel('mirror_fisher(512.0)')

visualise('mirror_fisher_prod_a_b.png', prod_a_b)
print 'Prepared and visualised product of a and b'


## Multiply b and c distributions and visualise...
draws = numpy.empty((count,2))
MeanShift.mult((mult_b, mult_c), draws)

prod_b_c = MeanShift()
prod_b_c.set_data(draws, 'df')
prod_b_c.copy_all(prod_a_b)

visualise('mirror_fisher_prod_b_c.png', prod_b_c)
print 'Prepared and visualised product of b and c'


## Multiply c and a distributions and visualise...
## This doesn't work - equal sampling of initial state, where one is ultimatly much more probable but forming two islands that could require millions of sampling steps to transfer between, so it ends up with the islands having equal probability when they really shouldn't...
draws = numpy.empty((count,2))
MeanShift.mult((mult_c, mult_a), draws)

prod_c_a = MeanShift()
prod_c_a.set_data(draws, 'df')
prod_c_a.copy_all(prod_a_b)

visualise('mirror_fisher_prod_c_a_wrong.png', prod_c_a)
print 'Prepared and visualised product of c and a'


# Test out the memory breakdown method...
print
print 'Memory breakdown of product of c and a:'
mem = prod_c_a.memory()

for key, value in mem.iteritems():
  if key=='kernel_ref_count' or key=='total':
    continue
  
  if key=='kernel':
    print '  %s: %i bytes (ref count = %i)' % (key, value, mem['kernel_ref_count'])
  else:
    print '  %s: %i bytes' % (key, value)

print 'total = %i bytes' % mem['total']
print
