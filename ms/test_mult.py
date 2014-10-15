#! /usr/bin/env python

# Copyright 2013 Tom SF Haines

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



# Generate four complex probability maps - grids of random values, with spatial coherance - sort of a simulated anealing-like generation process, otherwise known as 'me screwing around until it looks good'...
size = (128, 128)
pmap = []

for _ in xrange(4):
  level = numpy.random.gamma(0.1, 80.0, size=size)
  
  steps = 32
  for s in xrange(steps):
    scale = level.copy()
    scale[1:,:] += level[:-1,:]
    scale[:-1,:] += level[1:,:]
    scale[:,1:] += level[:,:-1]
    scale[:,:-1] += level[:,1:]
    scale /= 5.0
    
    shape = (s+1)
    scale /= (s+1)
    level = numpy.random.gamma(shape, scale, size=size)
    
    if s%4==0: level[level<level.mean()] = level.mean() * 0.2
  
  level /= level.sum()
  pmap.append(level)



# Visualise the probability maps...
space = 16
repeat = 2

img = numpy.zeros((size[0]*2 + space, size[1]*2 + space), dtype=numpy.float32)
img[:size[0], :size[1]] = pmap[0]
img[-size[0]:, :size[1]] = pmap[1]
img[:size[0], -size[1]:] = pmap[2]
img[-size[0]:, -size[1]:] = pmap[3]

img *= 255.0 / img.max()
img = img.repeat(repeat, axis=0).repeat(repeat, axis=1)
img = array2cv(img)
cv.SaveImage('mult_input_maps.png', img)



# Visualise their multiplication - easy to do as they are just images...
img = pmap[0] * pmap[1] * pmap[2] * pmap[3]

img *= 255.0 / img.max()
img = img.repeat(2*repeat, axis=0).repeat(2*repeat, axis=1)
img = array2cv(img)
cv.SaveImage('mult_input_mult.png', img)



# Use the probability maps to draw samples, with added Gaussian noise, so we have some suitably complex distributions to push the multiplication system to its limits...
draw = 1024
samples = []

for i in xrange(4):
  sam = numpy.random.multinomial(draw, pmap[i].ravel())
  sam = numpy.repeat(numpy.arange(sam.shape[0]), sam)
  sam = numpy.unravel_index(sam, pmap[i].shape)
  sam = np.concatenate(map(lambda a: a.reshape((-1,1)), sam), axis=1)
  
  sam = sam.astype(numpy.float)
  sam[:,0] += numpy.random.normal(scale=0.5, size=sam.shape[0])
  sam[:,1] += numpy.random.normal(scale=0.5, size=sam.shape[0])
  
  samples.append(sam)



# Visualise the samples that have been drawn...
imgs = []

draw_scale = 4
for i in xrange(4):
  img = numpy.zeros((draw_scale*size[0], draw_scale*size[1]), dtype=numpy.float32)
  
  for pos in samples[i]:
    try:
      img[int(pos[0]*draw_scale), int(pos[1]*draw_scale)] = 1.0
    except:
      pass
  
  img *= 255.0 / img.max()
  imgs.append(img)

img = numpy.zeros((imgs[0].shape[0]*2 + draw_scale*space, imgs[0].shape[1]*2 + draw_scale*space), dtype=numpy.float32)

img[:imgs[0].shape[0],:imgs[0].shape[1]] = imgs[0]
img[-imgs[1].shape[0]:,:imgs[1].shape[1]] = imgs[1]
img[:imgs[2].shape[0],-imgs[2].shape[1]:] = imgs[2]
img[-imgs[3].shape[0]:,-imgs[3].shape[1]:] = imgs[3]

img = array2cv(img)
cv.SaveImage('mult_input_draw.png', img)
  
  

# Iterate and do each of the normal kernels in turn - we want to really dig into this...
kernels = ['gaussian', 'uniform', 'triangular', 'epanechnikov', 'cosine', 'cauchy']

for kernel in kernels:
  print 'Processing', kernel
  # Create the four MeanShift objects...
  def to_ms(data):
    ms = MeanShift()
    ms.set_data(data, 'df')
    ms.set_kernel(kernel)
    ms.set_spatial('kd_tree')
    ms.quality = 1.0
    return ms
  
  ms = map(to_ms, samples)

  # Infer a good loo value for the first one, then set them all to the same...
  p = ProgBar()
  ms[0].scale_loo_nll(callback=p.callback)
  del p
  
  for i in xrange(1,4): ms[i].copy_scale(ms[0])

  # Visualise the distributions using KDE...
  imgs = []
  p = ProgBar()
  for i in xrange(4):
    p.callback(i, 4)
    img = numpy.zeros((draw_scale*size[0], draw_scale*size[1]), dtype=numpy.float32)
    
    sweep0 = numpy.linspace(0, size[0], img.shape[0])
    sweep1 = numpy.linspace(0, size[1], img.shape[1])
    
    for ij, j in enumerate(sweep0):
      points = numpy.append(j * numpy.ones(sweep1.shape[0]).reshape((-1,1)), sweep1.reshape((-1,1)), axis=1)
      img[ij,:] = ms[i].probs(points)
    
    img *= 255.0 / img.max()
    imgs.append(img)
  del p
  
  img = numpy.zeros((imgs[0].shape[0]*2 + draw_scale*space, imgs[0].shape[1]*2 + draw_scale*space), dtype=numpy.float32)

  img[:imgs[0].shape[0],:imgs[0].shape[1]] = imgs[0]
  img[-imgs[1].shape[0]:,:imgs[1].shape[1]] = imgs[1]
  img[:imgs[2].shape[0],-imgs[2].shape[1]:] = imgs[2]
  img[-imgs[3].shape[0]:,-imgs[3].shape[1]:] = imgs[3]
  
  img = array2cv(img)
  cv.SaveImage('mult_k_%s_probs.png'%kernel, img)

  # Fake multiplication by multiplying the KDE images; visualise...
  img = imgs[0] * imgs[1] * imgs[2] * imgs[3]
  img *= 255.0 / img.max()
  
  img = array2cv(img)
  cv.SaveImage('mult_k_%s_probs_mult.png'%kernel, img)

  # Multiply them togther properly...
  p = ProgBar()
  output = numpy.empty((draw, 2), dtype=numpy.float32);
  for i in xrange(draw):
    p.callback(i, draw)
    MeanShift.mult(ms, output[i,:].reshape((1,-1)), fake=2)
  del p
  
  mult = to_ms(output)
  mult.copy_scale(ms[0])

  # Visualise the resulting distribution - the actual multiplication...
  img = numpy.zeros((draw_scale*size[0], draw_scale*size[1]), dtype=numpy.float32)
    
  sweep0 = numpy.linspace(0, size[0], img.shape[0])
  sweep1 = numpy.linspace(0, size[1], img.shape[1])

  for ij, j in enumerate(sweep0):
    points = numpy.append(j * numpy.ones(sweep1.shape[0]).reshape((-1,1)), sweep1.reshape((-1,1)), axis=1)
    img[ij,:] = mult.probs(points)
    
  img *= 255.0 / img.max()
  img = array2cv(img)
  cv.SaveImage('mult_k_%s_mult.png'%kernel, img)
