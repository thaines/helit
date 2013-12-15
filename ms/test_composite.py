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



# Below demonstrates a composite kernel using 'cameras' in 2D space, so each has a 2D location and 2D orientation, represented by a unit vector...



# Define the distributions to obtain the data from...
cameras = [(64, 1.0, 2.0, 0.0), # samples, x, y, direction
           (64, 3.0, 1.0, numpy.pi * 0.5),
           (64, 2.0, 4.0, numpy.pi),
           (32, 4.0, 3.0, -numpy.pi*0.5)]

sd_x = 0.1
sd_y = 0.2
conc = 16.0



scale = 5.0
size = 512
angle_len = 0.2
angle_step = 32



# Draw a data set...
data = []

for camera in cameras:
  direction = numpy.random.vonmises(camera[3], conc, size=camera[0])
  x = numpy.random.normal(camera[1], sd_x, size=camera[0])
  y = numpy.random.normal(camera[2], sd_y, size=camera[0])
  
  block = numpy.concatenate((x.reshape((-1,1)), y.reshape((-1,1)), numpy.cos(direction).reshape((-1,1)), numpy.sin(direction).reshape((-1,1))), axis=1)
  data.append(block)
data = numpy.concatenate(data, axis=0)



# Construct the mean shift object from it, including a composite kernel...
ms = MeanShift()
ms.set_data(data, 'df')
ms.set_kernel('composite(2:gaussian,2:fisher(32.0))')
ms.set_spatial('kd_tree')
ms.set_scale(numpy.array([10.0,5.0,1.0,1.0]))
ms.merge_range = 0.05



# Print out information in a convoluted way to test some convoluted features!..
ms2 = MeanShift()
ms2.copy_kernel(ms)
print 'Kernel:', ms2.get_kernel()
del ms2



# For our first trick visualise the data set...
img = numpy.zeros((size, size, 3), dtype=numpy.float32)

for sample in data:
  s_x = (size-1) * sample[1] / scale
  s_y = (size-1) * sample[0] / scale
  e_x = (size-1) * (sample[1] + angle_len * sample[3]) / scale
  e_y = (size-1) * (sample[0] + angle_len * sample[2]) / scale
  
  for i in xrange(angle_step):
    t = float(i) / (angle_step-1)
    t_x = int(t * s_x + (1-t) * e_x)
    t_y = int(t * s_y + (1-t) * e_y)
    try:
      if img[t_y,t_x,0] < t:
        img[t_y,t_x,:] = t
    except:
      pass

img = array2cv(255.0 * img)
cv.SaveImage('composite_input.png', img)



# Now draw the same number of cameras again and visualise so a fleshy can check they are similar...
draw = ms.draws(data.shape[0], 0)

img = numpy.zeros((size, size, 3), dtype=numpy.float32)

for sample in draw:
  s_x = (size-1) * sample[1] / scale
  s_y = (size-1) * sample[0] / scale
  e_x = (size-1) * (sample[1] + angle_len * sample[3]) / scale
  e_y = (size-1) * (sample[0] + angle_len * sample[2]) / scale
  
  
  for i in xrange(angle_step):
    t = float(i) / (angle_step-1)
    t_x = int(t * s_x + (1-t) * e_x)
    t_y = int(t * s_y + (1-t) * e_y)
    try:
      if img[t_y,t_x,0] < t:
        img[t_y,t_x,:] = t
    except:
      pass

img = array2cv(255.0 * img)
cv.SaveImage('composite_draw.png', img)



# Visualise the probability - both spatial and rotational in a single image, with one colour channel each for 3 directions...
img = numpy.zeros((size, size, 3), dtype=numpy.float32)
p = ProgBar()

for y in xrange(size):
  p.callback(y, size)
  
  for index, orient_x, orient_y in [(0,1.0,0.0), (1,0.0,1.0), (2,-1.0,0.0)]:
    block = numpy.concatenate(((scale * y / float(size-1)) * numpy.ones(size).reshape((-1,1)), numpy.linspace(0.0, scale, size).reshape((-1,1)), orient_x * numpy.ones(size).reshape((-1,1)), orient_y * numpy.ones(size).reshape((-1,1))), axis=1)
    
    vals = ms.probs(block)
    img[y,:,index] = vals

del p

img *= 255 / img.max()
img = array2cv(img)
cv.SaveImage('composite_prob.png', img)



# Extract and visualise the modes of the data set...
modes, indices = ms.cluster()

img = numpy.zeros((size, size, 3), dtype=numpy.float32)

for sample in modes:
  s_x = (size-1) * sample[1] / scale
  s_y = (size-1) * sample[0] / scale
  e_x = (size-1) * (sample[1] + angle_len * sample[3]) / scale
  e_y = (size-1) * (sample[0] + angle_len * sample[2]) / scale
  
  
  for i in xrange(angle_step):
    t = float(i) / (angle_step-1)
    t_x = int(t * s_x + (1-t) * e_x)
    t_y = int(t * s_y + (1-t) * e_y)
    try:
      if img[t_y,t_x,0] < t:
        img[t_y,t_x,:] = t
    except:
      pass

img = array2cv(255.0 * img)
cv.SaveImage('composite_modes.png', img)



# Visualise the mean shift clustering result...
colours = []
for i in xrange(modes.shape[0]):
  colours.append(numpy.random.random(3))

img = numpy.zeros((size, size, 3), dtype=numpy.float32)

for i in xrange(indices.shape[0]):
  sample = data[i,:]
  colour = colours[indices[i]]
  
  s_x = (size-1) * sample[1] / scale
  s_y = (size-1) * sample[0] / scale
  e_x = (size-1) * (sample[1] + angle_len * sample[3]) / scale
  e_y = (size-1) * (sample[0] + angle_len * sample[2]) / scale
  
  
  for i in xrange(angle_step):
    t = float(i) / (angle_step-1)
    t_x = int(t * s_x + (1-t) * e_x)
    t_y = int(t * s_y + (1-t) * e_y)
    try:
      for j in xrange(3):
        if img[t_y,t_x,j] < t*colour[j]:
          img[t_y,t_x,j] = t*colour[j]
    except:
      pass

img = array2cv(255.0 * img)
cv.SaveImage('composite_clusters.png', img)
