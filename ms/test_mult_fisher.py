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



# An all-in type test - has multiplication with composition of Fisher and Gaussian distributions using a simulation of camera information, in 2D - its basically a straight test of the key functionality required for nonparameteric belief propagation with crazy mixed distributions...



# Define the distributions to obtain the data from...
cameras = [(128, 0.9, 1.05, 0.0), # samples, x, y, direction
           (64, 0.95, 1.0, -0.1),
           (64, 1.05, 0.95, 0.2),
           (32, 1.1, 1.0, -0.1)]

sd_x = 0.1
sd_y = 0.2
conc = 16.0



scale = 2.0
size = 512
angle_len = 0.2
angle_step = 64



# Draw 4 data sets...
data = []

for camera in cameras:
  direction = numpy.random.vonmises(camera[3], conc, size=camera[0])
  x = numpy.random.normal(camera[1], sd_x, size=camera[0])
  y = numpy.random.normal(camera[2], sd_y, size=camera[0])
  
  block = numpy.concatenate((x.reshape((-1,1)), y.reshape((-1,1)), numpy.cos(direction).reshape((-1,1)), numpy.sin(direction).reshape((-1,1))), axis=1)
  data.append(block)



# Construct the mean shift object from it, including a composite kernel...
kde = []
for ds in data:
  ms = MeanShift()
  ms.set_data(ds, 'df')
  if len(kde)==0:
    ms.set_kernel('composite(2:gaussian,2:fisher(32.0))')
    ms.set_spatial('kd_tree')
    ms.set_scale(numpy.array([10.0,5.0,1.0,1.0]))
    ms.merge_range = 0.05
  else:
    ms.copy_all(kde[0])
    
  kde.append(ms)



# Visualise the data set...
for ind, ds in enumerate(data):
  img = numpy.zeros((size, size, 3), dtype=numpy.float32)

  for sample in ds:
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
  cv.SaveImage('mult_fisher_input_%i.png'%ind, img)



# Multiply into a new MeanShift object - we are a little perverse in our ordering to test the reset method...
ms = MeanShift()
ms.set_data(numpy.zeros((64, 4), numpy.float32), 'df')
ms.copy_all(kde[0])

print 'Multiplying...'
p = ProgBar()
MeanShift.mult(kde, ms.get_dm())
del p

ms.reset()



# Visualise the samples - include the mode(s) in red...
img = numpy.zeros((size, size, 3), dtype=numpy.float32)

for sample in ms.get_dm():
  s_x = (size-1) * sample[1] / scale
  s_y = (size-1) * sample[0] / scale
  e_x = (size-1) * (sample[1] + angle_len * sample[3]) / scale
  e_y = (size-1) * (sample[0] + angle_len * sample[2]) / scale
  
  for i in xrange(angle_step):
    try:
      t = float(i) / (angle_step-1)
      t_x = int(t * s_x + (1-t) * e_x)
      t_y = int(t * s_y + (1-t) * e_y)
      try:
        if img[t_y,t_x,0] < t*0.666:
          img[t_y,t_x,:] = t*0.666
      except:
        pass
    except ValueError:
      print 'Nan:-('

modes, _ = ms.cluster()

for ii, sample in enumerate(modes):
  print 'mode %i: position = (%.3f, %.3f), direction = (%.3f,%.3f)' % (ii, sample[0], sample[1], sample[2], sample[3])
  s_x = (size-1) * sample[1] / scale
  s_y = (size-1) * sample[0] / scale
  e_x = (size-1) * (sample[1] + angle_len * sample[3]) / scale
  e_y = (size-1) * (sample[0] + angle_len * sample[2]) / scale
  
  
  for i in xrange(angle_step):
    try:
      t = float(i) / (angle_step-1)
      t_x = int(t * s_x + (1-t) * e_x)
      t_y = int(t * s_y + (1-t) * e_y)
      try:
        if img[t_y,t_x,2] < t:
          img[t_y,t_x,2] = t
      except:
        pass
    except ValueError:
      print 'Nan:-('

img = array2cv(255.0 * img)
cv.SaveImage('mult_fisher_mult.png', img)
