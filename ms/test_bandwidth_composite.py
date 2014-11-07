#! /usr/bin/env python

# Copyright 2014 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import numpy
import numpy.random

import cv
from utils.cvarray import *

from ms import MeanShift, MeanShiftCompositeScale



# Parameters...
samples = 4096
size = 1024
scale = 4.0
angle_step = 64
angle_len = 0.25 * (0.5 * size / scale)
mirror_len = 0.15 * (0.5 * size / scale)



# Create a mean shift object with some data and a composite kernel - data is a position using radial coordinates followed by two angles; also using this for testing the conversion system...
print 'a:'

direction = numpy.concatenate((numpy.random.normal(0.0, 0.1, samples//2), numpy.random.normal(numpy.pi, 1.0, samples//2)))
radius = numpy.concatenate((numpy.random.normal(2.0, 0.5, samples//2), numpy.random.normal(2.0, 0.5, samples//4), numpy.random.normal(3.0, 0.5, samples//4)))
ang_a = numpy.concatenate((numpy.random.normal(0.0, 0.3, samples//4), numpy.random.normal(numpy.pi, 0.3, samples//4), numpy.random.normal(0.5*numpy.pi, 0.3, samples//4), numpy.random.normal(1.5*numpy.pi, 0.3, samples//4)))
ang_b = numpy.concatenate((numpy.random.normal(0.0, 0.6, samples//4), numpy.random.normal(0.5*numpy.pi, 0.5, samples//2), numpy.random.normal(0.0, 0.6, samples//4)))

data = numpy.concatenate((direction[:,numpy.newaxis], radius[:,numpy.newaxis], ang_a[:,numpy.newaxis], ang_b[:,numpy.newaxis]), axis=1)

kernel = 'composite(2:composite(1:gaussian, 1:gaussian), 2:fisher(%(ca)s), 2:mirror_fisher(%(cb)s))' # Don't ever do this: Just wanted to check a composite kernel within a composite kernel doesn't break things!

ms = MeanShift()
ms.set_data(data, 'df', None, 'rAA')
ms.set_kernel(kernel % {'ca' : 64.0, 'cb' : 64.0})



# Use the MeanShiftCompositeScale object to optimise...
optimise_scale = MeanShiftCompositeScale(kernel)
optimise_scale.add_param_scale(0)
optimise_scale.add_param_kernel('ca')
optimise_scale.add_param_kernel('cb')

steps = optimise_scale(ms)

print 'Optimisation of "a" took %i steps' % steps
print 'kernel = %s' % ms.get_kernel()
print 'scale = %s' % ms.get_scale()
print



# Visualise - input and a draw from the input...
def visualise(fn, data):
  img = numpy.zeros((size, size, 3), dtype=numpy.float32)
  for sample in data:
    bx = numpy.cos(sample[0]) * sample[1]
    by = numpy.sin(sample[0]) * sample[1]
    
    s_x = (size-1) * 0.5 * (1.0 + bx / scale)
    s_y = (size-1) * 0.5 * (1.0 + by / scale)
    
    e_x = s_x + angle_len * numpy.cos(sample[2])
    e_y = s_y + angle_len * numpy.sin(sample[2])
    
    o_x = mirror_len * numpy.cos(sample[3])
    o_y = mirror_len * numpy.sin(sample[3])
    
    try:
      img[t_y,t_x,1] = 1.0
    except:
      pass
      
    for i in xrange(angle_step):
      t = float(i) / (angle_step-1)
      t_x = int(t * s_x + (1-t) * e_x)
      t_y = int(t * s_y + (1-t) * e_y)
      try:
        if img[t_y,t_x,0] < t:
          img[t_y,t_x,0] = t
      except:
        pass
      
    for i in xrange(angle_step):
      t = float(i) / (angle_step-1)
      t_x = int(s_x + (1-t) * o_x)
      t_y = int(s_y + (1-t) * o_y)
      try:
        if img[t_y,t_x,2] < t:
          img[t_y,t_x,2] = t
      except:
        pass
      
    for i in xrange(angle_step):
      t = float(i) / (angle_step-1)
      t_x = int(s_x - (1-t) * o_x)
      t_y = int(s_y - (1-t) * o_y)
      try:
        if img[t_y,t_x,2] < t:
          img[t_y,t_x,2] = t
      except:
        pass

  img = array2cv(255.0 * img)
  cv.SaveImage(fn, img)


visualise('bandwidth_a_input.png', data)
draw = ms.draws(samples)
visualise('bandwidth_a_draw.png', draw)



# Change the data and reoptimise - data with a very different scale...
print 'b:'

direction = []
radius = []
ang_a = []
ang_b = []

for i in xrange(8):
  count = samples//8
  
  direction.append(numpy.random.normal(numpy.pi*2.0 * i / 8.0, 0.2, count))
  radius.append(0.5 + numpy.random.normal(2.0 * numpy.sin(numpy.pi*4.0 * i / 8.0), 0.2, count))
  ang_a.append(numpy.random.normal(numpy.pi*0.5, 0.1, count))
  ang_b.append(numpy.random.normal(0.0, 0.1, count))
  
direction = numpy.concatenate(direction)
radius = numpy.concatenate(radius)
ang_a = numpy.concatenate(ang_a)
ang_b = numpy.concatenate(ang_b)

data = numpy.concatenate((direction[:,numpy.newaxis], radius[:,numpy.newaxis], ang_a[:,numpy.newaxis], ang_b[:,numpy.newaxis]), axis=1)


ms.set_data(data, 'df', None, 'rAA')
steps = optimise_scale(ms)

print 'Optimisation of "b" took %i steps' % steps
print 'kernel = %s' % ms.get_kernel()
print 'scale = %s' % ms.get_scale()
print



# Visualise, again...
visualise('bandwidth_b_input.png', data)
draw = ms.draws(samples)
visualise('bandwidth_b_draw.png', draw)
