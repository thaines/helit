#! /usr/bin/env python

# Copyright 2012 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import numpy
import numpy.random
import cv

from utils.prog_bar import ProgBar
from utils.cvarray import *

from df import *



# Parameters...
samples = 2048
diameter = 2.0
gamma_alpha = 2.0
gamma_beta = 2.0

pixel_width = 512
pixel_half_width = pixel_width // 2
axis_half_width = 6.0



# Generate some data - select random points on a circle and then offset them tangentially using a Gamma distribution...
## Draw for the axes...
xdir = numpy.random.standard_normal(samples)
ydir = numpy.random.standard_normal(samples)

## Convert to unit direction vectors...
length = numpy.sqrt(numpy.square(xdir) + numpy.square(ydir))
xdir /= length
ydir /= length

## Draw the offsets using a gamma distribution...
scales = diameter + numpy.random.gamma(gamma_alpha, 1.0/gamma_beta, samples)

## Create the data matrix, stick it into an ExemplarSet...
xdir *= scales
ydir *= scales

data = numpy.append(xdir.reshape((-1,1)), ydir.reshape((-1,1)), axis=1)

es = MatrixES(data)
doMP = False



# Visualise...
img = numpy.zeros((pixel_width, pixel_width, 3), dtype=numpy.float32)
for i in xrange(samples):
  x = int(pixel_half_width * xdir[i] / axis_half_width + pixel_half_width)
  y = int(pixel_half_width * ydir[i] / axis_half_width + pixel_half_width)
  
  if x>=0 and x<pixel_width and y>=0 and y<pixel_width:
    img[y,x,:] += 1.0

img /= img.max()
cv.SaveImage('test_de_circle_input.png',array2cv(img*255))



# Function to test a given generator for desnity estimation...
def doTest(name, gen):
  # Train the model...
  df = DF()
  df.setGoal(DensityGaussian(2)) # 2 = # of features
  df.setGen(gen)
  df.getPruner().setMinTrain(48) # Playing around shows that this is probably the most important number to get right when doing density estimation - the information gain heuristic just doesn't know when to stop.
  
  global es
  pb = ProgBar()
  df.learn(32, es, callback = pb.callback, mp=doMP) # 32 = number of trees to learn - you need a lot to get a good answer.
  del pb
  
  # Drop some stats...
  print '%i trees containing %i nodes.\nAverage error is %.3f.'%(df.size(), df.nodes(), df.error())
  
  # Visualise the density estimate...
  global img
  testSet = numpy.empty((pixel_width,2), dtype=numpy.float32)
  pb = ProgBar()
  
  for y in xrange(pixel_width):
    pb.callback(y,pixel_width)
    i = 0
    for x in xrange(pixel_width):
      testSet[i,0] = axis_half_width * float(x - pixel_half_width) / pixel_half_width
      testSet[i,1] = axis_half_width * float(y - pixel_half_width) / pixel_half_width
      i += 1
    
    test = MatrixES(testSet)
    res = df.evaluate(test, mp=doMP)
    
    i = 0
    for x in xrange(pixel_width):
      img[y,x,:] = res[i]
      i += 1
    
  del pb
  
  print 'Maximum probability = %.2f'%img.max()
  img /= img.max()
  cv.SaveImage('test_de_circle_%s.png'%name,array2cv(img*255))

  

# Run the test on a set of generators...
print 'Axis-aligned median generator:'
doTest('axis_median', AxisMedianGen(0,2)) # 0 = channel to use to generate tests, 2 = # of tests to try.
print

print 'Linear median generator:'
doTest('linear_median', LinearMedianGen(0,2,4,8)) # 0 = channel to use to generate tests, 2 = # of dimensions for each test, 4 = # of dimension possibilities to consider, 8 = # of orientations to consider.
print

print 'Axis-aligned random generator:'
doTest('axis_random', AxisRandomGen(0,4,8)) # 0 = channel to generate tests for, 4 = # of dimensions to try splits for, 8 = # of splits to try per dimension.
print

print 'Linear random generator:'
doTest('linear_random', LinearRandomGen(0,2,4,8,4)) # 0 = channel to generate tests for, 2 = # of dimensions used for each test, 4 = number of random dimension selections to try, 8 = number of random directions to try, 4 = number of random split points to try.
print
