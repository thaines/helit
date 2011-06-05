#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import os
import shutil
import math
import numpy
import cv

from utils.cvarray import *
from gaussian_prior import *



# Setup the inital prior...
gp = GaussianPrior(2)
gp.addPrior(numpy.array([0.0,0.0]),numpy.array([[100.0,0.0],[0.0,100.0]]))

# Draw a Gaussian to use as ground truth...
gt = gp.sample()
print 'gt mean =', gt.getMean()
print 'gt covariance =', str(gt.getCovariance()).replace('\n','').replace('  ',' ')

# Draw a lot of sample from the ground truth...
samples = map(lambda _: gt.sample(),xrange(32768))



# Create an output directory...
base = 'test_2d'
try: shutil.rmtree(base)
except: pass
os.mkdir(base)

# Select a range to render to the graph...
width = 400
height = 400
lowX = gt.getMean()[0] - 3.0*math.sqrt(gt.getCovariance()[0,0])
highX = gt.getMean()[0] + 3.0*math.sqrt(gt.getCovariance()[0,0])
lowY = gt.getMean()[1] - 3.0*math.sqrt(gt.getCovariance()[1,1])
highY = gt.getMean()[1] + 3.0*math.sqrt(gt.getCovariance()[1,1])



# Go through and, starting with no data, increase the amount of data in steps and save an image for each step - green for ground truth, red for the integrated out curve and blue for a draw from the prior...
def saveGraph(fn, red, green = None, blue = None):
  """Takes 3 optional 2d arrays of floats, normalises them and saves them to the 3 channels of an image."""
  maxValue = 0.0
  if red!=None: maxValue = max((maxValue,red.max()))
  if green!=None: maxValue = max((maxValue,green.max()))
  if blue!=None: maxValue = max((maxValue,blue.max()))

  img = numpy.zeros((width,height,3),dtype=numpy.float32)

  if red!=None: img[:,:,2] = red * (255.0/maxValue)
  if green!=None: img[:,:,1] = green * (255.0/maxValue)
  if blue!=None: img[:,:,0] = blue * (255.0/maxValue)

  img = array2cv(img)
  cv.SaveImage(fn,img)


def render(index, ex=''):
  draw = gp.sample()
  intOut = gp.intProb()

  red = numpy.zeros((width,height),dtype=numpy.float32)
  green = numpy.zeros((width,height),dtype=numpy.float32)
  blue = numpy.zeros((width,height),dtype=numpy.float32)
  for yPos in xrange(height):
    y = float(yPos)/float(height-1) * (highY-lowY) + lowY
    for xPos in xrange(width):
      x = float(xPos)/float(width-1) * (highX-lowX) + lowX
      red[yPos,xPos] = intOut.prob([x,y])
      green[yPos,xPos] = gt.prob([x,y])
      blue[yPos,xPos] = draw.prob([x,y])
  saveGraph('%s/graph_%06d%s.png'%(base,index,ex),red,green,blue)


render(0)
gp.addSample(samples[0])
render(1)
gp.addSample(samples[1])
render(2)
gp.addSample(samples[2])
gp.addSample(samples[3])
render(4)

scale = 4
while True:
  start = scale
  scale *= 2
  if scale>len(samples): break

  gp.addSamples(samples[start:scale])
  render(scale)

gp = GaussianPrior(2)
gp.addPrior(numpy.array([0.0,0.0]),numpy.array([[100.0,0.0],[0.0,100.0]]))
gp.addSamples(samples)
render(len(samples),'b')
