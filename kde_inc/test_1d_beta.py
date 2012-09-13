#! /usr/bin/env python

# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import os
import shutil

import math
import random

import numpy
import scipy.special

import cv

from kde_inc import KDE_INC



# Test of KDE_INC - uses it to perform a density estimate on a mixture of two Beta distributions, which is obviously not something that can be accuratly represented with a finite mixture of Gaussians.



# Parameters...
weight = [0.4,0.6] # Must sum to one, obviously.
alpha = [1.2,3.0]
beta = [1.6,1.2]
sigma = 0.1

low  = 0.0 # range for graphing the functions.
high = 1.0 # "

directory = '1d_beta'

samples = 512



# Pair of functions that define the ground truth distribution - one to darw from it, another to sample its pdf...
def pdfBeta(x, alpha, beta):
  ret  = math.pow(x, alpha-1.0)
  ret *= math.pow(1.0-x, beta-1.0)
  ret /= scipy.special.beta(alpha, beta)
  return ret

def pdf(x):
  if x<0.0: return 0.0
  if x>1.0: return 0.0

  ret = 0.0
  for i in xrange(len(weight)):
    ret += weight[i] * pdfBeta(x, alpha[i], beta[i])
  return ret



def draw():
  r = random.random()
  for i in xrange(len(weight)):
    r -= weight[i]
    if r<0.0 or i+1==len(weight):
      return random.betavariate(alpha[i], beta[i])



# Define a function to render the current state of the KDE_GMM - uses cv to write an image that contains the ground truth in green and the estimate in blue...
width = 256
height = 128
ground_truth = None # Cached to avoid repeated computation.

def plot(filename, kde_inc):
  global ground_truth

  # If needed generate the ground truth...
  if ground_truth==None:
    ground_truth = []
    for i in xrange(width):
      x = (high-low) * float(i)/float(width-1) + low
      ground_truth.append(pdf(x))

  # Generate the estimate plot from the current state of kde_gmm...
  estimate = []
  for i in xrange(width):
    x = (high-low) * float(i)/float(width-1) + low
    estimate.append(kde_inc.prob(x))

  # Create an image...
  img = cv.CreateImage((width,height),cv.IPL_DEPTH_32F,3)
  cv.Set(img,cv.CV_RGB(0.0,0.0,0.0))

  # Plot...
  maxGT = max(ground_truth)
  yGT = map(lambda x: height-1-int(float(height-1)*x/maxGT), ground_truth)

  maxE = max(estimate)
  yE = map(lambda x: height-1-int(float(height-1)*x/maxE), estimate)

  for i in xrange(width-1):
    cv.Line(img, (i,yGT[i]), (i+1,yGT[i+1]), cv.CV_RGB(0.0,255.0,0.0))
    cv.Line(img, (i,yE[i]), (i+1,yE[i+1]), cv.CV_RGB(255.0,0.0,0.0))

  # Save the image...
  cv.SaveImage(filename, img)



# Prepare the directory...
try: shutil.rmtree(directory)
except: pass
os.makedirs(directory)



# Iterate for the number of samples - draw them, update the model, and save the output...
kde_inc = KDE_INC(numpy.array([[1.0/(sigma*sigma)]], dtype=numpy.float32))

for s in xrange(samples):
  sam = draw()
  kde_inc.add(sam)
  fn = '%s/%04d.png'%(directory, s+1)
  plot(fn, kde_inc)
