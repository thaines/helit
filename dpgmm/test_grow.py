#! /usr/bin/env python

# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import os
import shutil
import random
import numpy
import cv

from utils import cvarray
from utils.prog_bar import ProgBar
from gcp.gaussian import Gaussian
from dpgmm import DPGMM



# Tests the dpgmm model by seeing how well it fits a mixture of 3 1D Gaussians, which is visualised using open CV. This version uses the multiGrowSolve method, to get the best answer possible.



# Define the Gaussian, draw the samples...
dims = 1
gt_weight = [0.2,0.5,0.3]
gt = [Gaussian(dims), Gaussian(dims), Gaussian(dims)]
gt[0].setMean([0.0])
gt[0].setCovariance([[1.0]])
gt[1].setMean([4.0])
gt[1].setCovariance([[5.0]])
gt[2].setMean([10.0])
gt[2].setCovariance([[4.0]])

sample_count = 2048
samples = []
for _ in xrange(sample_count):
  r = random.random()
  for i in xrange(len(gt)):
    r -= gt_weight[i]
    if r<0.0: break
  x = gt[i].sample()
  samples.append(x)



# Create the output directory...
out_dir = 'test_grow'
try: shutil.rmtree(out_dir)
except: pass
os.mkdir(out_dir)



# Output parameters...
low = -2.0
high = 14.0
width = 800
height = 400
scale = 1.5 * max(map(lambda i: gt_weight[i]*gt[i].prob(gt[i].getMean()), xrange(len(gt))))



# Iterate a number of sample counts...
out = [8,16,32,64,128,256,512,1024,2048]

for dpc in out:
  print '%i datapoints:'%(dpc)
  # Fill in the model...
  model = DPGMM(dims)
  for point in samples[:dpc]: model.add(point)
  model.setPrior()
  
  # Solve...
  p = ProgBar()
  model = model.multiGrowSolve(8)
  del p


  # Now plot the estimated distribution against the actual distribution...
  img = numpy.ones((height,width,3))
  draw = model.sampleMixture()

  for px in xrange(width):
    x = float(px)/float(width) * (high-low) + low
     
    y_gt = 0.0
    for ii  in xrange(len(gt)):
      y_gt += gt_weight[ii] * gt[ii].prob([x])
    y_gu = model.prob([x])
    y_gd = 0.0
    for ind,gauss in enumerate(draw[1]):
      y_gd += draw[0][ind] * gauss.prob([x])
        
    py_gt = int((1.0 - y_gt/scale) * height)
    py_gu = int((1.0 - y_gu/scale) * height)
    py_gd = numpy.clip(int((1.0 - y_gd/scale) * height),0,height-1)

    img[py_gt,px,:] = [0.0,1.0,0.0]
    img[py_gu,px,:] = [1.0,0.0,0.0]
    img[py_gd,px,:] = [0.0,0.0,1.0]

  # Save plot out...
  img = cvarray.array2cv(img*255.0)
  cv.SaveImage('%s/plot_%i.png'%(out_dir,dpc),img)
  print
