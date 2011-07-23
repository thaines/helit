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



# Tests the dpgmm model by seeing how well it fits a mixture of 2 1D Gaussians, which is visualised using open CV.



# Define the Gaussian, draw the samples...
dims = 1
gt_weight = [0.4,0.6]
gt = [Gaussian(dims), Gaussian(dims)]
gt[0].setMean([5.0])
gt[0].setCovariance([[4.0]])
gt[1].setMean([-1.0])
gt[1].setCovariance([[4.0]])

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
out_dir = 'test_1d_2mode'
try: shutil.rmtree(out_dir)
except: pass
os.mkdir(out_dir)



# Output parameters...
low = -5.0
high = 9.0
width = 400
height = 200
scale = 1.2 * max(map(lambda i: gt_weight[i]*gt[i].prob(gt[i].getMean()), xrange(len(gt))))



# Iterate, slowlly building up the number of samples used and outputting the fit for each...
out = [8,16,32,64,128,256,512,1024,2048]

model = DPGMM(dims, 8)
for i,point in enumerate(samples):
  model.add(point)
  
  if (i+1) in out:
    print '%i datapoints:'%(i+1)
    # First fit the model...
    model.setPrior()
    p = ProgBar()
    it = model.solve()
    del p
    print 'Updated fitting in %i iterations'%it

    # Some information...
    #print 'a:'
    #print model.alpha
    #print 'v:'
    #print model.v
    #print 'stick breaking weights:'
    #print model.v[:,0] / model.v.sum(axis=1)
    #print 'stick weights:'
    #print model.intMixture()[0]
    #print 'z sums:'
    #print model.z.sum(axis=0)

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
    cv.SaveImage('%s/plot_%i.png'%(out_dir,i+1),img)
    print
