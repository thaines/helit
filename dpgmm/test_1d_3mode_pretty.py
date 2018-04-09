#! /usr/bin/env python

# Copyright 2018 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import os
import shutil

import numpy
import matplotlib.pyplot as plt

from utils.prog_bar import ProgBar
from gcp.gaussian import Gaussian
from dpgmm import DPGMM



# Tests the dpgmm model by seeing how well it fits a mixture of 3 1D Gaussians...



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
  r = numpy.random.rand()
  for i in xrange(len(gt)):
    r -= gt_weight[i]
    if r<0.0: break
  x = gt[i].sample()
  samples.append(x)



# Parameters...
x_low = -3.0
x_high = 15.0
y_high = 0.15



# Create the output directory...
out_dir = 'test_1d_3mode_pretty'
try: shutil.rmtree(out_dir)
except: pass
os.mkdir(out_dir)



# Dump first figure - just the ground truth...
x = numpy.linspace(x_low, x_high, 1024)

y_true = numpy.zeros(1024)
for i in range(len(gt)):
  y_true += gt_weight[i] * gt[i].prob(x[:,None])

plt.figure(figsize=(16,8))
plt.xlim(x_low, x_high)
plt.ylim(0.0, y_high)
plt.plot(x, y_true, c='g')
plt.savefig(os.path.join(out_dir, '0000.png'), bbox_inches='tight')






# Iterate, slowlly building up the number of samples used and outputting the fit for each...
out = [8,16,32,64,128,256,512,1024,2048]

model = DPGMM(dims, 8)
model.setConcGamma(1/8., 1/8.)

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
    
    # Calculate it's posterior distribution...
    y_post = model.prob(x[:,None])
    
    # Generate and save a nice figure...
    plt.figure(figsize=(16,8))
    plt.xlim(x_low, x_high)
    plt.ylim(0.0, y_high)
    plt.plot(x, y_true, c='g')
    plt.plot(x, y_post, c='b')
    plt.savefig(os.path.join(out_dir, '{:04d}.png'.format(i+1)), bbox_inches='tight')
