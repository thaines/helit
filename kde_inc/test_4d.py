#! /usr/bin/env python

# Copyright 2012 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import os
import shutil

import cv
from utils.cvarray import *

import numpy
import numpy.random

from kde_inc import KDE_INC



# Parameters...
samples = 128
sigma = 0.75
mean = numpy.array([0.4, -2.0, 4.5, -1.2], dtype=numpy.float32)
cov = numpy.array([[4.0,2.0,-1.0,0.0],[2.0,3.0,0.0,-0.5],[-1.0,0.0,6.0,2.3],[0.0,-0.5,2.3,0.2]], dtype=numpy.float32)

directory = 'test_4d'
size = 128
gap = 8
sd_count = [2.0, 2.0, 2.0, 8.0]



# Prepare the directory...
try: shutil.rmtree(directory)
except: pass
os.makedirs(directory)



# Function to kick out renders of this shit...
def plot_pair(dimA, dimB):
  ret = numpy.zeros((size, size, 3), dtype=numpy.float32)
  
  startA = mean[dimA] - sd_count[dimA] * cov[dimA, dimA]
  endA = mean[dimA] + sd_count[dimA] * cov[dimA, dimA]
  startB = mean[dimB] - sd_count[dimB] * cov[dimB, dimB]
  endB = mean[dimB] + sd_count[dimB] * cov[dimB, dimB]
  
  for sam in sam_list:
    locA = int(size * (sam[dimA] - startA) / (endA - startA))
    locB = int(size * (sam[dimB] - startB) / (endB - startB))
    
    if locA>=0 and locA<size and locB>=0 and locB<size:
      ret[locA, locB, 1] = 1.0
  
  dist = kde_inc.marginalise([dimA, dimB])
  for y in xrange(size):
    ry = startA + (float(y)/size) * (endA - startA)
    for x in xrange(size):
      rx = startB + (float(x)/size) * (endB - startB)
      ret[y, x, 0] = dist.prob([ry, rx])
  ret[:, :, 0] /= ret[:, :, 0].max()
  
  return ret



def plot(fn):
  img01 = plot_pair(0, 1)
  img02 = plot_pair(0, 2)
  img03 = plot_pair(0, 3)
  img12 = plot_pair(1, 2)
  img13 = plot_pair(1, 3)
  img23 = plot_pair(2, 3)
  
  final = 0.2 * numpy.ones((3*size + 2*gap, 2*size + gap, 3), dtype=numpy.float32)
  
  final[0:size, 0:size, :] = img01
  final[size+gap:size*2+gap, 0:size, :] = img02
  final[size*2+gap*2:size*3+gap*2, 0:size, :] = img03
  
  final[0:size, size+gap:size*2+gap, :] = img12
  final[size+gap:size*2+gap, size+gap:size*2+gap, :] = img13
  final[size*2+gap*2:size*3+gap*2, size+gap:size*2+gap, :] = img23
  
  out = array2cv(final*255.0)
  cv.SaveImage(fn, out)



# Loop through and keep adding samples...
kde_inc = KDE_INC(numpy.eye(4, dtype=numpy.float32) / (sigma*sigma))

sam_list = []
for i in xrange(samples):
  sam = numpy.random.multivariate_normal(mean, cov)
  sam_list.append(sam)
  kde_inc.add(sam)

  plot('%s/after_%i.png'%(directory, i+1))
