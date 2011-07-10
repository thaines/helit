#! /usr/bin/env python

# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import math
import numpy
import numpy.random
import cv

from kde_inc import *



# Draw a bunch of data points uniformly from an ellipsoid based distribution, using rejection sampling (3D ellipsoid, but only 2 dimensions used as data points.)...

pointA = numpy.asarray([0.0,0.0,0.0])
pointB = numpy.asarray([3.0,0.0,0.0])
length = 5.0

h = math.sqrt(2.5**2.0 - 1.5**2.0)
minSam = numpy.asarray([-1.0,-h,-h])
maxSam = numpy.asarray([4.0,h,h])

sampleCount = 256



# Function to sample from it...
def sample():
  while True:
    x = numpy.random.random_sample([3,])
    x *= maxSam-minSam
    x += minSam

    dist = math.sqrt(numpy.square(x-pointA).sum()) + math.sqrt(numpy.square(x-pointB).sum())

    if dist<length: return x[:2]



# Draw some samples...
samples = []
for _ in xrange(sampleCount):
  samples.append(sample())



# Use loo to calculate precision matrices for various scenarios...
ploo = PrecisionLOO()
for s in samples: ploo.addSample(s)

ploo.solve()

print 'Best precision using all samples:'
print ploo.getBest()


sploo = SubsetPrecisionLOO()
for s in samples: sploo.addSample(s)

sploo.solve(8, sampleCount/8)

print 'Best precision using a subset of all the samples:'
print sploo.getBest()



# Do a density estimate using a loo estiamte...
kde_inc = KDE_INC(ploo.getBest())
for s in samples: kde_inc.add(s)



# Visualise...
low = minSam[:2] - 1.0
high = maxSam[:2] + 1.0
step = numpy.asarray([500,200])

img = cv.CreateImage((step[0],step[1]), cv.IPL_DEPTH_32F,3)
cv.Set(img, cv.CV_RGB(0.0,0.0,0.0))

maxP = 0.0
for a in xrange(step[0]):
  for b in xrange(step[1]):
    x = low + (high-low)*(numpy.asarray([a,b], dtype=numpy.float32)/step)
    p = kde_inc.prob(x)
    img[b,a] = (p,p,p)
    maxP = max(maxP,p)

cv.ConvertScale(img, img, 255.0/maxP)
cv.SaveImage('test_loo.png', img)
