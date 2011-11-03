#! /usr/bin/env python

# Copyright 2011 Tom SF Haines

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.



import sys
import os
import os.path
import pickle
import shutil

import math
import numpy
import cv

from utils.prog_bar import ProgBar
from utils.cvarray import *

from p_cat.kde_inc.loo_cov import PrecisionLOO
from p_cat.p_cat import ClassifyKDE
from pool import Pool
from iris.iris import Iris1D



# Simple test of the active learning system - not really for testing but saves out visualisations of the results, to demonstrate how the algorithm works.



# Parameters...
limit = 64
out_dir = 'iris_results'
prec_cache = 'precision_iris.pickle'
width = 800
height = 200



# Fetch the task...
if len(sys.argv)<2: task = 'p_wrong_hard'
else:
  task = sys.argv[1]
  assert(task in Pool.methods())
  
  
# Load the dataset...
data = Iris1D()
print 'Loaded %i examples'%data.getVectors().shape[0]



# Make the output directory, killing any previous versions...
try: shutil.rmtree(out_dir)
except: pass
os.makedirs(out_dir)



# This calculates a suitable precision matrix to use...
print 'Calculating loo optimal precision matrix for data set...'
p = ProgBar()
loo = PrecisionLOO()
for i in xrange(data.getVectors().shape[0]):
  loo.addSample(numpy.reshape(data.getVectors()[i], (1,1)))
loo.solve(p.callback)
precision = loo.getBest()
del p


print 'Optimal standard deviation = %s'%str(math.sqrt(1.0/precision[0,0]))



# Create and fill the pool...
print 'Filling the pool...'
pool = Pool()
p = ProgBar()
for i in xrange(data.getVectors().shape[0]):
  p.callback(i, data.getVectors().shape[0])
  pool.store(numpy.reshape(data.getVectors()[i], (1,)), data.getClasses()[i])
del p

# Create the classifier...
classifier = ClassifyKDE(precision)




# Calculate the dimensions for the visualisations...
low  = data.getVectors().min()
high = data.getVectors().max()
low  -= 0.2*(high-low)
high += 0.2*(high-low)



# Quickly visualise the dataset as lines in an image...
weight = 1.0
img = numpy.ones((height//4, width, 3))
for i in xrange(data.getVectors().shape[0]):
  colour = [[1.0,0.0,0.0], [0.0,1.0,0.0], [0.0,0.0,1.0]][data.getClasses()[i]]
  pos = int(width * (data.getVectors()[i]-low) / (high-low))
  img[:, pos, :] = colour


img = array2cv(numpy.clip(img,0.0,1.0)*255.0)
cv.SaveImage('%s/data.png'%out_dir, img)



# Define a function to visualise the model state...
def visualise(filename):
  # Create the image object...
  img = numpy.ones((height, width, 3))

  # Get a full set of probabilities for every x position in the image...
  y_dict = []
  for ix in xrange(width):
    x = (high-low)*(float(ix)/float(width)) + low
    y_dict.append(classifier.getDataProb(numpy.array([x])))

  # Find the maximums...
  y_max = dict(y_dict[0])
  for pd in y_dict:
    for key, prob in pd.iteritems():
      if y_max[key]<prob:
        y_max[key] = prob

  # Render the prior and all classes that have been found...
  for key in y_max.iterkeys():
    if key==None: colour = [0.5,0.5,0.5]
    else: colour = [[1.0,0.0,0.0], [0.0,1.0,0.0], [0.0,0.0,1.0]][key]
    iy = map(lambda pd: height-1 - int((height-1)*(pd[key]/y_max[key])), y_dict)
    for ix in xrange(len(iy)):
      lx = max(ix-1,0)
      hx = ix+2
      ly = min(iy[lx:hx])
      hy = max(iy[lx:hx])
      img[ly:hy+1,ix,:] = colour

  # Render both P(new) and P(wrong) interest curves...
  conc = pool.getConcentration()
  counts = classifier.getCatCounts()
  
  y_new = []
  y_wrong = []
  for ix in xrange(width):
    # Calculate the dp's distribution on options, including P(new)...
    prob = dict()
    div = 0.0
    for key, p in y_dict[ix].iteritems():
      mult = conc if key==None else counts[key]
      prob[key] = mult * p
      div += prob[key]
    for key in prob.iterkeys():
      prob[key] /= div

    y_new.append(prob[None])

    if len(prob.keys())==1: y_wrong.append(prob[None])
    else:
      best = None
      bestScore = 0.0
      for key, p in y_dict[ix].iteritems():
        if key!=None and p>bestScore:
          best = key
          bestScore = p

      y_wrong.append(1.0 - prob[best])

  new_max = max(y_new)
  iy = map(lambda p: height-1 - int((height-1)*p/new_max), y_new)
  for ix in xrange(len(iy)):
    lx = max(ix-1,0)
    hx = ix+2
    ly = min(iy[lx:hx])
    hy = max(iy[lx:hx])
    img[ly:hy+1,ix,:] = [0.0,0.5,1.0]

  wrong_max = max(y_wrong)
  iy = map(lambda p: height-1 - int((height-1)*p/wrong_max), y_wrong)
  for ix in xrange(len(iy)):
    lx = max(ix-1,0)
    hx = ix+2
    ly = min(iy[lx:hx])
    hy = max(iy[lx:hx])
    img[ly:hy+1,ix,:] = [0.0,0.0,0.0]

  # Save the image object...
  img = array2cv(numpy.clip(img,0.0,1.0)*255.0)
  cv.SaveImage(filename+'.png', img)

  # Save the csv file...
  f = open(filename+'.csv','w')
  f.write('prior, red, green, blue, P(new), P(wrong)\n')
  for ix in xrange(width):
    sd = y_dict[ix]
    
    pr = sd[None]/y_max[None]
    cl = []
    for c in xrange(3):
      cl.append(sd[c]/y_max[c] if c in sd else -1.0)
    pn = y_new[ix] / new_max
    pw = y_wrong[ix] / wrong_max
    f.write('%f, %f, %f, %f, %f, %f\n'%(pr, cl[2], cl[1], cl[0], pn, pw))
  f.close()

  # Return the interest in new classes...
  return conc / (conc + sum(counts.values()))



# Active learning loop - at each stage save an image of the current interest functions...
# (Uses P(wrong), but plots both P(wrong) and P(new).)
print 'Doing active learning...'
p = ProgBar()
p.callback(0,limit)
conc_graph = []
conc_graph.append(visualise('%s/query_%s_%02i'%(out_dir, task, 0)))

for ii in xrange(1,limit+1):
  p.callback(ii,limit)

  # Select a sample and update the model...
  pool.update(classifier)
  sample, prob_dict, cat = pool.select(task)
  classifier.add(sample, cat)

  # Visualise the model after this many querys...
  conc_graph.append(visualise('%s/query_%s_%02i'%(out_dir, task, ii)))

del p



# Save out the concentration graph, to indicate interest levels...
f = open('%s/conc_%s.csv'%(out_dir,task), 'w')
f.write('queries, new_interest\n')
for i,c in enumerate(conc_graph):
  f.write('%i, %f\n'%(i,c))
f.close()
