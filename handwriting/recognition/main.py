#! /usr/bin/env python

# Copyright 2016 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import sys
import os
import os.path
import string
import cPickle as pickle
import json
import bz2
import random
import time

import numpy

from frf import frf
  
from ply2 import ply2
from line_graph.line_graph import LineGraph

from lock_file import LockFile

from helpers import *
from utils.prog_bar import ProgBar



# Configuration stuff...
label_index = '_' + string.ascii_lowercase + string.ascii_uppercase + string.digits + string.punctuation # Note that '_' is for a ligament, and must always be in position 0.

trees = 64

max_depth = 20
min_split_samples = 64
feat_param = {'dir_travel':12.0, 'travel_max':512.0, 'travel_bins':8, 'travel_ratio':0.8, 'pos_bins':3, 'pos_ratio':0.9, 'radius_bins':1, 'density_bins':3}



# First check the user provided a directory to run in...
if len(sys.argv)<2:
  print 'Creates a database for handwriting spliting and recognition, using all line graphs found in the given directory and its subdirectories. Database is written into the directory as hwr.rf'
  print 'Usage:'
  print 'python main.py <dir to make db in> [optimise]'
  print
  sys.exit(1)

root_dir = sys.argv[1]



# If the user has requested optimisation then do just that - basically it either does random offsets from the best in a chain, or a random start, depending on a draw - totally ad-hoc...
if 'optimise' in sys.argv:
  runs = []
  
  # Load runs that have been done thus far...
  with LockFile('runs.json', 'r') as f:
    if f!=None:
      runs = json.load(f)
  
  ind_runs = to_index(runs)
  
  # Decide what to do based on runs thus far - in a While loop as it can fail...
  approach = ''
  while True:
    if len(runs)<32 or random.random()<0.1:
      approach = 'random'
      # Totally random initalisation...
      params = {}
      for key in ranges.keys():
        params[key] = random.randrange(len(ranges[key]))
      
      # Unlikelly, but check its not a duplicate...
      res = closest(ind_runs, params)
      if res[2]==0: continue
      
      # Only break and accept these values if the feature length is acceptable...
      if feature_len(params)<max_feat_length: break
    
    else:
      # Find all modes in the data set, random select a mode, or just select the best mode...
      if random.random()<0.1:
        approach = 'mode neighbour'
        best_runs = modes(ind_runs)
        mult_runs = []
        for run in best_runs:
          count = int(run['score'] * 100)
          if count<1: count = 1
          mult_runs += [run] * count
        sel_run = random.choice(mult_runs)
      
        params = sel_run.copy()
      else:
        approach = 'best neighbour'
        sel_run = ind_runs[0]
        for run in ind_runs:
          if sel_run['score']<run['score']:
            sel_run = run
        params = sel_run.copy()
      
      del params['score']
      
      # Offset from the mode by one step in a random dimension...
      okey = random.choice(params.keys())
      odir = 1 if random.random()>0.5 else -1
      
      if (params[okey] + odir) < 0: odir = 1
      if (params[okey] + odir) >= len(ranges[okey]): odir = -1
      
      params[okey] += odir
      
      # Check the position has not already been checked...
      res = closest(ind_runs, params)
      if res[2]==0: continue
    
      # Accept it...
      if feature_len(params)<max_feat_length: break

  # We have an object of parameter indices - store them ready for use...
  max_depth = ranges['max_depth'][params['max_depth']]
  min_split_samples = ranges['min_split_samples'][params['min_split_samples']]
      
  for key in feat_param.keys():
    feat_param[key] = ranges[key][params[key]]
  
  # Print parameters to screen...
  print 'Selection: %s' % approach
  print 'Parameters:'
  print '  max_depth =', max_depth
  print '  min_split_samples =', min_split_samples
  
  for key in feat_param.keys():
    print '  %s =' % key, feat_param[key]



# Obtain a list of all line graphs in the directory structure...
lg_fn = []
for root, _, files in os.walk(root_dir):
  for fn in [fn for fn in files if fn.endswith('.line_graph')]:
    lg_fn.append(os.path.join(root, fn))

if len(lg_fn)==0:
  print 'Failed to find any line graphs in the given directory'
  sys.exit(1)

print 'Found %i line graphs' % len(lg_fn)



# Load and process each in turn, to create a big database of feature/class...
train = []

for fn_num, fn in enumerate(lg_fn):
  print 'Processing %s: (%i of %i)' % (fn, fn_num+1, len(lg_fn))
  
  # Load line graph...
  print '    Loading...'
  f = open(fn, 'r')
  data = ply2.read(f)
  f.close()
    
  lg = LineGraph()
  lg.from_dict(data)
  lg.segment()
  
  # Extract features...
  print '    Extracting features...'
  fv = lg.features(**feat_param)
  
  # Extract labels, to match the features...
  segs = lg.get_segs()

  def seg_to_label(seg):
    tags = lg.get_tags(seg)
    if len(tags)==0: return 0 # Its a ligament
  
    for tag in tags:
      if tag[0]=='_': return -1 # Ignored - to be removed from training.
      code = filter(lambda c: c!='_', tag[0])
      try:
        return label_index.index(code)
      except ValueError:
        pass
    
    return -1
    
  seg_labels = numpy.array(map(seg_to_label, xrange(lg.segments)))
  
  # Create the relevant entity and store it in train...
  labels = seg_labels[segs]
  keep = labels>=0

  fv = fv[keep,:]
  labels = labels[keep]
  
  train.append((fv, labels))
  
  # Clean up...
  del lg
  print '    Done - %i features added' % fv.shape[0]



# Train contains a list of sets of exemplars - convert it into one super-fat data matrix with matching class-vector...
train_fv = numpy.concatenate([f for f, l in train], axis=0)
train_label = numpy.concatenate([l for f, l in train])
del train



# Generate weight vector so we can have equal probability of each class...
weight_by_label = numpy.bincount(train_label)
weight_by_label = numpy.array(weight_by_label, dtype=numpy.float32)
weight_by_label = numpy.clip(weight_by_label, 1.0, weight_by_label.max())
weight_by_label = weight_by_label.max() / weight_by_label
train_weight = weight_by_label[train_label]

print 'Train Weight: min = %f; mean = %f; max = %f' % (train_weight.min(), train_weight.mean(), train_weight.max())



# Train a random forest...
print 'Model fitting...'
print '(%i feature vectors)' % train_label.shape[0]

cull = 10000000 # Limit on how many feature vectors to use for trainning - my computer starts to swap if I set it any higher.

if train_label.shape[0]>cull:
  indices = numpy.random.permutation(train_label.shape[0])
  indices = indices[:cull]
  
  train_fv = train_fv[indices, :]
  train_label = train_label[indices]
  train_weight = train_weight[indices]
  print 'Culled to %i' % cull

forest = frf.Forest()
forest.configure('C', 'C', 'S' * train_fv.shape[1])
forest.min_exemplars = 8
forest.opt_features = int(numpy.sqrt(train_fv.shape[1]))
  
print 'frf learning:'
pb = ProgBar()
oob = forest.train(train_fv, [train_label,  ('w', train_weight)], trees, pb.callback)
del pb



# Report oob error rate for the forest, plus other stuff...
class_histogram = numpy.bincount(train_label)
popular = numpy.argmax(class_histogram)
popular_rate = class_histogram[popular] / float(class_histogram.sum())
popular_char = label_index[popular]

print '    Class count: %i' % len(label_index)
print '    Most common class: %s (%.2f%% of data set)' % (popular_char, popular_rate * 100.0)
  
print '    frf: OOB accuracy: %.2f%%' % ((1.0-oob.mean())*100.0)
nodes = sum(map(lambda t: t.nodes(), forest))
print '    frf: average node count: %.1f' % (nodes / float(len(forest)))



# If we are optimising record the new json...
if 'optimise' in sys.argv:
  params = feat_param.copy()
  params['max_depth'] = max_depth
  params['min_split_samples'] = min_split_samples
  params['score'] = float(1.0-oob.mean())
  
  with LockFile('runs.json', 'r+') as f:
    if f==None:
      runs = []
      f = open('runs.json', 'w')
    else:
      runs = json.load(f)

    runs.append(params)
      
    f.seek(0)
    f.truncate()
    json.dump(runs, f)



# Save the forest to disk, for future use, with the label index...
if 'optimise' not in sys.argv:
  start = time.clock()
    
  f = bz2.BZ2File(os.path.join(root_dir, 'hwr.rf'), 'w')
  pickle.dump({'classes':label_index, 'feat':feat_param}, f, pickle.HIGHEST_PROTOCOL)
    
  f.write(forest.save())
  
  for i in xrange(len(forest)):
    f.write(forest[i])
  
  f.close()
    
  end = time.clock()
  print '  frf: Saving time = %.3f' % (end-start)
    
  print '    Saved and done'
