# Copyright 2016 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import numpy



# Ranges for each parameter when optimising...
ranges = {'max_depth' : [7, 8, 9, 10, 12, 14, 16, 18, 20, 24, 28, 32, 40, 48, 56, 64],
          'min_split_samples' : [2, 4, 8, 16, 32, 6, 128, 256, 512, 1024],
          'dir_travel' : [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0],
          'travel_max' : [8.0, 16.0, 32.0, 48.0, 64.0, 96.0, 128.0, 192.0, 256.0, 384.0, 512.0, 768.0, 1024.0],
          'travel_bins' : [1, 2, 3, 4, 5, 6, 7, 8, 9],
          'travel_ratio' : [0.3, 0.5, 0.7, 0.8, 0.9, 1.0],
          'pos_bins' : [1, 2, 3, 4, 5, 6, 7],
          'pos_ratio' : [0.3, 0.5, 0.7, 0.8, 0.9, 1.0],
          'radius_bins' : [1, 2, 3, 4, 5],
          'density_bins' : [1, 2, 3, 4, 5]}

max_feat_length = 1024



def feature_len(params):
  """Returns the number of features for a parameter set as indices into ranges."""
  ret = 1
  ret *= ranges['travel_bins'][params['travel_bins']]
  ret *= ranges['pos_bins'][params['pos_bins']] ** 2
  ret *= ranges['radius_bins'][params['radius_bins']]
  ret *= ranges['density_bins'][params['density_bins']]
  return ret



def to_index(runs):
  """Given a set of runs converts every entry into an index into its list, rather than the actual value - this makes neighbour handling much easier."""
  keys = filter(lambda k: k!='score', ranges.keys())
  
  def wibble(params):
    ret = {}
    if 'score' in params: ret['score'] = params['score']
      
    for key in keys:
      ret[key] = numpy.argmin(numpy.fabs(numpy.array(ranges[key]) - params[key]))
    return ret
  
  return map(wibble, runs)



def closest(runs, params):
  """Given a set of runs and a single set of parameters returns the tuple (closest, index, distance). Makes the most sense if they are passed in as indices, and uses manhatten distance"""
  closest = None
  index = None
  distance = None
  
  keys = filter(lambda k: k!='score', ranges.keys())
  
  for i, run in enumerate(runs):
    dist = 0
    for key in keys:
      dist += numpy.abs(run[key] - params[key])
    
    if closest==None or dist < distance:
      closest = run
      index = i
      distance = dist
  
  return (closest, index, distance)



def modes(runs, distance = 2):
  """Given the runs, in index mode, returns all of the ones that are modes - i.e. no run within distance (manhatten) has a better score. Not very efficient, but I am lazy and its not like this is going to get that high."""
  keep = numpy.ones(len(runs), numpy.bool)
  
  for i in xrange(len(runs)):
    for j in xrange(i+1, len(runs)):
      run_i = runs[i]
      run_j = runs[j]
      
      dist = 0
      for key in filter(lambda k: k!='score', run_i.keys()):
        dist += numpy.abs(run_i[key] - run_j[key])
      
      if dist<=distance:
        if run_i['score'] < run_j['score']:
          keep[i] = False
        else:
          keep[j] = False
  
  return list(numpy.array(runs)[keep])
