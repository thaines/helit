#! /usr/bin/env python

# Copyright 2016 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import sys
import json
from lock_file import LockFile

from helpers import *



# Load runs...
runs = []

with LockFile('runs.json', 'r') as f:
  if f!=None:
    runs = json.load(f)



# Handle there being no data, print out the count...
print 'Found %i runs' % len(runs)
print
if len(runs)==0: sys.exit(0)



# Find the best...
best = 0

for i in xrange(1, len(runs)):
  if runs[best]['score'] < runs[i]['score']:
    best = i



# Analyse how many of its adjacent positions have been checked...
runs_ind = to_index(runs)
best_ind = to_index([runs[best]])[0]

checked = dict()
for key in runs[best].iterkeys():
  checked[key] = [False, False]

for run in runs_ind:
  dist = 0
  for key, value in best_ind.iteritems():
    if key!='score':
      dist += abs(value - run[key])

  if dist==1: # Its adjacent.
    for key, value in best_ind.iteritems():
      if value!=run[key]:
        if run[key]<value: checked[key][0] = True
        else: checked[key][1] = True



# Report...
for key, value in runs[best].iteritems():
  if key!='score':
    sval = str(value)
    fac = 24 - len(key) - len(sval)
    print '%s = %s%s(-1: %s, +1: %s)' % (key, sval, ' '*fac, '  checked' if checked[key][0] else 'unchecked', '  checked' if checked[key][1] else 'unchecked')
print
print 'Score = %.2f%%' % (runs[best]['score'] * 100.0)
print 'Feature length = %i' % (runs[best]['travel_bins'] * (runs[best]['pos_bins']**2) * runs[best]['radius_bins'] * runs[best]['density_bins'])
print
