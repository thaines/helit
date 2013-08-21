#! /usr/bin/env python

# Copyright 2013 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



# Simple script that prints out all kernels/spatials/balls that the code supports...

import ms



print 'Kernels:'
for kernel in ms.MeanShift.kernels():
  print '  >%s' % kernel
  
  d = ms.MeanShift.info(kernel)
  for i in xrange(0, len(d), 60):
    print '    %s' % d[i:i+60].strip()
  print



print
print 'Spatial:'
for spatial in ms.MeanShift.spatials():
  print '  >%s' % spatial
  
  d = ms.MeanShift.info(spatial)
  for i in xrange(0, len(d), 60):
    print '    %s' % d[i:i+60].strip()
  print


  
print
print 'Balls:'
for balls in ms.MeanShift.balls():
  print '  >%s' % balls
  
  d = ms.MeanShift.info(balls)
  for i in xrange(0, len(d), 60):
    print '    %s' % d[i:i+60].strip()
  print
