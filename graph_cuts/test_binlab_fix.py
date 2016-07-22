#! /usr/bin/env python
# Copyright 2016 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import numpy
from binary_label import BinaryLabel


for fix in [None, (0,1,1), (3,2,-1)]:
  print 'Fixing %s' % str(fix)
  
  bl = BinaryLabel((4,4))

  cf = numpy.ones((4,4))
  cf[:,:] *= numpy.arange(4).reshape((1,-1))
  cf[:,:] *= numpy.arange(4).reshape((-1,1))
  bl.addCostFalse(cf)

  ct = numpy.ones((4,4))
  ct[::-1,::-1] *= numpy.arange(4).reshape((1,-1))
  ct[::-1,::-1] *= numpy.arange(4).reshape((-1,1))
  ct += 1e-3 # Bias towards False
  bl.addCostTrue(ct)

  bl.addCostDifferent(0, numpy.ones((1,1)))
  bl.addCostDifferent(1, numpy.ones((1,1)))

  fix_arr = numpy.zeros((4,4), dtype=numpy.int32)
  if fix!=None:
    fix_arr[fix[0], fix[1]] = fix[2]
  bl.fix(fix_arr)


  assignment, cost = bl.solve()


  for y in xrange(4):
    print '  ',
    for x in xrange(4):
      print 'T ' if assignment[y,x] else 'F ',
    print
  print
