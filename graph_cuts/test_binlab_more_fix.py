#! /usr/bin/env python
# Copyright 2016 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import numpy
from binary_label import BinaryLabel





for true_cost in [3.0, 7.0]:
  print 'True Cost = %.1f' % true_cost
  
  bl = BinaryLabel((5,))

  bl.addCostTrue(true_cost * numpy.ones(5))
  bl.addCostDifferent(0, 9.0 * numpy.ones(1))

  fix = numpy.zeros(5, dtype=numpy.int32)
  fix[2] = 1
  bl.fix(fix)
  
  assignment, cost = bl.solve()
  print '  Cost = %.1f' % cost
  
  print '  ',
  for x in xrange(5):
    print 'T ' if assignment[x] else 'F ',
  print
  print
