#! /usr/bin/env python

# Copyright 2016 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import numpy
from ddp import DDP



# Test the full type, where you provide a complete cost matrix...
dp = DDP()
dp.prepare(9, 3)

dp.unary(0, [0.0, 5.0, 5.0])

cyclic = numpy.ones((3,3), dtype=numpy.float32)
cyclic[0,1] = 0.0
cyclic[1,2] = 0.0
cyclic[2,0] = 0.0
dp.pairwise(0, ['full'] * 8, numpy.repeat(cyclic[numpy.newaxis,:,:], 8, axis=0))



best, cost = dp.best()

print 'Best cost = %.1f' % cost
print 'Best solution: %s' % str(best)
print 'Costs:'
for i in xrange(dp.variables):
  print '[' + ' | '.join(map(lambda val: '%.1f'%val, dp.costs(i))) + ']'

print

best, cost = dp.best(2,0)
print 'Fixed to pass through (2,0):'
print 'Best cost = %.1f' % cost
print 'Best solution: %s' % str(best)
