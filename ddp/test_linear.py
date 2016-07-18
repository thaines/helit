#! /usr/bin/env python

# Copyright 2016 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import numpy
from ddp import DDP



# Simple test of using the linear pair cost...
dp = DDP()

dp.prepare(12, 5)

uc = numpy.zeros((12, 5), dtype=numpy.float32)
uc[0,:]  = [0.0, 5.0, 5.0, 5.0, 5.0]
uc[2,:]  = [5.0, 5.0, 0.0, 5.0, 5.0]
uc[5,:]  = [5.0, 5.0, 5.0, 5.0, 0.0]
uc[8,:]  = [5.0, 5.0, 0.0, 5.0, 5.0]
uc[11,:] = [0.0, 5.0, 5.0, 5.0, 5.0]
dp.unary(0, uc)


pc = numpy.ones((11, 1), dtype=numpy.float32)
pc *= 0.5
dp.pairwise(0, ['linear'] * 11, pc)



best, cost = dp.best()

print 'Best cost = %.1f' % cost
print 'Best solution: %s' % str(best)
print 'Correct:       [0 _ 2 _ _ 4 _ _ 2 _ _ 0]'

print 'Costs:'
for i in xrange(dp.variables):
  print '[%.1f | %.1f | %.1f | %.1f | %.1f]' % tuple(dp.costs(i))
