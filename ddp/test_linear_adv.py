#! /usr/bin/env python

# Copyright 2016 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import numpy
from ddp import DDP



# More advanced test of the linear node; makes use of the linear offset with variable numbers of nodes...
dp = DDP()

dp.prepare([3, 5, 7, 9, 7, 5, 3])

dp.unary(0, [10.0, 0.0, 10.0])
dp.unary(3, [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 0.0])
dp.unary(6, [10.0, 0.0, 10.0])

pc = numpy.empty((6, 4), dtype=numpy.float32)
pc[:,0] = [1.0/3.0, 1.0/5.0, 1.0/7.0, 1.0/9.0, 1.0/7.0, 1.0/5.0]
pc[:,1] = [0.2, 0.2, 0.2, -0.2, -0.2, -0.2]
pc[:,2] = [3.0/5.0, 5.0/7.0, 7.0/9.0, 9.0/7.0, 7.0/5.0, 5.0/3.0]
pc[:,3] = 0.3

dp.pairwise(0, ['linear'] * 6, pc)



best, cost = dp.best()

print 'Best cost = %.1f' % cost
print 'Best solution: %s' % str(best)
print 'Costs:'
for i in xrange(dp.variables):
  print '[' + ' | '.join(map(lambda val: '%.1f'%val, dp.costs(i))) + ']'
