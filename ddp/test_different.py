#! /usr/bin/env python

# Copyright 2016 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

from ddp import DDP



# Simple test of the different pair cost...
dp = DDP()

dp.prepare(9, 3)

dp.unary(0, [0.0, 1.0, 2.0])
dp.unary(4, [2.0, 1.0, 0.0])
dp.unary(8, [0.0, 1.0, 2.0])

dp.pairwise(0, ['different'] * 8, [[0.5]] * 8)


dp.solve()

best, cost = dp.best()

print 'Best cost = %.1f' % cost
print 'Best solution: %s' % str(best)

print 'Costs:'
for i in xrange(dp.variables):
  print '[%.1f | %.1f | %.1f]' % tuple(dp.costs(i))
