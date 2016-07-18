#! /usr/bin/env python

# Copyright 2016 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import numpy
from ddp import DDP



# Test of the ordered node, used when you have a fixed sequence and want to infer the splits...
dp = DDP()
dp.prepare(32, 4)

uc = numpy.empty((32, 4), dtype=numpy.float32)
uc[:, 0] = 1.0 - numpy.cos(0.0*numpy.pi + 3.0 * numpy.pi * numpy.arange(32) / 31.0)
uc[:, 1] = 1.0 - numpy.cos(1.0*numpy.pi + 3.0 * numpy.pi * numpy.arange(32) / 31.0)
uc[:, 2] = 1.0 - numpy.cos(2.0*numpy.pi + 3.0 * numpy.pi * numpy.arange(32) / 31.0)
uc[:, 3] = 1.0 - numpy.cos(3.0*numpy.pi + 3.0 * numpy.pi * numpy.arange(32) / 31.0)
uc[0, 1:] = float('inf')
dp.unary(0, uc)

dp.pairwise(0, ['ordered'] * 32, [[0.0, 5.0]] * 32)



best, cost = dp.best(3)

print 'Best cost = %.1f' % cost
print 'Best solution: %s' % str(best)
#print 'Costs:'
#for i in xrange(dp.variables):
#  print '[' + ' | '.join(map(lambda val: '%.1f'%val, dp.costs(i))) + ']'
