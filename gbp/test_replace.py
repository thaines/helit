#! /usr/bin/env python
# Copyright 2014 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import numpy
from gbp import GBP



# Test the ability to replace values using the prev_exp keyword parameter - minimal test as its almost the same code path as everything else...
solver = GBP(2)

for ii in xrange(3):
  print 'V%i:' % ii
  
  if ii==0:
    solver.unary_sd(0, 50.0, 0.1)
    solver.pairwise_sd(0, 1, 10.0, 2.0)
  else:
    pe = 1.0 if ii==1 else 0.0
    solver.unary_sd(0, -100.0, 0.5, prev_exp = pe)
    solver.pairwise_sd(0, 1, 20.0, 1.0, prev_exp = pe)

  solver.solve()

  mean, sd = solver.result_sd(slice(solver.node_count))

  for i in xrange(solver.node_count):
    print '  %i: mean = %f, sd = %f' % (i, mean[i], sd[i])
  print
