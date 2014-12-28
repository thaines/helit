#! /usr/bin/env python
# Copyright 2014 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import numpy
from gbp import GBP



# Verify that setting unary variance to infinity does the right thing (its not supported for pairwise terms)...

for alg in ['bp', 'trw-s']:
  print 'Solve with %s:' % alg
  
  solver = GBP(6)
  solver.unary(0, -4.0, numpy.inf)
  solver.unary_sd(2, 93.0, 0.0)
  solver.unary_raw(4, -32.0, numpy.inf) # In raw case you send in the mean if precision is infinite!
  
  solver.pairwise([0,2,4], [1,3,5], 5.0, 1.0)
  
  if alg=='bp':
    solver.solve_bp()
  else:
    solver.solve_trws()
  
  mean, sd = solver.result_sd(slice(solver.node_count))

  for i in xrange(solver.node_count):
    print '  %i: mean = %f, sd = %f' % (i, mean[i], sd[i])
  print
