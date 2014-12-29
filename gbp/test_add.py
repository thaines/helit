#! /usr/bin/env python
# Copyright 2014 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import numpy
from gbp import GBP



# Helper for below...
def present(solver):
  iters = solver.solve()
  print '  iters =', iters
  
  mean, prec = solver.result()
  
  for i in xrange(0, mean.shape[0], 5):
    print '  mean:     ' + ' '.join(['%7.2f'%v for v in mean[i:i+5]])
    print '  precison: ' + ' '.join(['%7.2f'%v for v in prec[i:i+5]])
  print



# Start with a suitably boring graph...
solver = GBP(1)
solver.unary(0, 0.0, 1e3)

print 'One node:'
present(solver)



# Make it more interesting...
base = solver.add(4)
indices = [base + i for i in xrange(4)]

solver.pairwise(indices, [i-1 for i in indices], -1.0, 1e3)

print 'Five, as chain:'
present(solver)



# More interesting, again...
base = solver.add(5)
indices = [base + i for i in xrange(5)]
solver.pairwise(xrange(5), indices, 10.0, 1e3)

print 'Extra row:'
present(solver)



# Yet another row, but with pairwise terms again...
base = solver.add(5)
more_indices = [base + i for i in xrange(5)]

solver.pairwise(indices, more_indices, 5.0, 1e1)
solver.pairwise(more_indices[:-1], more_indices[1:], 0.0, 1e3)

print 'Yet another row:'
present(solver)
