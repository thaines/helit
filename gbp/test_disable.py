#! /usr/bin/env python
# Copyright 2014 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import numpy
from gbp import GBP



# Setup a chain...
print 'Whole chain...'
solver = GBP(5)

solver.unary(0, 0.0, 15.0)
solver.unary(4, 10.0, 15.0)
solver.pairwise(slice(None,-1), slice(1, None), 0.5, 1.0)

iters = solver.solve_bp()
print 'iters =', iters

mean, prec = solver.result()

print 'Mean:     ' + ' '.join(['%.2f'%v for v in mean])
print 'Precison: ' + ' '.join(['%.2f'%v for v in prec])
print



# Disable middle and resolve...
print 'Middle gone...'
solver.disable(2)

iters = solver.solve_bp()
print 'iters =', iters

mean, prec = solver.result()

print 'Mean:     ' + ' '.join(['%.2f'%v for v in mean])
print 'Precison: ' + ' '.join(['%.2f'%v for v in prec])
print



# Change the configuration...
print 'Tail gone...'
solver = solver.clone() # Sneak in a test of clone.
solver.enable(2)
solver.disable(4)

iters = solver.solve_trws()
print 'iters =', iters

mean, prec = solver.result()

print 'Mean:     ' + ' '.join(['%.2f'%v for v in mean])
print 'Precison: ' + ' '.join(['%.2f'%v for v in prec])
print



# Back to the starting state...
print 'Original...'
solver.enable(4)

iters = solver.solve_trws()
print 'iters =', iters

mean, prec = solver.result()

print 'Mean:     ' + ' '.join(['%.2f'%v for v in mean])
print 'Precison: ' + ' '.join(['%.2f'%v for v in prec])
print
