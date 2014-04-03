#! /usr/bin/env python
# Copyright 2014 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import numpy
from gbp import GBP



# Chain 1...
print 'Stretching to hit endpoints:'
solver = GBP(8)

solver.unary(0, 0.0, 5.0)
solver.unary(7, 10.0, 5.0)
solver.pairwise(slice(None,-1), slice(1, None), 0.0, 1.0)

iters = solver.solve()
print 'iters =', iters

mean, prec = solver.result()

print 'Mean:     ' + ' '.join(['%.2f'%v for v in mean])
print 'Precison: ' + ' '.join(['%.2f'%v for v in prec])
print



# Chain 2...
print 'Compressing to hit endpoints:'
solver = GBP(8)

solver.unary(0, 0.0, 5.0)
solver.unary(7, 10.0, 5.0)
solver.pairwise(slice(None,-1), slice(1, None), 2.0, 1.0)

iters = solver.solve()
print 'iters =', iters

mean, prec = solver.result()

print 'Mean:     ' + ' '.join(['%.2f'%v for v in mean])
print 'Precison: ' + ' '.join(['%.2f'%v for v in prec])
print



# Chain 3...
print 'Single pivot:'
solver = GBP(8)

solver.unary(0, 0.0, 5.0)
solver.pairwise(slice(None,-1), slice(1, None), 1.0, 1.0)

iters = solver.solve()
print 'iters =', iters

mean, prec = solver.result()

print 'Mean:     ' + ' '.join(['%.2f'%v for v in mean])
print 'Precison: ' + ' '.join(['%.2f'%v for v in prec])
print



# Chain 3...
print 'Snapped:'
solver = GBP(8)

solver.unary(0, 0.0, 5.0)
solver.unary(7, 3.0, 5.0)
solver.pairwise(slice(None,-1), slice(1, None), 1.0, 1.0)
solver.reset_pairwise(3, 4)

iters = solver.solve()
print 'iters =', iters

mean, prec = solver.result()

print 'Mean:     ' + ' '.join(['%.2f'%v for v in mean])
print 'Precison: ' + ' '.join(['%.2f'%v for v in prec])
print
