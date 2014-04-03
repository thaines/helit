#! /usr/bin/env python
# Copyright 2014 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import numpy
from gbp import GBP



# Grid 1...
print 'Single Pivot:'
solver = GBP(25) #5x5

solver.unary(12, 5.0, 4.0)

for row in xrange(5):
  solver.pairwise(slice(row*5,(row+1)*5-1), slice(row*5+1,(row+1)*5), 1.0, 1.0)

for col in xrange(5):
  solver.pairwise(slice(col,col+20,5), slice(col+5,col+25,5), 1.0, 1.0)

iters = solver.solve()
print 'iters =', iters

for row in xrange(5):
  mean, prec = solver.result(slice(row*5, (row+1)*5))
  print ' '.join(['%.4f'%v for v in mean])
  print ' '.join(['(%.2f)'%v for v in prec])

print



# Grid 2...
print 'Stretch:'
solver = GBP(25) #5x5

solver.unary(0, 0.0, 5.0)
solver.unary(4, 9.0, 5.0)
solver.unary(20, 9.0, 5.0)
solver.unary(24, 0.0, 5.0)

for row in xrange(5):
  solver.pairwise(slice(row*5,(row+1)*5-1), slice(row*5+1,(row+1)*5), 0.0, 1.0)

for col in xrange(5):
  solver.pairwise(slice(col,col+20,5), slice(col+5,col+25,5), 0.0, 1.0)

iters = solver.solve()
print 'iters =', iters

for row in xrange(5):
  mean, prec = solver.result(slice(row*5, (row+1)*5))
  print ' '.join(['%.4f'%v for v in mean])
  print ' '.join(['(%.2f)'%v for v in prec])

print



# Grid 3...
print 'Compress:'
solver = GBP(25) #5x5

solver.unary(0, 7.0, 5.0)
solver.unary(4, 2.0, 5.0)
solver.unary(20, 2.0, 5.0)
solver.unary(24, 7.0, 5.0)

for row in xrange(5):
  solver.pairwise(slice(row*5,(row+1)*5-1), slice(row*5+1,(row+1)*5), 3.0, 1.0)

for col in xrange(5):
  solver.pairwise(slice(col,col+20,5), slice(col+5,col+25,5), 3.0, 1.0)

iters = solver.solve()
print 'iters =', iters

for row in xrange(5):
  mean, prec = solver.result(slice(row*5, (row+1)*5))
  print ' '.join(['%.4f'%v for v in mean])
  print ' '.join(['(%.2f)'%v for v in prec])

print



# Grid 4...
print 'Broken:'
solver = GBP(25) #5x5

solver.unary(0, 0.0, 5.0)
solver.unary(12, 100.0, 5.0)
solver.unary(24, 5.0, 5.0)

for row in xrange(5):
  solver.pairwise(slice(row*5,(row+1)*5-1), slice(row*5+1,(row+1)*5), 0.0, 1.0)

for col in xrange(5):
  solver.pairwise(slice(col,col+20,5), slice(col+5,col+25,5), 0.0, 1.0)

solver.reset_unary(12)
solver.reset_pairwise([4,8,8,12,17,17,21], [9,9,13,13,16,12,22])
#solver.pairwise(21, 22, 0.0, 0.1)

iters = solver.solve()
print 'iters =', iters

for row in xrange(5):
  mean, prec = solver.result(slice(row*5, (row+1)*5))
  print ' '.join(['%.4f'%v for v in mean])
  print ' '.join(['(%.2f)'%v for v in prec])

print
