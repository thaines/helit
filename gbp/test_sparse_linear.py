#! /usr/bin/env python
# Copyright 2014 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import numpy
from linear import solve_sym



# Create a suitably large a x = b problem, but really sparse...
dims = 64
terms = dims

a = numpy.zeros((dims, dims), dtype=numpy.float32)
for _ in xrange(terms):
  cx = numpy.random.randint(dims)
  cy = numpy.random.randint(dims)
  val = numpy.random.normal()
  
  a[cx,cy] = val
  a[cy,cx] = val

x = numpy.random.normal(size=dims)

b = a.dot(x)



# Solve using a GBP object...
gbp = solve_sym(a, b)
iters = gbp.solve_trws(1024*1024)
x_calc, x_prec = gbp.result()



# Communicate the output...
#print 'Ground truth:', x
#print 'Output:', x_calc

dist = numpy.sqrt(numpy.square(x-x_calc).sum())
diff = numpy.fabs(x-x_calc)

print '%i iterations' % iters
print 'Euclidean distance to ground truth = %f' % dist

for threshold in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0]:
  print '# dims within %.6f: %i' % (threshold, (diff<threshold).sum())
print 'Total dims = %i' % dims
