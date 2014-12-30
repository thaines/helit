#! /usr/bin/env python
# Copyright 2014 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import numpy
from linear import solve_sym



# Create a a x = b problem with a symmetric a matrix...
dim = 4
a = numpy.random.normal(size=(dim,dim))
a += a.T

x = numpy.random.normal(size=dim)

b = a.dot(x)



# Solve using Jacobi iterations, as they are very similar to what we are doing here...
jacobi_x = numpy.zeros(x.shape, dtype=numpy.float32)
diag = a[range(dim), range(dim)]
zero_diag = a.copy()
zero_diag[range(dim), range(dim)] = 0.0

for _ in xrange(1024):
  jacobi_x = (b - zero_diag.dot(jacobi_x)) / diag



# Solve using a GBP object, both approaches...
gbp1 = solve_sym(a, b)
gbp2 = gbp1.clone()

bp_iters = gbp1.solve()
bp_x_calc, bp_x_prec = gbp1.result()

trws_iters = gbp2.solve_trws()
trws_x_calc, trws_x_prec = gbp2.result()



# Determine if the matrix is diagonally dominant...
diag = numpy.fabs(a[xrange(dim),xrange(dim)])
offdiag = numpy.fabs(a).sum(axis=0) - diag
diag_dom = numpy.all(diag>offdiag)

# Calculate its spectral radius...
spec_rad = numpy.linalg.eig(numpy.eye(dim) - a)[0].max()



# Print out the details...
print 'a ='
print a
print 'b =', b
print
print 'diagonally dominant =', diag_dom
print 'spectral radius =', spec_rad, '(too large)' if spec_rad>=1.0 else ''
print 'det =', numpy.linalg.det(a)
print 'bp iters =', bp_iters
print 'trws iters =', trws_iters
print
print 'true x =', x
print 'lina x =', numpy.linalg.solve(a,b)
print 'Jacobi x =', jacobi_x
print 'bp calc x =', bp_x_calc
#print '  bp prec =', bp_x_prec
print 'trws calc x =', trws_x_calc
#print '  trws prec =', trws_x_prec
