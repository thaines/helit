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



# Solve using a GBP object...
gbp = solve_sym(a, b)
iters = gbp.solve()
x_calc, x_prec = gbp.result()



# Print out the details...
print 'a ='
print a
print 'b =', b
print
print 'det =', numpy.linalg.det(a)
print 'iters =', iters
print
print 'true x =', x
print 'lina x =', numpy.linalg.solve(a,b)
print 'calc x =', x_calc
print '    prec =', x_prec
