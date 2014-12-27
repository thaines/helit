# Copyright 2014 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import numpy
from gbp import GBP



def solve_sym(a, b, epsilon=1e-6):
  """Given the symmetric matrix a and the vector b this returns a GBP object such that, after solve_trws() (prefered over the default solve_bp) has been called, the result() method returns x as the mean, such that a x = b, i.e. it solves the symmetric linear equation. This is poor compared to typical solvers as it suffers from the spectral radius being less than 1 requirement (typical of iterative methods) - really exists because I can, and its a good test that the system works. Its still useful if there are a lot of zeroes in (a) however, as it is stable with enough sparseness and becomes computationally efficient, though you may want to rewrite what it does to properlly utilise a sparse matrix class if that is the case. This is an implimentation of the paper 'Gaussian Belief Propagation Solver for Systems of Linear Equations' by Shental, Siegel, Wolf, Bickson and Dolev. The use of Gaussian TRW-S instead of Gaussian BP seems to make it converge far more often than otherwise, and I suspect is weakening some of the limitations discussed in the paper - it often works when Jacobi iterations (which it is almost equivalent to for normal BP) do not. Also note that the system has the weirdness of negative precision values when used for this, i.e. imaginary standard deviations. Make of this what you will, particularly the fact these often come out alongside the correct answer!"""
  assert(len(a.shape)==2)
  assert(len(b.shape)==1)
  assert(a.shape[0]==a.shape[1])
  assert(a.shape[0]==b.shape[0])
  
  ret = GBP(b.shape[0])
  
  r = range(b.shape[0])
  ret.unary_raw(r, b, a[r,r])
  
  for i in r:
    for j in xrange(i+1, b.shape[0]):
      if numpy.fabs(a[i, j])>epsilon:
        ret.pairwise(i, j, float(a[i, j]))
  
  return ret
