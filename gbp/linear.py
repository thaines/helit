# Copyright 2014 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import numpy
from gbp import GBP



def solve_sym(a, b, epsilon=1e-6):
  """Given the symmetric matrix a and the vector b this returns a GBP object such that, after solve() has been called, the result() method returns x as the mean, such that a x = b, i.e. it solves the symmetric linear equation. This is not as good as typical solvers as there are situations where it fails (albeit rarely badly) - really exists because I can, and its a good test the system works. Its still useful if there are a lot of zeos in (a) however, as it respects that, though you may want to rewrite what it does to properlly utlise a sparse matrix class if that is the case."""
  assert(len(a.shape)==2)
  assert(len(b.shape)==1)
  assert(a.shape[0]==a.shape[1])
  assert(a.shape[0]==b.shape[0])
  
  ret = GBP(b.shape[0])
  
  r = range(b.shape[0])
  ret.unary(r, b / a[r,r], a[r,r])
  
  for i in r:
    for j in xrange(i+1, b.shape[0]):
      if numpy.fabs(a[i, j])>epsilon:
        ret.pairwise(i, j, a[i, j])
  
  return ret
