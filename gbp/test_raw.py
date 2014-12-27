#! /usr/bin/env python
# Copyright 2014 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import numpy
from gbp import GBP



# Verify the raw access methods that utilise p-mean/p-offset work...
solver = GBP(2)
solver.unary_raw(0,  30.0 / 0.01, 1.0 / 0.01)
solver.pairwise_raw(0, 1, 60.0 / 4.0 , 1.0 / 4.0)

solver.solve()

pmean, prec = solver.result_raw(slice(solver.node_count))

for i in xrange(solver.node_count):
  print '%i: mean = %f, sd = %f' % (i, pmean[i]/prec[i], 1.0 / numpy.sqrt(prec[i]))
