#! /usr/bin/env python
# Copyright 2014 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import numpy
from gbp import GBP



# Create a graph, solve it, print out the result...
solver = GBP(9, 2) # Force it to do lots of blocking, so we can test that as well.

solver.unary(4, 2.0, 10.0)
solver.pairwise([4, 4, 3, 3, 5, 5], [3, 5, 0, 6, 2, 8], 1.0, 5.0)
solver.solve()

print 'H:'
mean, prec = solver.result()
print '% .3f % .3f % .3f' % (mean[0], mean[1], mean[2])
print '% .3f % .3f % .3f' % (mean[3], mean[4], mean[5])
print '% .3f % .3f % .3f' % (mean[6], mean[7], mean[8])



# Now reset everything...
solver.reset_unary()
solver.reset_pairwise()



# Add new links and data, solve again...
solver.unary(0, -5.0, 10.0)
solver.unary(4, 7.0, 10.0)
solver.pairwise([0, 1, 2, 5], [1, 2, 5, 8], 1.0, 5.0)
solver.pairwise([0, 3, 6, 7], [3, 6, 7, 8], 1.0, 5.0)

solver.solve()

print 'Circle:'
mean, prec = solver.result()
print '% .3f % .3f % .3f' % (mean[0], mean[1], mean[2])
print '% .3f % .3f % .3f' % (mean[3], mean[4], mean[5])
print '% .3f % .3f % .3f' % (mean[6], mean[7], mean[8])



# Reset all edges that leave a single vertex...
solver.reset_unary(4)
solver.reset_pairwise(8)



# Adjust to another graph, solve again...
solver.pairwise([5, 7, 4], [4, 4, 8], 1.0, 5.0)

solver.solve()

print 'Q:'
mean, prec = solver.result()
print '% .3f % .3f % .3f' % (mean[0], mean[1], mean[2])
print '% .3f % .3f % .3f' % (mean[3], mean[4], mean[5])
print '% .3f % .3f % .3f' % (mean[6], mean[7], mean[8])



# Reset the edges between some nodes...
solver.reset_unary(0)
solver.reset_pairwise([0, 0], [1, 3])



# New graph, solve one last time...
solver.unary(8, 9, 10.0)
solver.pairwise(0, 4, -1.5, 10.0)
solver.solve()

print 'Meh:'
mean, prec = solver.result()
print '% .3f % .3f % .3f' % (mean[0], mean[1], mean[2])
print '% .3f % .3f % .3f' % (mean[3], mean[4], mean[5])
print '% .3f % .3f % .3f' % (mean[6], mean[7], mean[8])
