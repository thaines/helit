#! /usr/bin/env python

# Copyright 2014 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import numpy
from ms import MeanShift



# Simply creates two identical Fisher distributions, one using the approximation, the other the correct value, and then runs some tests to verify that they are the same.



# Parameters...
dims = 4
kernel = 'fisher'
conc = 256.0
samples = 1024



# Create the two distributions - don't need to be complicated...
correct = MeanShift()
correct.set_data(numpy.array([1.0] + [0.0]*(dims-1), dtype=numpy.float32), 'f')
correct.set_kernel('%s(%.1fc)' % (kernel, conc))
correct.quality = 1.0

approximate = MeanShift()
approximate.set_data(numpy.array([1.0] + [0.0]*(dims-1), dtype=numpy.float32), 'f')
approximate.set_kernel('%s(%.1fa)' % (kernel, conc))
approximate.quality = 1.0



# Draw a bunch of samples and compare the probabilities in both to check they are basically the same...
sample = correct.draws(samples)

cp = correct.probs(sample)
ap = approximate.probs(sample)

diff = numpy.fabs(cp-ap)

print 'Maximum probabilities =', cp.max(), ap.max()
print 'Maximum probability difference =', diff.max(), (diff.max() / cp.max())
print 'Average probability difference =', diff.mean(), (diff.mean() / cp.max())
