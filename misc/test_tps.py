#! /usr/bin/env python
# Copyright 2016 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import numpy
from scipy.misc import imsave

from tps import TPS



# Quick test that the thin splate spline class works...



# Build a model - extrema of a sine curve...
x = numpy.array([0.0, 31.0, 63.0, 95.0, 127.0])
y = numpy.array([16.0, 48.0, 16.0, 48.0, 16.0])

model = TPS(1)
model.learn(x[:,None], y)



# Sample and create an image...
sx = numpy.arange(127)
sy = model(sx[:,None]).astype(numpy.int32)

image = numpy.zeros((64, 128), dtype=numpy.float32)
image[sy,sx] = 1.0
image[y.astype(numpy.int32), x.astype(numpy.int32)] = 2.0



# Save image out...
imsave('test_tps.png', image)
