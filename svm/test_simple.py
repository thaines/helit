#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2010 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import svm
import numpy

p = svm.Params()
#p.setPoly(2)

ds = svm.Dataset()

r = numpy.random.random((10,2))
s = numpy.random.random((15,2))
r[:,0] -= 0.8
ds.addMatrix(r,['neg']*10)
ds.addMatrix(s,['pos']*15)

m = svm.solvePair(p,ds,'neg','pos')

print 'Negative:',m.multiClassify(r)
print 'Positive:',m.multiClassify(s)

print 'loo:', svm.looPair(p,ds.getTrainData('neg','pos'))[0]
