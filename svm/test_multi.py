#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2010 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import svm
import numpy

from utils.prog_bar import ProgBar


pList = svm.ParamsRange()
pList.setCList(map(lambda x:10.0**x,xrange(-3,4)))
#pList.setKernelList([svm.Kernel.polynomial])
#pList.setP1List([2,3])


ds = svm.Dataset()

r = numpy.random.random((25,2))
s = numpy.random.random((25,2))
t = numpy.random.random((25,2))
u = numpy.random.random((25,2))
r[:,0] -= 0.8
t[:,0] -= 0.8
t[:,1] -= 0.8
u[:,1] -= 0.8

ds.addMatrix(r,['r']*r.shape[0])
ds.addMatrix(s,['s']*s.shape[0])
ds.addMatrix(t,['t']*t.shape[0])
ds.addMatrix(u,['u']*u.shape[0])


p = ProgBar()
mm = svm.MultiModel(pList,ds,callback=p.callback)
del p


print 'r success =',len(filter(lambda x:x=='r', mm.multiClassify(r))) / float(r.shape[0])
print 's success =',len(filter(lambda x:x=='s', mm.multiClassify(s))) / float(s.shape[0])
print 't success =',len(filter(lambda x:x=='t', mm.multiClassify(t))) / float(t.shape[0])
print 'u success =',len(filter(lambda x:x=='u', mm.multiClassify(u))) / float(u.shape[0])

print 'models ='
for params in mm.paramsList():
  print params
