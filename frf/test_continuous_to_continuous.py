#! /usr/bin/env python

# Copyright 2014 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import frf

import numpy



# A toy problem of learning the relationship from standard coordinates to polar coordinates.



def sample():
  """Returns a sample as x,y, both continuous length 2 vectors."""
  x = numpy.random.multivariate_normal((0.0, 0.0), ((1.0, 0.0), (0.0,1.0)))
  y = numpy.array([numpy.arctan2(x[1], x[0]), numpy.sqrt((x**2).sum())])
  
  return (x, y)


 
# Create a data set...
x = numpy.empty((1024*8, 2))
y = numpy.empty((x.shape[0], 2))

for i in xrange(x.shape[0]):
  x[i,:], y[i,:] = sample()

  
  
# Train a forest...
forest = frf.Forest()
forest.configure('BN', 'BN', 'SS')
forest.min_exemplars = 4

oob = forest.train(x, y, 8)

print 'Made forest:'
for i in xrange(len(forest)):
  if oob!=None:
    extra = ', oob = %s' % str(oob[i,:])
  else:
    extra = ''
    
  print '  Tree %i: %i bytes, %i nodes%s' % (i, forest[i].size, forest[i].nodes(), extra)
print



# Test...
x = numpy.empty((1024, 2))
y = numpy.empty((x.shape[0], 2))

for i in xrange(x.shape[0]):
  x[i,:], y[i,:] = sample()

  
print 'Test average error:'
res = forest.predict(x)[0]
#print x[0,:], y[0,:], res['mean'][0,:]
error = numpy.fabs(y - res['mean']).mean(axis=0)
print error, '(%.4f)' % numpy.sqrt((error**2).sum())



# Check the error method...
print
print 'Test error:'
print forest.error(x, y)[0][0]
