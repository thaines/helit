#! /usr/bin/env python

# Copyright 2014 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import frf
from sklearn.ensemble import RandomForestClassifier

import numpy
from utils.prog_bar import ProgBar



# A toy problem using continuous attributes only - a simulation of a politician/marketing/tele-sales differentiator.



# Define the parameters for the drawing of the attributes...
class Style:
  Loudness = 0
  Speed = 1

  length = 2

  politician_mean = (1.0,0.0)
  politician_covar = ((0.3,0.0),(0.0,0.3))

  marketing_mean = (0.5,0.8)
  marketing_covar = ((0.5,0.0),(0.0,0.5))

  tele_sales_mean = (0.5,0.3)
  tele_sales_covar = ((0.2,0.0),(0.0,1.0))


class Content:
  Bullshit = 0
  Lies = 1
  Exaggeration = 2

  length = 3

  politician_mean = (1.0,1.0,0.9)
  politician_covar = ((0.1,0.5,0.5),(0.5,0.1,0.5),(0.5,0.5,0.1))

  marketing_mean = (0.9,0.4,1.0)
  marketing_covar = ((0.3,0.4,-0.4),(0.4,0.2,0.4),(-0.4,0.1,0.4))

  tele_sales_mean = (0.7,0.7,0.7)
  tele_sales_covar = ((0.5,0.0,0.0),(0.0,1.0,0.0),(0.0,0.0,0.6))


class Clothes:
  Suit = 0
  Casual = 1

  length = 2

  politician_mean = (1.0,0.0)
  politician_covar = ((0.1,-1.0),(-1.0,0.2))

  marketing_mean = (0.5,0.5)
  marketing_covar = ((0.3,-0.1),(-0.1,0.2))

  tele_sales_mean = (0.0,1.0)
  tele_sales_covar = ((0.4,-0.5),(-0.5,0.4))


attributes = [Style, Content, Clothes]



# Functions to generate examples of the three classes...
def make_politician():
  length = sum(map(lambda a: a.length, attributes))
  ret = numpy.empty(length, dtype=numpy.float32)

  offset = 0
  for att in attributes:
    ret[offset:offset+att.length] = numpy.random.multivariate_normal(att.politician_mean, att.politician_covar)
    offset += att.length

  return ret

def make_marketing():
  length = sum(map(lambda a: a.length, attributes))
  ret = numpy.empty(length, dtype=numpy.float32)

  offset = 0
  for att in attributes:
    ret[offset:offset+att.length] = numpy.random.multivariate_normal(att.marketing_mean, att.marketing_covar)
    offset += att.length

  return ret

def make_tele_sales():
  length = sum(map(lambda a: a.length, attributes))
  ret = numpy.empty(length, dtype=numpy.float32)

  offset = 0
  for att in attributes:
    ret[offset:offset+att.length] = numpy.random.multivariate_normal(att.tele_sales_mean, att.tele_sales_covar)
    offset += att.length

  return ret



# Generate the trainning set...
politician_count = 2048
marketing_count = 2048
tele_sales_count = 2048
total_count = politician_count + marketing_count + tele_sales_count

feat_length = sum(map(lambda a: a.length, attributes))
dm = numpy.empty((total_count, feat_length), dtype=numpy.float32)
cat = numpy.empty(total_count, dtype=numpy.int32)

for i in xrange(total_count):
  if i<politician_count:
    dm[i,:] = make_politician()
    cat[i] = 0
  elif i<(politician_count+marketing_count):
    dm[i,:] = make_marketing()
    cat[i] = 1
  else:
    dm[i,:] = make_tele_sales()
    cat[i] = 2



# Train the frf model...
forest = frf.Forest()
forest.configure('C', 'C', 'S' * dm.shape[1])
forest.min_exemplars = 4
forest.opt_features = int(numpy.sqrt(dm.shape[1]))

pb = ProgBar()
oob = forest.train(dm, cat, 64, pb.callback)
del pb

print 'Made frf forest (oob = %.2f%%):' % ((1.0 - oob.mean()) * 100.0)
for i in xrange(min(len(forest),4)):
  if oob!=None:
    extra = ', oob = %.2f%%' % ((1.0 - oob[i,0]) * 100.0)
  else:
    extra = ''
    
  print '  Tree %i: %i bytes, %i nodes%s' % (i, forest[i].size, forest[i].nodes(), extra)
print



# Train the scikit learn model...
model = RandomForestClassifier(64, 'entropy', min_samples_leaf=4, oob_score=True)

pb = ProgBar()
model.fit(dm, cat)
del pb

print 'Made scikit learn forest (oob = %.2f%%):' % (model.oob_score_ * 100.0)
for i in xrange(min(len(model.estimators_),4)):
  print '  Tree %i: %i nodes' % (i, model.estimators_[i].tree_.feature.shape[0])
print



# Test...
data = numpy.empty((256, dm.shape[1]))
for i in xrange(data.shape[0]):
  data[i,:] = make_politician()

res_frf = forest.predict(data)[0]
correct_frf = (numpy.argmax(res_frf['prob'], axis=1)==0).sum()

correct_scikit = (model.predict(data)==0).sum()

print 'Of %i politicians:' % data.shape[0]
print '    frf: %i (%.1f%%) were correctly detected.'%(correct_frf, 100.0*correct_frf/float(data.shape[0]))
print '    scikit: %i (%.1f%%) were correctly detected.'%(correct_scikit, 100.0*correct_scikit/float(data.shape[0]))
print


data = numpy.empty((256, dm.shape[1]))
for i in xrange(data.shape[0]):
  data[i,:] = make_marketing()

res_frf = forest.predict(data)[0]
correct_frf = (numpy.argmax(res_frf['prob'], axis=1)==1).sum()

correct_scikit = (model.predict(data)==1).sum()

print 'Of %i marketers:' % data.shape[0]
print '    frf: %i (%.1f%%) were correctly detected.'%(correct_frf, 100.0*correct_frf/float(data.shape[0]))
print '    scikit: %i (%.1f%%) were correctly detected.'%(correct_scikit, 100.0*correct_scikit/float(data.shape[0]))
print


data = numpy.empty((256, dm.shape[1]))
for i in xrange(data.shape[0]):
  data[i,:] = make_tele_sales()

res_frf = forest.predict(data)[0]
correct_frf = (numpy.argmax(res_frf['prob'], axis=1)==2).sum()

correct_scikit = (model.predict(data)==2).sum()

print 'Of %i tele-sellers:' % data.shape[0]
print '    frf: %i (%.1f%%) were correctly detected.'%(correct_frf, 100.0*correct_frf/float(data.shape[0]))
print '    scikit: %i (%.1f%%) were correctly detected.'%(correct_scikit, 100.0*correct_scikit/float(data.shape[0]))
print
