#! /usr/bin/env python

# Copyright 2011 Tom SF Haines

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.



import math
import random
import os
import shutil

import numpy
import numpy.random
import cv

import ddhdp
print 'Algorithm =', ddhdp.getAlgorithm()


from utils.prog_bar import ProgBar
from utils.cvarray import *



# A Delta-Dual-HDP test. Extends the original Dual-HDP test that has a 5x5 grid with 10 topics - the horizontal and vertical lines of grid locations, which form clusters. Adds a bunch of abnormalities and trains for them.



# Details...
abList = ['square','hor_star','vert_star','hor_in_vert','vert_in_hor']


# Parameters...
samples_per_doc = 256 # Number of samples in a document.

normal_doc_train = 256 # Number of normal documents for training.
square_doc_train = 0
hor_star_doc_train = 0
vert_star_doc_train = 0
hor_in_vert_doc_train = 16
vert_in_hor_doc_train = 0

docGridWidth = 10 # Width of the grid into which documents are written to disk
gridScale = 5
gridSpace = 2

normal_doc_test = 20
square_doc_test = 0
hor_star_doc_test = 0
vert_star_doc_test = 0
hor_in_vert_doc_test = 20
vert_in_hor_doc_test = 0

abStrength = 0.2
abMark = True



# Code to sample a document...
def sampleTopic(topic):
  r = random.randrange(5)
  if topic<5: # Vertical.
    return topic*5 + r
  else: # Horizontal.
    return r*5 + (topic-5)


def sampleDocument(abnormal = None):
  # Create the multinomial to draw topics from...
  ab = numpy.zeros(5)
  vertical = random.choice([True,False])
  
  if abnormal=='square':
    ab[0] = abStrength
  if abnormal=='hor_star':
    ab[1] = abStrength
    vertical = False
  if abnormal=='vert_star':
    ab[2] = abStrength
    vertical = True
  if abnormal=='hor_in_vert':
    ab[3] = abStrength
    vertical = True
  if abnormal=='vert_in_hor':
    ab[4] = abStrength
    vertical = False
  
  multi = numpy.random.mtrand.dirichlet(numpy.ones(5))
  if vertical:
    multi = numpy.hstack(((1.0-abStrength)*multi,numpy.zeros(5),ab))
  else:
    multi = numpy.hstack((numpy.zeros(5),(1.0-abStrength)*multi,ab))
  multi /= multi.sum()

  # Draw sample counts...
  counts = numpy.random.multinomial(samples_per_doc,multi)

  # Generate and return the resulting dictionary...
  ret = dict()
  for t in xrange(10):
    for _ in xrange(counts[t]):
      word = sampleTopic(t)
      if word in ret: ret[word] += 1
      else: ret[word] = 1

  for _ in xrange(counts[10]): # Square
    word = random.randrange(4) + 6
    if word>7: word += 3
    if word in ret: ret[word] += 1
    else: ret[word] = 1

  for _ in xrange(counts[11]): # Horizontal star
    word = random.choice([0,2,6,10,12])
    if word in ret: ret[word] += 1
    else: ret[word] = 1

  for _ in xrange(counts[12]): # Vertical star
    word = random.choice([6,8,12,16,18])
    if word in ret: ret[word] += 1
    else: ret[word] = 1

  for _ in xrange(counts[13]): # Horizontal in vertical
    word = sampleTopic(9)
    if word in ret: ret[word] += 1
    else: ret[word] = 1

  for _ in xrange(counts[14]): # Vertical in horizontal
    word = sampleTopic(3)
    if word in ret: ret[word] += 1
    else: ret[word] = 1

  return ret


# Convert a document as a dictionary to a grid shaped numpy array...
def docDicToArray(docDic):
  ret = numpy.zeros((5,5), dtype=numpy.float_)
  for key,value in docDic.iteritems():
    x = key//5
    y = key%5
    ret[x,y] += value
  ret /= ret.max()
  return ret



# Create a corpus...
corpus = ddhdp.Corpus()
corpus.setWordCount(5*5)



# Fill in the documents...
docs = []

behClasses = [None] + abList
trainCounts = [normal_doc_train, square_doc_train, hor_star_doc_train, vert_star_doc_train, hor_in_vert_doc_train, vert_in_hor_doc_train]
for c in xrange(len(behClasses)):
  for _ in xrange(trainCounts[c]):
    ab = behClasses[c]
    docDic = sampleDocument(ab)
    docs.append(docDicToArray(docDic))
    doc = ddhdp.Document(docDic,[ab] if (abMark and ab!=None) else [])
    corpus.add(doc)



# (Re)create the output directory...
try: shutil.rmtree('test_abnorm_lines')
except: pass
os.makedirs('test_abnorm_lines')



# Save the documents to an image...
docs = map(lambda d: numpy.repeat(numpy.repeat(d,gridScale,axis=1),gridScale,axis=0),docs)

rows = []
for i in xrange(sum(trainCounts)//docGridWidth):
  row = docs[i*docGridWidth:(i+1)*docGridWidth]
  rowExt = []
  for r in row:
    rowExt.append(r)
    rowExt.append(numpy.zeros((gridSpace,gridScale*5), dtype=numpy.float_))
  rowExt = rowExt[:-1]
  rows.append(numpy.vstack(rowExt))

stack = []
for r in rows:
  stack.append(r)
  stack.append(numpy.zeros((r.shape[0],gridSpace), dtype=numpy.float_))
stack = stack[:-1]
docImage = numpy.hstack(stack).T * 255.0
img = array2cv(docImage)
cv.SaveImage('test_abnorm_lines/docs.png',img)



# Train...
params = ddhdp.Params()
params.runs = 1
params.samples = 1
#params.burnIn = 10000
#c.setOneCluster(True)


print 'Fitting model...'
p = ProgBar()
model = corpus.sampleModel(params,p.callback)
del p

#model.bestSampleOnly()


sam = model.getSample(0)

def smartVecPrint(numVec):
  ret = []
  ret.append('[')
  for i in xrange(numVec.shape[0]):
    ret.append('%s%.3f'%(' ' if i!=0 else '',numVec[i]))
  ret.append(']')
  return ''.join(ret)

print 'Discovered topics:', sam.getTopicCount()
print 'Discovered clusters:', sam.getClusterCount()
print 'Phi:', smartVecPrint(sam.getPhi())
for c in xrange(sam.getClusterCount()):
  print 'cluster %i bmn:'%c, smartVecPrint(sam.getClusterInstBehMN(c))



# Visualise the topics...
for t in xrange(sam.getTopicCount()):
  multi = sam.getTopicMultinomial(t)
  dicForm = dict()
  for i in xrange(multi.shape[0]): dicForm[i] = multi[i]
  topicGrid = docDicToArray(dicForm).T
  topicGrid = numpy.repeat(numpy.repeat(topicGrid,gridScale,axis=1),gridScale,axis=0)
  img = array2cv(topicGrid*255.0)
  cv.SaveImage('test_abnorm_lines/topic_%i.png'%t,img)


# Visualise the abnormal topics...
for key, index in sam.getAbnormDict().iteritems():
  multi = sam.getAbnormMultinomial(index)
  dicForm = dict()
  for i in xrange(multi.shape[0]): dicForm[i] = multi[i]
  topicGrid = docDicToArray(dicForm).T
  topicGrid = numpy.repeat(numpy.repeat(topicGrid,gridScale,axis=1),gridScale,axis=0)
  img = array2cv(topicGrid*255.0)
  cv.SaveImage('test_abnorm_lines/ab_topic_%s.png'%key,img)


# Provide some information on the clusters...
out = open('test_abnorm_lines/clusters.txt','w')
out.write('cluster draw conc = %f\n\n'%sam.getClusterDrawConc())

for c in xrange(sam.getClusterCount()):
  out.write('cluster %i:\n'%c)
  out.write('weight = %i\n'%sam.getClusterDrawWeight(c))
  out.write('inst count = %i\n'%sam.getClusterInstCount(c))
  out.write('conc = %f\n'%sam.getClusterInstConc(c))
  for ci in xrange(sam.getClusterInstCount(c)):
    out.write('inst %i: {topic = %i, count = %i}\n'%(ci, sam.getClusterInstTopic(c,ci), sam.getClusterInstWeight(c,ci)))
  out.write('\n')
out.close()



# Provide a visualisation of the clusters...
cImg = []
maxWeight = max(map(lambda c: sam.getClusterDrawConc() + sam.getClusterDrawWeight(c), xrange(sam.getClusterCount())))

topicWidth = 10
topicHeight = 50
weightHeight = 10
spacer = 5

twl = []
twlMax = 0.0

for c in xrange(sam.getClusterCount()):
  tWeight = numpy.zeros(sam.getTopicCount(),dtype=numpy.float_)

  wSum = sam.getTopicConc()
  for t in xrange(sam.getTopicCount()):
    tWeight[t] += sam.getClusterInstConc(c) * sam.getTopicUseWeight(t)
    wSum += sam.getTopicUseWeight(t)
  tWeight /= wSum

  for ci in xrange(sam.getClusterInstCount(c)):
    tWeight[sam.getClusterInstTopic(c,ci)] += sam.getClusterInstWeight(c,ci)

  twl.append(tWeight)
  twlMax = max((twlMax,tWeight.max()))

for c in xrange(sam.getClusterCount()):
  tWeight = twl[c] / twlMax
  
  img = numpy.zeros((topicHeight+weightHeight+spacer,topicWidth*sam.getTopicCount(),3), dtype=numpy.float_)

  offset = int(topicWidth*sam.getTopicCount()*(sam.getClusterDrawConc() + sam.getClusterDrawWeight(c))/maxWeight)
  img[topicHeight:topicHeight+weightHeight,0:offset,1] = 1.0

  for t in xrange(sam.getTopicCount()):
    col = 0 if (t%2==0) else 2
    h = int(tWeight[t]*topicHeight)
    img[topicHeight-h:topicHeight,t*topicWidth:(t+1)*topicWidth,col] = 1.0
  
  cImg.append(img)

cluImg = numpy.vstack(cImg) * 255.0
cv.SaveImage('test_abnorm_lines/clusters.png', array2cv(cluImg))



# Now create some documents and test them...
# Below indexed by [actual, result], with 0 to mean normal and 1 for abnormal...
print 'Testing model on new documents...'
result = numpy.zeros((6,6),dtype=numpy.int32)

testCounts = [normal_doc_test, square_doc_test, hor_star_doc_test, vert_star_doc_test, hor_in_vert_doc_test, vert_in_hor_doc_test]

p = ProgBar()
stepsDone = 0
stepsTotal = sum(testCounts)
for ab in ([None] + [x for x in sam.getAbnormDict().iterkeys()]):
  c = behClasses.index(ab)
  for _ in xrange(testCounts[c]):
    p.callback(stepsDone,stepsTotal)
    stepsDone += 1

    doc = ddhdp.Document(sampleDocument(ab))
    abnormList = model.mlDocAbnorm(doc, lone = True, cap = 0) ######## Cap is super low for testing.

    truth = c
    guess = 0
    if len(abnormList)!=0:
      guess = behClasses.index(abnormList[0]) # Doesn't handle it thinking that multiple abnormalities are present.

    result[truth,guess] += 1
del p


print 'Confusion matrix:'
print result
print


for c in xrange(len(behClasses)):
  name = behClasses[c]
  if name==None: name = 'normal'

  total = result[c,:].sum()
  trueCount = result[c,c]
  falseCount = total - trueCount

  if total!=0:
    print name,'behaviour:'
    print 'True %s = %i (%.1f%%)' % (name,trueCount,100.0*float(trueCount)/total)
    print 'False not %s = %i (%.1f%%)' % (name,falseCount,100.0*float(falseCount)/total)
    print
