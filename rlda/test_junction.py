#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2010 Tom SF Haines

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

import numpy
import cv

import rlda
print 'Algorithm =', rlda.getAlgorithm()


import sys
sys.path.append(sys.path[0]+'/..')
from utils.prog_bar import ProgBar
from utils.cvarray import *



# Simulation of a crossroads where cars never turn right, except when they do, which should hopefully flash up an abnormality. Don't bother with seperating lanes, use 6x6 grid, but with 4 2x2 patches where cars never travel that are hence unsused, so 20 identifiers, 4 topics, 5 regions, 4 words.



# Parameters...
sweep = False

lowTDC = 1
highTDC = 30
runs = 128

sampleCount = 100
abnormalityChance = 1.0/20.0
trainDocCount = 100
testDocCount = 100



# Mappings from grid coordinates to identifier and back again, that remove the corner 2x2's from the system...

c2i = [[-1,-1, 0, 1,-1,-1],
       [-1,-1, 2, 3,-1,-1],
       [ 4, 5, 6, 7, 8, 9],
       [10,11,12,13,14,15],
       [-1,-1,16,17,-1,-1],
       [-1,-1,18,19,-1,-1]]

i2c = [(2,0), (3,0), (2,1), (3,1), (0,2), (1,2), (2,2), (3,2), (4,2), (5,2), (0,3), (1,3), (2,3), (3,3), (4,3), (5,3), (2,4), (3,4), (2,5), (3,5)]

def coordToIdent(x,y):
  return c2i[y][x]

def identToCoord(i):
  return i2c[i]

def identCount():
  return len(i2c)



# The 5 sets of locations, as identifiers...
locs = [[0,1,2,3], # 4 inroads, clockwise
        [8,9,14,15],
        [16,17,18,19],
        [4,5,10,11],
        [6,7,12,13]] # Junction



# Code to generate examples from each topic, plus the abnormalities, return (ident,word) pairs...
def sampleTopic(topic): # topic in [0,1,2,3]
  l = random.randrange(6)
  if l==0 or l==1: # Entering junction
    return (random.choice(locs[topic]),topic)
  elif l==2: # Straight over junction.
    return (random.choice(locs[4]),topic)
  elif l==3: # Turning left at junction
    return (random.choice(locs[4]),(topic+3)%4)
  elif l==4: # Exiting after straight over
    return (random.choice(locs[(topic+2)%4]),topic)
  elif l==5: # Exiting after turning left
    return (random.choice(locs[(topic+1)%4]),(topic+3)%4)

def sampleAbnormalTopic(topic):
  l = random.randrange(3)
  if l==0: # Entering junction
    return (random.choice(locs[topic]),topic)
  elif l==1: # Turning right at junction
    return (random.choice(locs[4]),(topic+1)%4)
  elif l==2: # Exiting after turning right
    return (random.choice(locs[(topic+3)%4]),(topic+1)%4)



# Code to generate documents...
def genDoc(allowAlt = True):
  # Pick two topics (Assumption is that topics follow one another, and the time frames are short enough to only ever capture two - model can't capture this of course, but it makes code easier.)...
  topicA = random.randrange(4)
  topicB = random.randrange(4)
  if allowAlt:
    abTopicA = random.random()<abnormalityChance
    abTopicB = random.random()<abnormalityChance
  else:
    abTopicA = False
    abTopicB = False
  ratio = random.random()

  ret = dict()
  for i in xrange(sampleCount):
    if random.random()<ratio: # topic A
      if abTopicA: key = sampleAbnormalTopic(topicA)
      else: key = sampleTopic(topicA)
    else: # topic B
      if abTopicB: key = sampleAbnormalTopic(topicB)
      else: key = sampleTopic(topicB)
    if key in ret: ret[key] += 1
    else: ret[key] = 1
  return (ret,abTopicA or abTopicB)



# Function that does a single pass...
def doRun(tdc):
  # Create directory to put images into...
  if not sweep:
    try:
      os.makedirs('junction')
    except:
      pass

    
  # Create a corpus...    
  c = rlda.Corpus(10,4)
  c.setIdentWordCounts(identCount(),4)

  for i in xrange(tdc):
    dic, abn = genDoc(False)
    doc = rlda.Document(dic)
    doc.abn = abn
    c.add(doc)

    if not sweep:
      prob = numpy.zeros((6,6,4),dtype=numpy.float_)
      for key,item in dic.iteritems():
        x,y = identToCoord(key[0])
        prob[x,y,key[1]] = item

      multProb = 255.0/prob.max()
      img = cv.CreateImage((6*25,6*25),cv.IPL_DEPTH_32F,3)
      for y in xrange(6):
        for x in xrange(6):
          coords = [(x*25,y*25),((x+1)*25,y*25),((x+1)*25,(y+1)*25),(x*25,(y+1)*25)]
          centre = (x*25+12,y*25+12)
          for d in xrange(4):
            if d%2==0:
              col = cv.RGB(0.0,prob[x,y,d]*multProb,0.0)
            else:
              col = cv.RGB(prob[x,y,d]*multProb,0.0,0.0)
            cv.FillPoly(img, [(coords[d],coords[(d+1)%4],centre)], col)

      cv.SaveImage('junction/xdoc_%i_%s.png'%(i,str(abn)),img)


  # Fit a model...
  params = rlda.Params()
  params.setRuns(16)

  print 'Fitting model...'
  p = ProgBar()
  c.fit(params,p.callback)
  del p

  ir = c.getIR()
  wrt = c.getWRT()


  # Visualise the regions...
  if not sweep:
    mult = 255.0/ir.max()
    for r in xrange(ir.shape[1]):
      rend = numpy.zeros((6,6),dtype=numpy.float_)
      for i in xrange(ir.shape[0]): rend[identToCoord(i)] = ir[i,r] * mult
      rend = numpy.repeat(numpy.repeat(rend,25,axis=0),25,axis=1)
      cv.SaveImage('junction/region_%i.png'%r,array2cv(rend))
        


  # Visualise the topics...
  if not sweep:
    for t in xrange(wrt.shape[2]):
      prob = numpy.zeros((6,6,4),dtype=numpy.float_)
      for i in xrange(ir.shape[0]):
        x,y = identToCoord(i)
        for r in xrange(wrt.shape[1]):
          for w in xrange(wrt.shape[0]):
            prob[x,y,w] += ir[i,r] * wrt[w,r,t]

      multProb = 255.0/prob.max()
      img = cv.CreateImage((6*25,6*25),cv.IPL_DEPTH_32F,3)
      for y in xrange(6):
        for x in xrange(6):
          coords = [(x*25,y*25),((x+1)*25,y*25),((x+1)*25,(y+1)*25),(x*25,(y+1)*25)]
          centre = (x*25+12,y*25+12)
          for d in xrange(4):
            if d%2==0:
              col = cv.RGB(0.0,prob[x,y,d]*multProb,0.0)
            else:
              col = cv.RGB(prob[x,y,d]*multProb,0.0,0.0)
            cv.FillPoly(img, [(coords[d],coords[(d+1)%4],centre)], col)
      
      cv.SaveImage('junction/topic_%i.png'%t,img)


  # Test on a bunch of documents, creating a list of abnormality score/actually an abnormality pairs...
  ab_gt = []
  print 'Testing...'
  p = ProgBar()
  for i in xrange(testDocCount):
    p.callback(i,testDocCount)
    dic, abn = genDoc()
    doc = rlda.Document(dic)
    doc.fit(ir,wrt)
    ab_gt.append((doc.negLogLikeRegionVec().max(),abn))
  del p

  ab_gt.sort(reverse=True)


  # Use the pairs to construct a roc...
  posCount = len(filter(lambda p:p[1]==True,ab_gt))
  negCount = len(ab_gt) - posCount
  print 'positive samples = ',posCount
  print 'negative samples = ',negCount

  truePos = 0
  falsePos = 0
  trueNeg = negCount
  falseNeg = posCount

  roc = []

  for p in ab_gt:
    if p[1]:
      truePos += 1
      falseNeg -= 1
    else:
      falsePos +=1
      trueNeg -= 1

    pnt = (float(falsePos)/float(falsePos+trueNeg), float(truePos)/float(truePos+falseNeg))
    roc.append(pnt)


  # Save the roc to disk...
  if not sweep:
    f = open('junction_roc.txt','w')
    f.write('0.0 0.0\n')
    for pnt in roc: f.write('%f %f\n'%pnt)
    f.close()


  # Calculate and print out the area under the roc...
  area = 0.0
  for i in xrange(1,len(roc)):
    area += 0.5*(roc[i-1][1]+roc[i][1]) * (roc[i][0]-roc[i-1][0])
  print 'area under roc =',area, '(above',(1.0-area),')'

  return area



if not sweep:
  doRun(trainDocCount)
else:
  f = open('junction_runs.txt','w')
  for tdc in xrange(lowTDC,highTDC+1):
    f.write('%i'%tdc)
    for r in xrange(runs):
      area = doRun(tdc)
      f.write(' %f'%area)
      f.flush()
    f.write('\n')
  f.close()
