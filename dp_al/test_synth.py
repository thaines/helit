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
import numpy
import os
import shutil

from utils.prog_bar import ProgBar
from p_cat.p_cat import ClassifyGaussian
from pool import Pool



# Simple synthetic test and comparison of assorted active learning methods.

# Parameters...
train = [(-3.0,0.125,2,'a'), (-2.0,0.25,4,'b'), (-1.0,0.5,8,'c'), (0.0,1.0,128,'d'), (1.0,0.5,8,'e'), (2.0,0.25,4,'f'), (3.0,0.125,2,'g')]
test = [(-3.0,0.125,32,'a'), (-2.0,0.25,32,'b'), (-1.0,0.5,32,'c'), (0.0,1.0,32,'d'), (1.0,0.5,32,'e'), (2.0,0.25,32,'f'), (3.0,0.125,32,'g')]

runs = 8
out_dir = 'synth'



def sampleClass(mean, sd, count, cat):
  """Generates samples from a Gaussian as a list of tuples of (1D numpy vectors, cat)"""
  ret = []
  for _ in xrange(count):
    x = random.normalvariate(mean, sd)
    ret.append((numpy.array([x], dtype=numpy.float32),cat))
  return ret


def sampleWorld(spec):
  """Given a world specification - see the definitions of the global variables train and test for an example, this returns all the samples in it."""
  ret = []
  for s in spec:
    ret += sampleClass(s[0],s[1],s[2],s[3])
  return ret


def fullPool(spec):
  """Creates a pool and fills it with the world drawn from the provided specification, before returning it."""
  pool = Pool()
  
  data = sampleWorld(spec)
  for pair in data:
    pool.store(pair[0], pair[1])

  return pool



def drainPool(pool, method, test):
  """Given a full pool and a method, as a string to be passed to the select method of the Pool object, this tests that method. Should also be given a test set of samples. Returns a tuple of 3 lists. The first is when each category was discovered, consisting of pairs of (iteration, cat), sorted. Note that iteration is 1 based, as it is the number of querys made. The second is a list indexed by the number of queries that gives the number of categories that have been found by that point. Finally, the third gives the inlier rate for the test set as indexed by the number of queries made."""
  retQ = []
  retK = [0]
  retT = [0.0]
  found = dict()
  i = 0

  classifier = ClassifyGaussian(1)

  for entity in pool.data():
    classifier.priorAdd(entity.sample)
  
  while not pool.empty():
    pool.update(classifier)
    res = pool.select(method)
    
    i += 1

    if res.ident not in found:
      retQ.append((i,res.ident))
      found[res.ident] = True
    retK.append(len(found))

    classifier.add(res.sample,res.ident)

    inliers = 0
    for t in test:
      if classifier.getCat(t[0])==t[1]:
        inliers += 1
    retT.append(float(inliers)/float(len(test)))
      
  return (retQ,retK,retT)



def drainPools(train, test, method, runs, callback=None):
  """Does a given number of runs, where it creates a new world and pool for each - in the end returns the same information as drainPool, but averaged over multiple runs. For the first piece of information the category is thrown away, such that it becomes the average number of iterations to find the 1st, 2nd etc category."""
  retQ = None
  retK = None
  retT = None


  for r in xrange(runs):
    if callback!=None: callback(r,runs)

    q, k, t = drainPool(fullPool(train), method, test)
    q = [0] + map(lambda x: x[0], q)
    
    if retQ==None: retQ = q
    else: retQ = map(lambda a,b: a + b, retQ, q)

    if retK==None: retK = k
    else: retK = map(lambda a,b: a + b, retK, k)

    if retT==None: retT = t
    else: retT = map(lambda a,b: a + b, retT, t)


  retQ = map(lambda x: float(x) / float(runs), retQ)
  retK = map(lambda x: float(x) / float(runs), retK)
  retT = map(lambda x: float(x) / float(runs), retT)

  return (retQ, retK, retT)



# Iterate the methods and run each through the classDiscover function, printing out stats...
def drainAllPools(train, test, methods, runs, base):
  for method in methods:
    print 'method = %s'%method
    p = ProgBar()
    aq, ak, at = drainPools(train, test, method, runs, p.callback)
    del p

    print 'Average # querys by # classes found:'
    for i, aqc in enumerate(aq):
      if i>1: print '  %i classes found: average of %.2f querys'%(i, aqc)
    print

    fn = '%s%s_p%i_r%i.csv'%(base,method,sum(map(lambda x: x[2], train)),runs)
    f = open(fn,'w')
    
    f.write('querys, classes, inlier\n')
    for i in xrange(len(ak)): f.write('%i, %f, %f\n'%(i, ak[i], at[i]))

    f.close()



# Run through the test sequence...
try: shutil.rmtree(out_dir)
except: pass
os.makedirs(out_dir)

drainAllPools(train, sampleWorld(test), Pool.methods(), runs, out_dir+'/')
