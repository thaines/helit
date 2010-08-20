# -*- coding: utf-8 -*-

# Copyright 2010 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



from params import Params
from smo import SMO
from loo import looPair, looPairRange, looPairBrute

import math
import time
import multiprocessing as mp
import numpy



def mpLooPairRange(params, data, lNeg, lPos, looDist):
  """multiprocess wrapper around looPair needed for multiprocessing support."""
  model = looPairRange(params, data, looDist)
  return (lNeg,lPos,model[0],model[1])



class MultiModel:
  """This represents a model with multiple labels - uses one against one voting. Even if you only have two labels you are best off using this interface, as it makes everything neat. Supports model selection as well."""
  def __init__(self, params, dataset, weightSVM = True, callback = None, pool = None, looDist = 1.1):
    """Trains the model given the dataset and either a params object or a iterator of params objects. If a list it trys all entrys of the list for each pairing, and selects the one that gives the best loo, i.e. does model selection. If weightSVM is True (The default) then it makes use of the leave one out scores calculated during model selection to weight the classification boundaries - this can result in slightly better behavour at the meeting points of multiple classes in feature space. The pool parameter can be passed in a Pool() object from the multiprocessing python module, or set to True to have it create an instance itself. This enables multiprocessor mode for doing each loo calculation required - good if you have lots of models to test and/or lots of labels."""
    self.weightSVM = weightSVM

    # Get a list of labels, create all the relevant pairings. A mapping from labels to numbers is used...
    self.labels = dataset.getLabels()
    self.labelToNum = dict()
    for i,label in enumerate(self.labels):
      self.labelToNum[label] = i
    
    self.models = dict()
    for lNeg in xrange(len(self.labels)):
      for lPos in xrange(lNeg+1,len(self.labels)):
        #print self.labels[lNeg], self.labels[lPos]
        self.models[(lNeg,lPos)] = None

    # Generate the list of models that need solving...
    solveList = []
    for lNeg,lPos in self.models.keys():
      if isinstance(params,Params):
        solveList.append((lNeg,lPos,params))
      else:
        for p in params:
          solveList.append((lNeg,lPos,p))

    # Loop through all models and solve them, reporting progress if required...
    if pool==None:
      # Single process implimentation...
      for i,data in enumerate(solveList):
        lNeg,lPos,params = data
        if callback: callback(i,len(solveList))

        model = looPairRange(params, dataset.getTrainData(self.labels[lNeg], self.labels[lPos]), looDist)
        #print model[0], looPair(params, dataset.getTrainData(self.labels[lNeg], self.labels[lPos]))[0], looPairBrute(params, dataset.getTrainData(self.labels[lNeg], self.labels[lPos]))[0]
        if self.models[lNeg,lPos]==None or model[0]>self.models[lNeg,lPos][0]:
          self.models[lNeg,lPos] = model
    else:
      # Multiprocess implimentation...

      # Create a pool if it hasn't been provided...
      if type(pool)==type(True):
        pool = mp.Pool()

      # Callback for when each job completes...
      self.numComplete = 0
      if callback: callback(self.numComplete,len(solveList))
      
      def taskComplete(ret):
        self.numComplete += 1
        if callback: callback(self.numComplete,len(solveList))
        
        lNeg = ret[0]
        lPos = ret[1]
        model = (ret[2],ret[3])
        
        if self.models[lNeg,lPos]==None or model[0]>self.models[lNeg,lPos][0]:
          self.models[lNeg,lPos] = model
      
      try:
        # Create all the jobs, set them running...
        jobs = []
        for lNeg,lPos,params in solveList:
          jobs.append(pool.apply_async(mpLooPairRange,(params,dataset.getTrainData(self.labels[lNeg], self.labels[lPos]), lNeg, lPos, looDist), callback = taskComplete))
        
      finally:
        # Wait for them all to complete...
         while len(jobs)!=0:
          if jobs[0].ready():
            del jobs[0]
            continue
          time.sleep(0.1)


  def getLabels(self):
    """Returns a list of the labels supported."""
    return self.labels

  def getModel(self,labA,labB):
    """Returns a tuple of (model,neg label,pos label, loo) where model is the model between the pair and the two labels indicate which label is associated with the negative result and which with the positive result. loo is the leave one out score of this particular boundary."""
    la = self.labelToNum[labA]
    lb = self.labelToNum[labB]
    if la<lb:
      return (self.models[(la,lb)][1],labA,labB,self.models[(la,lb)][0])
    else:
      return (self.models[(lb,la)][1],labB,labA,self.models[(lb,la)][0])

  def paramsList(self):
    """Returns a list of parameters objects used by the model - good for curiosity."""
    return map(lambda x: x[1].getParams(),self.models.values())


  def classify(self,feature):
    """Classifies a single feature vector - returns the most likelly label."""
    if self.weightSVM:
      cost = numpy.zeros(len(self.labels),dtype=numpy.float_)
      
      for lNeg,lPos in self.models.keys():
        m = self.models[lNeg,lPos]
        cg = -math.log(max((m[0],1e-3)))
        cb = -math.log(max((1.0-m[0],1e-3))) # max required incase its perfect.
        
        val = m[1].classify(feature)
        if val<0:
          cost[lNeg] += cg
          cost[lPos] += cb
        else:
          cost[lNeg] += cb
          cost[lPos] += cg

      return self.labels[cost.argmin()]
    else:
      score = numpy.zeros(len(self.labels),dtype=numpy.int_)
    
      for lNeg,lPos in self.models.keys():
        val = self.models[lNeg,lPos][1].classify(feature)
        if val<0: score[lNeg] += 1
        else: score[lPos] += 1
      
      return self.labels[score.argmax()]

  def multiClassify(self,features):
    """Given a matrix where every row is a feature - returns a list of labels for the rows."""
    if self.weightSVM:
      cost = numpy.zeros((features.shape[0], len(self.labels)), dtype=numpy.float_)

      for lNeg,lPos in self.models.keys():
        m = self.models[lNeg,lPos]
        cg = -math.log(m[0])
        cb = -math.log(max((1.0-m[0],1e-3))) # max required incase its perfect.
        
        vals = m[1].multiClassify(features)
        cost[numpy.nonzero(vals<0)[0],lNeg] += cg
        cost[numpy.nonzero(vals>0)[0],lNeg] += cb
        cost[numpy.nonzero(vals>0)[0],lPos] += cg
        cost[numpy.nonzero(vals<0)[0],lPos] += cb

      ret = []
      for i in xrange(features.shape[0]):
        ret.append(self.labels[cost[i,:].argmin()])
      return ret
    else:
      score = numpy.zeros((features.shape[0], len(self.labels)), dtype=numpy.int_)
    
      for lNeg,lPos in self.models.keys():
        vals = self.models[lNeg,lPos][1].multiClassify(features)
        score[numpy.nonzero(vals<0)[0],lNeg] += 1
        score[numpy.nonzero(vals>0)[0],lPos] += 1

      ret = []
      for i in xrange(features.shape[0]):
        ret.append(self.labels[score[i,:].argmax()])
      return ret
