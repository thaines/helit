# -*- coding: utf-8 -*-

# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import time

import multiprocessing as mp
import multiprocessing.synchronize # To make sure we have all the functionality.

import numpy
import numpy.random
import scipy

import solve_shared as shared
from solve_python import fitModel
from solve_python import fitDocModel



def fitModelWrapper(state,params,doneIters):
  """Wrapper around fitModel to make it suitable for multiprocessing."""
  def next():
    doneIters.value += 1
  fitModel(state,params,next)
  return state



def fit(corpus,params,callback = None):
  """Complete fitting function - given a corpus fits a model. params is a Params object from solve-shared. callback if provided should take two numbers - the first is the number of iterations done, the second the number of iterations that need to be done; used to report progress. Note that it will probably not be called for every iteration, as that would be frightfully slow."""
  
  # Create the state from the corpus and a pool of worker proccesses...
  s = shared.State(corpus)
  
  pool = mp.Pool()
  
  # Create a value for sub-processes to report back their progress with...
  manager = mp.Manager()
  doneIters = manager.Value('i',0)
  totalIters = params.runs * (1 + params.burnIn + params.samples + (params.samples-1)*params.lag)
  
  # Create a callback for when a job completes...
  def onComplete(state):
    s.absorbClone(state)
  
  # Create all the jobs...
  try:
    jobs = []
    for r in xrange(params.runs):
      jobs.append(pool.apply_async(fitModelWrapper,(s.clone(),params,doneIters),callback = onComplete))
  finally:
    # Close the pool and wait for all the jobs to complete...
    pool.close()
    while len(jobs)!=0:
      if jobs[0].ready():
        del jobs[0]
        continue
      time.sleep(0.1)
      if callback!=None:
        callback(doneIters.value,totalIters)
    pool.join()


  # Extract the final model into the corpus...
  s.extractModel(corpus)



def fitDoc(doc,topicsWords,alpha,params):
  """Given a single document finds the documents model parameter in the same way as the rest of the system, i.e. Gibbs sampling. Provided with a document to calculate for, a topics-words array giving the already trained topic-to-word distribution, the alpha parameter and a Params object indicating how much sampling to do."""

  # Normalise input to get P(word|topic)...
  tw = (topicsWords.T/topicsWords.sum(axis=1)).T
  
  # First generate a two column array - first column word index, second column its currently assigned topic...
  state = numpy.empty((doc.dupWords(),2),dtype=numpy.uint)
  
  index = 0
  for uIndex in xrange(doc.uniqueWords()):
    wIdent,count = doc.getWord(uIndex)
    for c in xrange(count):
      state[index,0] = wIdent
      state[index,1] = 0
      index += 1

  # Zero out the model...
  doc.model = numpy.zeros(tw.shape[0],dtype=numpy.float_)
  
  # Create a pool of processes to run the fitting in...
  pool = mp.Pool()
  
  # Callback for when a process completes...
  def onComplete(model):
    doc.model += model

  # Create all the jobs...
  try:
    jobs = []
    for r in xrange(params.runs):
      jobs.append(pool.apply_async(fitDocModel,(state.copy(),tw,alpha,params),callback = onComplete))
  finally:
    # Close the pool and wait for all the jobs to complete...
    pool.close()
    pool.join()
  
  # Renormalise the sum of models...
  doc.model /= doc.model.sum()
