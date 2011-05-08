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



import time
import multiprocessing as mp
import multiprocessing.synchronize # To make sure we have all the functionality.

import numpy

import solve_shared as shared
from solve_weave import fitModel, fitModelDoc



def fitModelWrapper(state, params, doneIters, seed):
  """Wrapper around fitModel to make it suitable for multiprocessing."""
  numpy.random.seed(seed)
  def next(amount = 1):
    doneIters.value += amount
  fitModel(state,params,next)
  return state



def fit(corpus, params, callback = None):
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
    seeds = numpy.random.random_integers(0,10000000,params.runs)
    for r in xrange(params.runs):
      jobs.append(pool.apply_async(fitModelWrapper,(shared.State(s),params,doneIters,seeds[r]),callback = onComplete))
  finally:
    # Close the pool and wait for all the jobs to complete...
    pool.close()
    while len(jobs)!=0:
      if jobs[0].ready():
        del jobs[0]
        continue
      time.sleep(0.01)
      if callback!=None:
        callback(doneIters.value,totalIters)
    pool.join()

  # Extract the final model into the corpus...
  s.extractModel(corpus)



def fitModelDocWrapper(state,irR,wrtT,wrt,alpha,params,norm,seed):
  """Wrapper around fitModel to make it suitable for multiprocessing."""
  numpy.random.seed(seed)
  tCount = numpy.zeros(wrt.shape[2],dtype=numpy.uint)
  return fitModelDoc(state,tCount,irR,wrtT,wrt,alpha,params,norm)



def fitDoc(doc,ir,wrt,alpha,params,norm):
  """Given a document, the two parts of a model (ir and wrt) plus an alpha value and params object this Gibbs samples under the assumption that ir and wrt are correct, to determine the documents model, i.e. the multinomial from which topics are drawn for the given document."""
  irR = (ir.T/ir.sum(axis=1)).T
  wrtT = wrt.copy()
  for t in xrange(wrt.shape[2]): wrtT[:,:,t] /= wrtT[:,:,t].sum()

  # Construct the state matrix - each row is (topic,identifier,word)...
  state = numpy.empty((doc.getSampleCount(),3),dtype=numpy.uint)
  index = 0
  words = doc.getWords()
  for w in xrange(words.shape[0]):
    for c in xrange(words[w,2]):
      state[index,0] = 10000000 # Deliberate bad value, so bad initialisation would cause a crash.
      state[index,1] = words[w,0]
      state[index,2] = words[w,1]
      index += 1
  assert(index==state.shape[0])

  # Create the process pool, create the callback for when jobs complete...
  pool = mp.Pool()

  samples = []
  def onComplete(samp):
    samples.extend(samp)

  # Do all the runs, collate the samples...
  # Create all the jobs...
  try:
    jobs = []
    seeds = numpy.random.random_integers(0,10000000,params.runs)
    for r in xrange(params.runs):
      jobs.append(pool.apply_async(fitModelDocWrapper,(state.copy(),irR,wrtT,wrt,alpha,params,norm,seeds[r]),callback = onComplete))
  finally:
    # Close the pool and wait for all the jobs to complete...
    pool.close()
    while len(jobs)!=0:
      if jobs[0].ready():
        del jobs[0]
        continue
      time.sleep(0.01)
    pool.join()

  # Merge the samples to get the final model, write it into the given doc object...
  model = numpy.zeros(wrt.shape[2],dtype=numpy.float_)
  for i,sample in enumerate(samples):
    model += ((sample[0].astype(numpy.float_) + alpha) - model) / float(i+1)
  doc.setModel(model)

  # Combine the region probability estimates, write the region negative log likelihoods to the document...
  doc.nllRegion = numpy.zeros(wrt.shape[1],dtype=numpy.float_)
  doc.sizeRegion = numpy.zeros(wrt.shape[1],dtype=numpy.float_)
  for i,sample in enumerate(samples):
    doc.nllRegion += (sample[1] - doc.nllRegion) / float(i+1)
    doc.sizeRegion += (sample[2].astype(numpy.float_) - doc.sizeRegion) / float(i+1)
