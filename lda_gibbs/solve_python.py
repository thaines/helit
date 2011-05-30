# -*- coding: utf-8 -*-

# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import numpy
import numpy.random
import scipy

import solve_shared as shared



def iniGibbs(state):
  """Does the initialisation gibbs pass, where it incrimentally sets the starting topic assignments based on the documents so far fitted. Builds the count matrices/vectors in the state at the same time."""
  
  dist = numpy.empty(state.topicCount.shape[0], dtype = numpy.float_)
  boostQuant = state.alpha*(state.alphaMult-1.0)
  
  for w in xrange(state.state.shape[0]): # Loop the words that consititute the state
    # Calculate the unnormalised distribution...
    dist[:] = numpy.asfarray(state.docTopicCount[state.state[w,0],:]) + state.alpha

    if state.boost[state.state[w,0]]!=-1:
      boostAmount = boostQuant
      dist[state.boost[state.state[w,0]]] += boostQuant
    else:
      boostAmount = 0.0
    
    dist /= numpy.asfarray(state.docCount[state.state[w,0]]) + state.topicCount.shape[0]*state.alpha + boostAmount
    
    dist *= numpy.asfarray(state.topicWordCount[:,state.state[w,1]]) + state.beta
    dist /= numpy.asfarray(state.topicCount) + state.topicWordCount.shape[1]*state.beta

    
    # Normalise...
    dist /= dist.sum()
    
    # Select and set the state...
    state.state[w,2] = numpy.nonzero(numpy.random.multinomial(1,dist))[0][0]
    
    # Incriment the relevant counts from each of the 4 arrays...
    state.topicWordCount[state.state[w,2],state.state[w,1]] += 1
    state.topicCount[state.state[w,2]] += 1
    state.docTopicCount[state.state[w,0],state.state[w,2]] += 1
    state.docCount[state.state[w,0]] += 1



def gibbs(state,iters,next):
  """Does iters number of full gibbs iterations."""
  
  dist = numpy.empty(state.topicCount.shape[0], dtype = numpy.float_)
  
  for i in xrange(iters):
    for w in xrange(state.state.shape[0]): # Loop the words that consititute the state
      # Decriment the relevant counts from each of the 4 arrays...
      state.topicWordCount[state.state[w,2],state.state[w,1]] -= 1
      state.topicCount[state.state[w,2]] -= 1
      state.docTopicCount[state.state[w,0],state.state[w,2]] -= 1
      state.docCount[state.state[w,0]] -= 1
    
      # Calculate the unnormalised distribution...
      dist[:] = numpy.asfarray(state.docTopicCount[state.state[w,0],:]) + state.alpha

      if state.boost[state.state[w,0]]!=-1:
        boostAmount = boostQuant
        dist[state.boost[state.state[w,0]]] += boostQuant
      else:
        boostAmount = 0.0

      dist /= numpy.asfarray(state.docCount[state.state[w,0]]) + state.topicCount.shape[0]*state.alpha + boostAmount

      dist *= numpy.asfarray(state.topicWordCount[:,state.state[w,1]]) + state.beta
      dist /= numpy.asfarray(state.topicCount) + state.topicWordCount.shape[1]*state.beta
    
      # Normalise...
      dist /= dist.sum()
    
      # Select and set the state...
      state.state[w,2] = numpy.nonzero(numpy.random.multinomial(1,dist))[0][0]
    
      # Incriment the relevant counts from each of the 4 arrays...
      state.topicWordCount[state.state[w,2],state.state[w,1]] += 1
      state.topicCount[state.state[w,2]] += 1
      state.docTopicCount[state.state[w,0],state.state[w,2]] += 1
      state.docCount[state.state[w,0]] += 1
      
    # Update the iter count...
    next()



def fitModel(state,params,next):
  """Given a state object generates samples."""
  iniGibbs(state)
  next()
  if params.burnIn>params.lag:
    gibbs(state,params.burnIn-params.lag,next)
  for i in xrange(params.samples):
    gibbs(state,params.lag,next)
    state.sample()
    next()



def fit(corpus,params,callback = None):
  """Complete fitting function - given a corpus fits a model. params is a Params object from solve-shared. callback if provided should take two numbers - the first is the number of iterations done, the second the number of iterations that need to be done; used to report progress. Note that it will probably not be called for every iteration, as that would be frightfully slow."""
  
  # Class to allow progress to be reported...
  class Reporter:
    def __init__(self,params,callback):
      self.doneIters = 0
      self.totalIters = params.runs * (1 + params.burnIn + params.samples + (params.samples-1)*params.lag)
      self.callback = callback
      
      if self.callback:
        self.callback(self.doneIters,self.totalIters)

    def next(self):
      self.doneIters += 1
      if self.callback:
        self.callback(self.doneIters,self.totalIters)
  report = Reporter(params,callback)
  
  s = shared.State(corpus)
  
  # Iterate and do each of the runs...
  for r in xrange(params.runs):
    ss = s.clone()
    fitModel(ss,params,report.next)
    s.absorbClone(ss)
  
  # Extract the final model into the corpus...
  s.extractModel(corpus)



def iniGibbsDoc(state,topicCount,topicsWords,alpha):
  dist = numpy.empty(topicCount.shape[0], dtype = numpy.float_)
  
  for w in xrange(state.shape[0]): # Loop the words that consititute the state
    # Calculate the unnormalised distribution...
    dist[:]  = topicsWords[:,state[w,0]]
    dist *= numpy.asfarray(topicCount) + alpha
    dist /= w + topicCount.shape[0]*alpha
    
    # Normalise...
    dist /= dist.sum()
    
    # Select and set the state...
    state[w,1] = numpy.nonzero(numpy.random.multinomial(1,dist))[0][0]
    
    # Incriment the relevant count for the words-per-topic array...
    topicCount[state[w,1]] += 1



def gibbsDoc(state,topicCount,topicsWords,alpha,iters):
  dist = numpy.empty(topicCount.shape[0], dtype = numpy.float_)
  
  for i in xrange(iters):
    for w in xrange(state.shape[0]): # Loop the words that consititute the state
      # Decriment the relevant count for the words-per-topic array...
      topicCount[state[w,1]] -= 1
    
      # Calculate the unnormalised distribution...
      dist[:]  = topicsWords[:,state[w,0]]
      dist *= numpy.asfarray(topicCount) + alpha
      dist /= state.shape[0] - 1.0 + topicCount.shape[0]*alpha
    
      # Normalise...
      dist /= dist.sum()
    
      # Select and set the state...
      state[w,1] = numpy.nonzero(numpy.random.multinomial(1,dist))[0][0]
    
      # Incriment the relevant count for the words-per-topic array...
      topicCount[state[w,1]] += 1



def fitDocModel(state,topicsWords,alpha,params):
  # Storage required - number of words assigned to each topic in the document...
  topicCount = numpy.zeros(topicsWords.shape[0], dtype=numpy.uint)
  
  # Do the initialisation step...
  iniGibbsDoc(state,topicCount,topicsWords,alpha)
  
  # If required do burn in...
  if params.burnIn>params.lag:
    gibbsDoc(state,topicCount,topicsWords,alpha,params.burnIn-params.lag)

  # Collect the samples...
  ret = numpy.zeros(topicsWords.shape[0],dtype=numpy.float_)
  prep = numpy.zeros(topicsWords.shape[0],dtype=numpy.float_)
  
  for i in xrange(params.samples):
    # Iterations...
    gibbsDoc(state,topicCount,topicsWords,alpha,params.lag)
    
    # Sample...
    prep[:] = numpy.asfarray(topicCount)
    prep += alpha
    prep /= float(state.shape[0]) + topicCount.shape[0]*alpha
    
    prep /= prep.sum()
    ret += prep
  
  # Return the model for combining...
  return ret # Normalisation left for fitDoc method, as each run will return same number of samples.



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
  
  # Iterate and do each of the runs...
  for i in xrange(params.runs):
    doc.model += fitDocModel(state.copy(),tw,alpha,params)
  
  # Renormalise the sum of models...
  doc.model /= doc.model.sum()
