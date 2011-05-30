# -*- coding: utf-8 -*-

# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import numpy
import scipy.weave as weave

import solve_shared as shared



def iniGibbs(s):
  """Does the initialisation gibbs pass, where it incrimentally sets the starting topic assignments based on the documents so far fitted. Builds the count matrices/vectors in the state at the same time."""
  
  dist = numpy.empty(s.topicCount.shape[0], dtype = numpy.float_)
  
  topicWordCount = s.topicWordCount
  topicCount = s.topicCount
  docTopicCount = s.docTopicCount
  docCount = s.docCount
  state = s.state
  alpha = s.alpha
  beta = s.beta
  boostAmount = s.alpha*(s.alphaMult-1.0)
  boost = s.boost
  
  rand = numpy.random.random(state.shape[0])
  
  code = """
  for (int w=0;w<Nstate[0];w++)
  {
   // Calculate the unnormalised distribution...
    float sum = 0.0;
    int bt = BOOST1(STATE2(w,0));
    for (int t=0;t<Ndist[0];t++)
    {
     float top1 = TOPICWORDCOUNT2(t,STATE2(w,1)) + beta;
     float bottom1 = TOPICCOUNT1(t) + NtopicWordCount[1]*beta;
     float top2 = DOCTOPICCOUNT2(STATE2(w,0),t) + alpha;
     if (bt==t) top2 += boostAmount;
     
     DIST1(t) = (top1/bottom1) * top2;
     sum += DIST1(t);
    }
    
   // Normalise the distribution...
    for (int t=0;t<Ndist[0];t++) DIST1(t) /= sum;
   
   // Select and set the state...
    sum = 0.0;
    for (int t=0;t<Ndist[0];t++)
    {
     STATE2(w,2) = t;
     sum += DIST1(t);
     if (sum>RAND1(w)) break;
    }
  
   // Incriment the relevant counts from each of the 4 arrays...
    TOPICWORDCOUNT2(STATE2(w,2),STATE2(w,1)) += 1;
    TOPICCOUNT1(STATE2(w,2)) += 1;
    DOCTOPICCOUNT2(STATE2(w,0),STATE2(w,2)) += 1;
    DOCCOUNT1(STATE2(w,0)) += 1;
  }
  """
  weave.inline(code, ['dist', 'topicWordCount', 'topicCount', 'docTopicCount', 'docCount', 'state', 'alpha', 'beta', 'rand', 'boostAmount', 'boost'])



def gibbs(s,iters,next):
  """Does iters number of Gibbs iterations."""
  
  # Variables...
  dist = numpy.empty(s.topicCount.shape[0], dtype = numpy.float_)
  
  topicWordCount = s.topicWordCount
  topicCount = s.topicCount
  docTopicCount = s.docTopicCount
  docCount = s.docCount
  state = s.state
  alpha = s.alpha
  beta = s.beta
  boostAmount = s.alpha*(s.alphaMult-1.0)
  boost = s.boost
  
  # Code...
  code = """
  for (int i=0;i<numIters;i++)
  {
   for (int w=0;w<Nstate[0];w++)
   {
    // Decriment the relevant counts from each of the 4 arrays...
     TOPICWORDCOUNT2(STATE2(w,2),STATE2(w,1)) -= 1;
     TOPICCOUNT1(STATE2(w,2)) -= 1;
     DOCTOPICCOUNT2(STATE2(w,0),STATE2(w,2)) -= 1;
     DOCCOUNT1(STATE2(w,0)) -= 1;

    // Calculate the unnormalised distribution...
     float sum = 0.0;
     int bt = BOOST1(STATE2(w,0));
     for (int t=0;t<Ndist[0];t++)
     {
      float top1 = TOPICWORDCOUNT2(t,STATE2(w,1)) + beta;
      float bottom1 = TOPICCOUNT1(t) + NtopicWordCount[1]*beta;
      float top2 = DOCTOPICCOUNT2(STATE2(w,0),t) + alpha;
      if (bt==t) top2 += boostAmount;
      
      float val = (top1/bottom1) * top2;
      DIST1(t) = val;
      sum += val;
     }
   
    // Select and set the state...
     float offset = 0.0;
     float threshold = RAND2(i,w) * sum;
     STATE2(w,2) = Ndist[0]-1;
     for (int t=0;t<Ndist[0];t++)
     {
      offset += DIST1(t);
      if (offset>threshold)
      {
       STATE2(w,2) = t;
       break;
      }
     }
  
    // Incriment the relevant counts from each of the 4 arrays...
     TOPICWORDCOUNT2(STATE2(w,2),STATE2(w,1)) += 1;
     TOPICCOUNT1(STATE2(w,2)) += 1;
     DOCTOPICCOUNT2(STATE2(w,0),STATE2(w,2)) += 1;
     DOCCOUNT1(STATE2(w,0)) += 1;
   }
  }
  """
  
  # Execution, taking care to not let the random number array get too large...
  chunkSize = (8*1024*1024)/state.shape[0] + 1
  while iters!=0:
    numIters = min(chunkSize,iters)
    iters -= numIters
    rand = numpy.random.random((numIters,state.shape[0]))

    weave.inline(code, ['dist', 'topicWordCount', 'topicCount', 'docTopicCount', 'docCount', 'state', 'alpha', 'beta', 'rand', 'numIters', 'boostAmount', 'boost'])
  
    next(numIters)



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

    def next(self,amount = 1):
      self.doneIters += amount
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
  rand = numpy.random.random(state.shape[0])
  
  code = """
  for (int w=0;w<Nstate[0];w++)
  {
   // Calculate the unnormalised distribution...
    float sum = 0.0;
    for (int t=0;t<Ndist[0];t++)
    {
     DIST1(t)  = TOPICSWORDS2(t,STATE2(w,0));
     DIST1(t) *= TOPICCOUNT1(t) + alpha;
     DIST1(t) /= float(w) + NtopicCount[0]*alpha;
     sum += DIST1(t);
    }
    
   // Normalise...
    for (int t=0;t<Ndist[0];t++) DIST1(t) /= sum;
    
   // Select and set the state...
    sum = 0.0;
    for (int t=0;t<Ndist[0];t++)
    {
     STATE2(w,1) = t;
     sum += DIST1(t);
     if (sum>RAND1(w)) break;
    }
    
   // Incriment the relevant count for the words-per-topic array...
    TOPICCOUNT1(STATE2(w,1)) += 1;
  }
  """
  weave.inline(code,['state', 'topicCount', 'topicsWords', 'alpha', 'dist', 'rand'])



def gibbsDoc(state,topicCount,topicsWords,alpha,iters):
  dist = numpy.empty(topicCount.shape[0], dtype = numpy.float_)
  
  # Code...
  code = """
  for (int i=0;i<numIters;i++)
  {
   for (int w=0;w<Nstate[0];w++)
   {    
    // Decriment the relevant count for the words-per-topic array...
     TOPICCOUNT1(STATE2(w,1)) -= 1;

    // Calculate the unnormalised distribution...
     float sum = 0.0;
     for (int t=0;t<Ndist[0];t++)
     {
      DIST1(t)  = TOPICSWORDS2(t,STATE2(w,0));
      DIST1(t) *= TOPICCOUNT1(t) + alpha;
      DIST1(t) /= Nstate[0] - 1.0 + NtopicCount[0]*alpha;
      sum += DIST1(t);
     }
    
    // Normalise...
     for (int t=0;t<Ndist[0];t++) DIST1(t) /= sum;
    
    // Select and set the state...
     sum = 0.0;
     for (int t=0;t<Ndist[0];t++)
     {
      STATE2(w,1) = t;
      sum += DIST1(t);
      if (sum>RAND2(i,w)) break;
     }
    
    // Incriment the relevant count for the words-per-topic array...
     TOPICCOUNT1(STATE2(w,1)) += 1;
   }
  }
  """
  
  chunkSize = (8*1024*1024)/state.shape[0] + 1
  while iters!=0:
    numIters = min(iters,chunkSize)
    iters -= numIters
    
    rand = numpy.random.random((numIters,state.shape[0]))
  
    weave.inline(code,['state', 'topicCount', 'topicsWords', 'alpha', 'dist', 'rand', 'numIters'])



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
