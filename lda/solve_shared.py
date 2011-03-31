# -*- coding: utf-8 -*-

# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import heapq

import numpy



class State:
  """The state required by the solvers - stored as a few large numpy arrays for conveniance and speed. It is generated from a Corpus, but once generated is stand alone. It includes the ability to split out the final answer into the Corpus, as this is not seen as time critical (Must be the exact same and unedited Corpus.). To support parallel sampling you can duplicate this object, run the solvers on each duplicate, then merge them together, before extracting the final answer back into the Corpus."""
  def __init__(self, corpus):
    """Given a Corpus calculates the state object by duplicating all of the needed information."""
    
    # Break if cloning - let the work be done elsewhere...
    if corpus==None:
      return

    # Counts of the assorted things avaliable...
    dCount = corpus.documentCount()
    tCount = corpus.topicCount()
    wCount = corpus.wordCount()

    # Create tempory variables...
    self.topicWordTemp = numpy.empty((tCount, wCount), dtype=numpy.float_)
    self.docTopicTemp = numpy.empty((dCount,tCount), dtype=numpy.float_)

    # Copy over simple stuff...
    self.alpha = corpus.getAlpha()
    self.alphaMult = corpus.getAlphaMult()
    self.beta = corpus.getBeta()
    
    # Create the counters that are kept in synch with the current state, initialised to zero for the initialisation routine...
    
    # Number of times each word is assigned to each topic over all documents...
    self.topicWordCount = numpy.zeros((tCount, wCount), dtype=numpy.uint)
    
    # Number of words assigned to each topic over all documents...
    self.topicCount = numpy.zeros(tCount, dtype=numpy.uint)
    
    # Number of times in each document a word is assigned to a topic...
    self.docTopicCount = numpy.zeros((dCount,tCount), dtype=numpy.uint)
    
    # Number of words in each document (Constant)...
    self.docCount = numpy.zeros(dCount, dtype=numpy.uint)
    
    
    # Most important part is the state - this is a words x 3 matrix of uint32's, where the first column is the document ident, the second column the word ident and the third column the topic ident - first create the array, then fill it...
    self.state = numpy.empty((corpus.totalWordCount(),3), dtype=numpy.uint)
    
    index = 0
    for dIdent in xrange(dCount):
      doc = corpus.getDocument(dIdent)
      for uIndex in xrange(doc.uniqueWords()):
        wIdent, count = doc.getWord(uIndex)
        for c in xrange(count):
          self.state[index,0] = dIdent
          self.state[index,1] = wIdent
          self.state[index,2] = 1000000000 # Dummy bad value.
          index += 1
    assert(index==self.state.shape[0])


    # The boost array - for each document this provides the index of which topic alpha should be increased for, or -1 if none.
    self.boost = numpy.empty(dCount,dtype=numpy.int_)
    for dIdent in xrange(dCount):
      val = corpus.getDocument(dIdent).getTopic()
      if val==None: self.boost[dIdent] = -1
      else: self.boost[dIdent] = val


    # The output, before its been dropped into the corpus's data format - simply the output model obtained from the samples so far. It is the mean of the output calculated from each sample, calculated incrimentally. As a pair of matrices, rather than the shattered vectors stored in the Corpus...
    self.sampleCount = 0
    self.topicModel = numpy.zeros((tCount,wCount), dtype=numpy.float_)
    self.documentModel = numpy.zeros((dCount,tCount), dtype=numpy.float_)


  def clone(self):
    """Returns a clone of the object in question - typically called just after creation to make clones for each process of a multi-process implimentation."""
    ret = State(None)
    
    ret.topicWordTemp = self.topicWordTemp.copy()
    ret.docTopicTemp = self.docTopicTemp.copy()
    
    ret.alpha = self.alpha
    ret.alphaMult = self.alphaMult
    ret.beta = self.beta
    
    ret.topicWordCount = self.topicWordCount.copy()
    ret.topicCount = self.topicCount.copy()
    ret.docTopicCount = self.docTopicCount.copy()
    ret.docCount = self.docCount.copy()
    
    ret.state = self.state.copy()

    ret.boost = self.boost.copy()
    
    ret.sampleCount = self.sampleCount
    ret.topicModel = self.topicModel.copy()
    ret.documentModel = self.documentModel.copy()
    
    return ret
    
  def absorbClone(self, clone):
    """Given a previous clone this merges back in the sample information, effectivly combining the samples to get a better estimate ready for extraction. Note that the clone is no longer usable after this operation. This is evidently incorrect, as you can't combine topic vectors - there is not even a guarantee that they contain the same topics! However, in practise it works well enough, and any topic not well enough defined to appear in all samples is probably useless anyway and best smoothed out by being combined with different topics."""
    assert(self.sampleCount+clone.sampleCount>0)
    
    # Get reordering of topics in clone to best match the destination, if needed...
    if self.sampleCount==0:
      self.sampleCount = clone.sampleCount
      self.topicModel[:] = clone.topicModel
      self.documentModel[:] = clone.documentModel
    else:
      indices = self.matchTopics(self.topicModel,clone.topicModel)
    
      # Calculate weights...
      sWeight = float(self.sampleCount)/float(self.sampleCount+clone.sampleCount)
      cWeight = float(clone.sampleCount)/float(self.sampleCount+clone.sampleCount)
    
      # Combine them...
      self.sampleCount += clone.sampleCount
      self.topicModel = sWeight*self.topicModel + cWeight*clone.topicModel[indices,:]
      self.documentModel = sWeight*self.documentModel + cWeight*clone.documentModel[:,indices]


  def extractModel(self, corpus):
    """Extracts the calculated model into the given corpus, to be called once the fitting is done."""
    assert(self.sampleCount!=0)
    
    # (Note that the below renormalises the outputs, as they could of suffered numerical error in the incrimental averaging.)
    
    # First the topics...
    self.topicModel /= self.topicModel.sum()
    for t in xrange(self.topicModel.shape[0]):
      model = self.topicModel[t,:]
      corpus.getTopic(t).setModel(model)
    
    # Now the documents...
    for d in xrange(self.documentModel.shape[0]):
      model = self.documentModel[d,:]
      model /= model.sum()
      corpus.getDocument(d).setModel(model)


  def sample(self):
    """Samples the current state into the internal storage - needed by all solvers but not done regularly enough or is complex enough to require specialist optimisation."""
    
    # Calculate the topic model, with normalisation to get P(topic,word)...
    self.topicWordTemp[:] = numpy.asfarray(self.topicWordCount)
    self.topicWordTemp += self.beta
    self.topicWordTemp /= self.topicWordTemp.sum()
    #self.topicWordTemp = (self.topicWordTemp.T / (numpy.asfarray(self.topicCount) + self.topicWordCount.shape[1]*self.beta)).T # Normalisation to get P(words|topic), kept from the original implimentation.
    
    # Calculate the document model, with normalisation to get P(topic|doc)...
    self.docTopicTemp[:] = numpy.asfarray(self.docTopicCount)
    self.docTopicTemp += self.alpha
    boostInd = (self.boost+1).nonzero()
    self.docTopicTemp[boostInd,self.boost[boostInd]] += self.alpha*(self.alphaMult-1.0)
    
    self.docTopicTemp = (self.docTopicTemp.T / (numpy.asfarray(self.docCount) + self.docTopicCount.shape[1]*self.alpha + numpy.where(self.boost+1,self.alpha*(self.alphaMult-1.0),0.0))).T

    # Store...
    self.sampleCount += 1
    self.topicModel += (self.topicWordTemp-self.topicModel) / float(self.sampleCount)
    self.documentModel += (self.docTopicTemp-self.documentModel) / float(self.sampleCount)
  
  
  def matchTopics(self,topicWordA,topicWordB):
    """Returns the indices into topicWordB that best equate with topicWordA, used as topics do not necesarilly appear in the same order for each sample. Uses symmetric KL-divergance with greedy selection."""
    
    # Below could be made a lot faster, but not worth it as topic counts are relativly small in the grand scheme of things.

    # Normalise topicWord arrays by row, so we have P(word|topic), as needed for this matching method...
    twA = (topicWordA.T/topicWordA.sum(axis=1)).T
    twB = (topicWordB.T/topicWordB.sum(axis=1)).T
    
    # Create a list of tuples - (cost, a index, b index), to get greedy on - this is basically calculating all the symmetric KL divergances...
    heap = []
    for aInd in xrange(twA.shape[0]):
      for bInd in xrange(twA.shape[0]):
        cost = 0.5 * (twA[aInd,:]*numpy.log(twA[aInd,:]/twB[bInd,:]) + twB[bInd,:]*numpy.log(twB[bInd,:]/twA[aInd,:])).sum()
        heap.append((cost,aInd,bInd))
    
    # Turn the list into a heap...
    heapq.heapify(heap)
    
    # Keep pulling items from the heap to construct the translation table...
    aUsed = numpy.zeros(twA.shape[0],dtype=numpy.int_)
    bUsed = numpy.zeros(twA.shape[0],dtype=numpy.int_)
    remain = twA.shape[0]
    ret = numpy.zeros(twA.shape[0],dtype=numpy.int_)
    
    while remain!=0:
      match = heapq.heappop(heap)
      if (aUsed[match[1]]==0) and (bUsed[match[2]]==0):
        aUsed[match[1]] = 1
        bUsed[match[2]] = 1
        remain -= 1
        ret[match[1]] = match[2]

    return ret



class Params:
  """Parameters for running the fitter that are universal to all fitters."""
  def __init__(self):
    self.runs = 8
    self.samples = 10
    self.burnIn = 1000
    self.lag = 100

  
  def setRuns(self,runs):
    """Sets the number fom runs, i.e. how many seperate chains are run."""
    self.runs = runs
  
  def setSamples(self,samples):
    """Number of samples to extract from each chain - total number of samples going into the final estimate will then be sampels*runs."""
    self.samples = samples
  
  def setBurnIn(self,burnIn):
    """Number of Gibbs iterations to do for burn in before sampling starts."""
    self.burnIn = burnIn
  
  def setLag(self,lag):
    """Number of Gibbs iterations to do between samples."""
    self.lag = lag


  def getRuns(self):
    """Returns the number of runs."""
    return self.runs

  def getSamples(self):
    """Returns the number of samples."""
    return self.samples

  def getBurnIn(self):
    """Returns the burn in length."""
    return self.burnIn

  def getLag(self):
    """Returns the lag length."""
    return self.lag


  def fromArgs(self,args,prefix = ''):
    """Extracts from an arg string, typically sys.argv[1:], the parameters, leaving them untouched if not given. Uses --runs, --samples, --burnIn and --lag. Can optionally provide a prefix which is inserted after the '--'"""
    try:
      ind = args[:-1].index('--'+prefix+'runs')
      self.runs = int(args[ind+1])
    except:
      pass

    try:
      ind = args[:-1].index('--'+prefix+'samples')
      self.samples = int(args[ind+1])
    except:
      pass

    try:
      ind = args[:-1].index('--'+prefix+'burnIn')
      self.burnIn = int(args[ind+1])
    except:
      pass

    try:
      ind = args[:-1].index('--'+prefix+'lag')
      self.lag = int(args[ind+1])
    except:
      pass
