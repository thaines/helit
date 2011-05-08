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



import heapq
import numpy
import corpus



class State:
  """The state required by the solvers - stored as a few large numpy arrays for conveniance and speed. It is generated from a Corpus, but once generated is stand alone. It includes the ability to split out the final answer into the Corpus, as this is not seen as time critical (Must be the exact same and unedited Corpus.). To support parallel sampling you can duplicate this object, run the solvers on each duplicate, then merge them together, before extracting the final answer back into the Corpus."""
  def __init__(self, obj):
    """Prepares the state object given a corpus, or alternativly another State, which it will then clone."""
    if isinstance(obj,State):
      self.alpha = obj.alpha
      self.beta = obj.beta
      self.gamma = obj.gamma
      
      self.state = obj.state.copy()
      self.sIndex = obj.sIndex.copy()
      self.ir = obj.ir.copy()
      self.wrt = obj.wrt.copy()
      self.mt = obj.mt.copy()
      self.mr = obj.mr.copy()
      self.dt = obj.dt.copy()
    else:
      # Assume obj is a Corpus, or has the Corpus interface, and work from there.

      # Counts of the assorted things avaliable...
      sCount = obj.getSampleCount()
      dCount = len(obj.documentList())
      tCount = obj.getTopicCount()
      rCount = obj.getRegionCount()
      iCount = obj.getMaxIdentNum() + 1
      wCount = obj.getMaxWordNum() + 1
      
      # Extract the parameters needed from the corpus...
      self.alpha = obj.getAlpha()
      self.beta = obj.getBeta()
      self.gamma = obj.getGamma()
      
      # Create the samples array, this being an array with 4 columns - [document, topic, identifier, word]. It is sorted by identifier, and an index exists to grab the ranges for each identifier. During runtime only the topic column changes, during the t phase...
      # Make array, plus indexing array...
      self.state = numpy.empty((sCount,4), dtype=numpy.uint)
      self.sIndex = numpy.zeros(iCount+1, dtype=numpy.uint)

      # Fill array...
      index = 0
      for doc in obj.documentList():
        dNum = doc.getNum()
        words = doc.getWords()
        for j in xrange(words.shape[0]):
          for _ in xrange(words[j,2]): # Iterate the count of the number of words.
            self.state[index,0] = dNum
            self.state[index,1] = 1000000 # Dummy value.
            self.state[index,2] = words[j,0]
            self.state[index,3] = words[j,1]
            self.sIndex[words[j,0]+1] += 1
            index += 1
      assert(index==sCount)

      # Sort array by identifier...
      self.state = self.state[self.state[:,2].argsort(),:]

      # Create index from the data currently in the index...
      for i in xrange(iCount):
        self.sIndex[i+1] += self.sIndex[i]
      assert(self.sIndex[-1]==sCount)
      

      # Create the remaining state - the mapping from identifier to region - a simple vector indexed by identifier. Randomise it...
      self.ir = numpy.random.randint(0, rCount, iCount)

      # Create the region/topic/word count, of how many times each combo exists, [word,region,topic]. Initialise it zeroed out ready to be grown...
      self.wrt = numpy.zeros((wCount,rCount,tCount), dtype=numpy.uint)

      # Marginalised version of the wrt matrix - the first is marginalised over w and r, the second over w and t...
      self.mt = numpy.zeros(tCount, dtype=numpy.uint)
      self.mr = numpy.zeros(rCount, dtype=numpy.uint)

      # Create the topic/document count, of how many instances of each topic exist in each document, [document,topic]. Zeroed out again...
      self.dt = numpy.zeros((dCount,tCount), dtype=numpy.uint)

    # Create the data structure to store sampled states in - just a list of tuples, (ir,wrt,dt)...
    self.samples = []


  def absorbClone(self,clone):
    """Absorbs a clone of this object by copying its sampled states into this objects sampled states list."""

    if len(self.samples)==0:
      self.samples += clone.samples
    else:
      # Need to merge the two samples lists - unfortunatly both regions and topics can end up at different indices, so we have to match them up (Which is of course assuming they can be matched up, which is not technically true. Works reasonably well however.)...

      # First, regions...
      # For each region calculate a multinomial from region to identifier...
      # (Also adds one to every bin, to simulate a Dirichlet prior with a single parameter of 1, i.e. a flat prior.)
      rCount = self.wrt.shape[1]
      # self...
      sMult = numpy.ones((rCount,self.ir.shape[0]), dtype=numpy.float_)
      for sample in self.samples:
        sMult[sample[0],numpy.arange(len(sample[0]))] += 1.0
      sMult = (sMult.T / sMult.sum(axis=1)).T

      # clone...
      cMult = numpy.ones((rCount,self.ir.shape[0]), dtype=numpy.float_)
      for sample in clone.samples:
        cMult[sample[0],numpy.arange(len(sample[0]))] += 1.0
      cMult = (cMult.T / cMult.sum(axis=1)).T

      # Calculate Kullback-Leiber divergance between them all...
      regionHeap = [] # List of tuples (symmetric kl, self r, clone r)
      for sr in xrange(rCount):
        for cr in xrange(rCount):
          cost = 0.5 * (sMult[sr,:]*numpy.log(sMult[sr,:]/cMult[cr,:]) + cMult[cr,:]*numpy.log(cMult[cr,:]/sMult[sr,:])).sum()
          regionHeap.append((cost,sr,cr))

      # Use a heap to match them up, using a greedy algorithm...
      heapq.heapify(regionHeap)

      sUsed = numpy.zeros(rCount,dtype=numpy.int_)
      cUsed = numpy.zeros(rCount,dtype=numpy.int_)
      remain = rCount
      regionC2S = numpy.zeros(rCount,dtype=numpy.int_)
      regionS2C = numpy.zeros(rCount,dtype=numpy.int_)

      while remain!=0:
        match = heapq.heappop(regionHeap)
        if (sUsed[match[1]]==0) and (cUsed[match[2]]==0):
          sUsed[match[1]] = 1
          cUsed[match[2]] = 1
          remain -= 1
          regionS2C[match[1]] = match[2]
          regionC2S[match[2]] = match[1]


      # Second, topics - same idea as above basically...
      tCount = self.wrt.shape[2]
      # self...
      sMult = numpy.zeros(self.wrt.shape,dtype=numpy.float_)
      for i, sample in enumerate(self.samples):
        sMult += ((sample[1].astype(numpy.float_) + self.beta) - sMult) / float(i+1)

      for t in xrange(tCount):
        sMult[:,:,t] /= sMult[:,:,t].sum()

      # clone...
      cMult = numpy.zeros(self.wrt.shape,dtype=numpy.float_)
      for i, sample in enumerate(clone.samples):
        cMult += ((sample[1][:,regionS2C,:].astype(numpy.float_) + self.beta) - cMult) / float(i+1)
        
      for t in xrange(tCount):
        cMult[:,:,t] /= cMult[:,:,t].sum()

      # Calculate Kullback-Leiber divergance between them all...
      topicHeap = [] # List of tuples (symmetric kl, self t, clone t)
      for st in xrange(tCount):
        for ct in xrange(tCount):
          cost = 0.5 * (sMult[:,:,st]*numpy.log(sMult[:,:,st]/cMult[:,:,ct]) + cMult[:,:,ct]*numpy.log(cMult[:,:,ct]/sMult[:,:,st])).sum()
          topicHeap.append((cost,st,ct))

      # Use a heap to match them up, using a greedy algorithm...
      heapq.heapify(topicHeap)

      sUsed = numpy.zeros(tCount,dtype=numpy.int_)
      cUsed = numpy.zeros(tCount,dtype=numpy.int_)
      remain = tCount
      topicS2C = numpy.zeros(tCount,dtype=numpy.int_)

      while remain!=0:
        match = heapq.heappop(topicHeap)
        if (sUsed[match[1]]==0) and (cUsed[match[2]]==0):
          sUsed[match[1]] = 1
          cUsed[match[2]] = 1
          remain -= 1
          topicS2C[match[1]] = match[2]


      # Thirdly, merge in the data, making the relevant transform...
      for sample in clone.samples:
        nIR = regionC2S[sample[0]]
        nWRT = sample[1][:,regionS2C,:][:,:,topicS2C]
        nDT = sample[2][:,topicS2C]
        self.samples.append((nIR,nWRT,nDT))


  def extractModel(self,corpus):
    """Extracts the calculated model into the given corpus, to be called once the sampling is done."""
    assert(len(self.samples)>0)
    
    ir = numpy.zeros((self.ir.shape[0],self.wrt.shape[1]),dtype=numpy.float_)
    wrt = numpy.zeros(self.wrt.shape,dtype=numpy.float_)
    dt = numpy.zeros(self.dt.shape,dtype=numpy.float_)

    for i in xrange(len(self.samples)):
      sIR, sWRT, sDT = self.samples[i]
      ir[numpy.arange(len(sIR)),sIR] += 1.0 # I ignore gamma - I'ld rather keep the distributions hard and I abused it for something else already.
      wrt += ((sWRT.astype(numpy.float_) + self.beta) - wrt) / float(i+1)
      dt += ((sDT.astype(numpy.float_) + self.alpha) - dt) / float(i+1)
    
    corpus.setModel(wrt,ir)
    for doc in corpus.documentList():
      doc.setModel(dt[doc.getNum(),:])


  def sample(self):
    """Samples the current state, storing it so it can be later used to determine the model."""
    ss = (self.ir.copy(),self.wrt.copy(),self.dt.copy())
    self.samples.append(ss)
