# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import numpy
import numpy.random.mtrand
import scipy.special



class VLDA:
  """A variational implimentation of LDA - everything is in a single class. Has an extensive feature set, and is capable of more than simply fitting a model."""
  def __init__(self, topics, words):
    """Initialises the object - you provide the number of topics and the number of words."""
    self.topicCount = topics
    self.wordCount = words
    
    self.alpha = numpy.ones(self.topicCount, dtype=numpy.float32)
    self.gamma = numpy.ones(self.wordCount, dtype=numpy.float32)

    self.docRecycle = [] # List of unused indices in self.docs.
    self.docs = dict() # A dictionary indexed by document identifier, that goes to an integer array [entry,2] where, for each i, [i,0] is the word index and [i,1] the number of instances of that word in the document. Sorted by word index.
    
    self.docLock = set() # A set containing all the documents that are locked, and therefore should not be used for processing.
    self.betaLock = False # If True beta will not be updated.

    self.beta = numpy.ones((self.topicCount,self.wordCount), dtype=numpy.float32) # The priors over the beta distribution - each [t,:] is the parameters for the DP over that particular multinomial.
    self.betaLogExp = numpy.ones((self.topicCount,self.wordCount), dtype=numpy.float32) * (scipy.special.psi(1.0) - scipy.special.psi(self.wordCount)) # The expected value of the log of beta, cached, and updated at different times from the above whilst solving.
    self.theta = dict() # Indexed by document ident this provides the DP over the theta parameter for each document, as a numpy array.
    self.z = dict() # This provides the multinomials for each z value by word - the dictionary is indexed by document ident, and goes to a 2D numpy array, [entry,topics], where each row matches up with the self.doc row, and contains the parameters of the multinomial from which z for that particular word is drawn.

    self.epsilon = 1e-5 # Amount of change below which it stops iterating.
    self.delta = None # Best of above acheived, for if there is an iteration cap.

    # Temporary stuff, to reduce memory churn...
    self.wordMN = numpy.empty(self.wordCount, dtype=numpy.float32)
    self.betaPrior = numpy.empty((self.topicCount,self.wordCount), dtype=numpy.float32)

  def clone(self):
    """Returns a copy of this object."""
    ret = VLDA(self.topicCount, self.wordCount)

    ret.alpha = self.alpha.copy()
    ret.gamma = self.gamma.copy()

    ret.docRecycle = self.docRecycle[:]
    for ident, data in self.docs.iteritems():
      ret.docs[ident] = data.copy()

    ret.docLock = self.docLock.copy()
    ret.betaLock = self.betaLock

    ret.beta = self.beta.copy()
    ret.betaLogExp = self.betaLogExp.copy()
    for ident, data in self.theta.iteritems():
      ret.theta[ident] = data.copy()
    for ident, data in self.z.iteritems():
      ret.z[ident] = data.copy()

    ret.epsilon = self.epsilon

    return ret


  def numTopics(self):
    """Returns the number of topics - idents are therefore all natural numbers less than this."""
    return self.topicCount

  def numWords(self):
    """Returns the number of words - idents are therefore all natural numbers less than this."""
    return self.wordCount

    
  def setAlpha(self, alpha):
    """Sets the alpha prior - you should use either a complete numpy array, of length topics, or a single value, which will be replicated for all values of the DP."""
    self.alpha[:] = numpy.asarray(alpha)

  def getAlpha(self):
    """Returns the current alpha value, the parameters for the DP prior, as a numpy array. This is the prior over the per-document multinomial from which topics are drawn."""
    return self.alpha

  def setGamma(self, gamma):
    """Sets the gamma prior - you should use either a complete numpy array, of length words, or a single value, which will be replicated for all values of the DP."""
    self.gamma[:] = numpy.asarray(gamma)

  def getGamma(self):
    """Returns the current gamma value, the parameters for the DP prior, as a numpy array. This is the prior over the per-topic multinomial from which words are drawn."""
    return self.gamma

  def setThreshold(self, epsilon):
    """Sets the threshold for parameter change below which it considers it to have converged, and stops iterating."""
    self.epsilon = epsilon


  def add(self, dic):
    """Adds a document. Given a dictionary indexed by word identifier (An integer [0,wordCount-1]) that leads to a count of how many of the given words exist in the document. Omitted words are assumed to have no instances. Returns an identifier (An integer) that can be used to request information about the document at a later stage."""
    
    # Select an integer as an identifier, reusing old ones if avaliable...
    if len(self.docRecycle)!=0:
      ident = self.docRecycle[0]
      self.docRecycle = self.docRecycle[1:]
    else:
      ident = len(self.docs)

    # Convert the dictionary into an array...
    data = numpy.empty((len(dic),2), dtype=numpy.int32)
    i = 0
    for word, count in dic.iteritems():
      data[i,0] = word
      data[i,1] = count
      i += 1

    # Store and return...
    self.docs[ident] = data
    return ident

  def rem(self, doc):
    """Removes a document, as identified by its identifier returned by add. Note that this results in some memory efficiency, though it will use newly added documents to fill the gap."""
    if doc in self.docs:
      self.docRecycle.append(doc)
      del self.docs[doc]
      if doc in self.docLock: self.docLock.remove(doc)
      if doc in self.theta: del self.theta[doc]
      if doc in self.z: del self.z[doc]

  def docCount(self):
    """Returns the number of documents in the system."""
    return len(self.docs)


  def lockDoc(self, doc, lock=True):
    """Given a document identifier this locks it, or unlocks it if lock==False - can be used for repeated calls to solve to reduce computation if desired."""
    if lock:
      self.docLock.add(doc)
    else:
      if doc in self.docLock:
        self.docLoc.remove(doc)

  def lockAllDoc(self, lock=True):
    """Same as lockDoc, but for all documents."""
    if lock: self.lockDoc = set(self.docs.iterkeys())
    else: self.lockDoc = set()

  def lockBeta(self, lock=True):
    """Locks, or unlocks, the beta parameter from being updated during a solve - can be useful for repeated calls to solve, to effectivly lock the model and analyse new documents"""
    self.betaLock = lock
  
  def solve(self, maxIter = None):
    """Solves the model, such that you can query all the model details. Returns how many passes it took to acheive convergance. You can optionally set maxIter, to avoid it running forever. Due to its incrimental nature repeated calls with maxIter whilst keeping an eye on the delta can be used to present progress."""

    # Check for new documents and extend the model accordingly...
    for ident in self.docs.iterkeys():
      if ident not in self.docLock: # Don't bother if the document is locked - might as well leave it empty if an empty document has been locked.
        if ident not in self.theta:
          self.theta[ident] = numpy.ones(self.topicCount, dtype=numpy.float32)
        if ident not in self.z:
          self.z[ident] = numpy.asarray(numpy.random.mtrand.dirichlet(self.alpha, size=self.docs[ident].shape[0]), dtype=numpy.float32) # Initialised randomly as otherwise it can't converge - the maths is symmetric and so won't break symmetry.

    # Create some extra storage/cache stuff...
    # The beta prior - basically gamma repeated in a beta shaped array, with 1 subtracted from it in advance, and the contributions from all the omitted documents added in...
    if not self.betaLock:
      self.betaPrior[:,:] = self.gamma.reshape((1,self.wordCount))
      for doc in self.docLock:
        if doc in self.z:
          d = self.docs[doc]
          z = self.z[doc]
          self.betaPrior[:,d[:,0]] += (d[:,1].reshape((d.shape[0],1)) * z).T

    # Iterate until convergance...
    docList = filter(lambda x: x not in self.docLock, self.docs.iterkeys())
    maxWordCount = 0
    for doc in docList:
      maxWordCount = max((maxWordCount,self.docs[doc].shape[0]))
    prevCache = numpy.empty((maxWordCount,self.topicCount), dtype=numpy.float32)

    passes = 0
    while passes!=maxIter:
      passes += 1
      self.delta = 0.0
      
      # Reset beta to the prior, without touching the cached log expectation...
      if not self.betaLock:
        self.beta[:,:] = self.betaPrior
      
      # Iterate the documents - in a single pass we can do basically everything, except for the final refresh of the beta log expectation cache...
      for doc in docList:
        d = self.docs[doc]
        theta = self.theta[doc]
        z = self.z[doc]
        
        # First update theta...
        theta[:] = self.alpha
        theta += (z * d[:,1].reshape((d.shape[0],1))).sum(axis=0)

        # Calculate theta-s expectation of the log...
        thetaExpLog = scipy.special.psi(theta)
        thetaExpLog -= scipy.special.psi(theta.sum())

        # Update the z values...
        prevCache[:z.shape[0],:] = z
        z[:,:] = thetaExpLog.reshape((1,self.topicCount))
        z += self.betaLogExp[:,d[:,0]].T
        z[:,:] = numpy.exp(z)
        z /= z.sum(axis=1).reshape((z.shape[0],1))

        # Measure the amount of change...
        delta = numpy.abs(prevCache[:z.shape[0],:]-z).sum(axis=1).max()
        self.delta = max(self.delta, delta)

        # Contribute to the beta estimation from this document...
        if not self.betaLock:
          self.beta[:,d[:,0]] += (d[:,1].reshape((d.shape[0],1)) * z).T
      
      # Update the beta log expectation cache...
      if not self.betaLock:
        betaSum = self.beta.sum(axis=1)
        self.betaLogExp[:,:] = scipy.special.psi(self.beta)
        self.betaLogExp -= scipy.special.psi(betaSum).reshape((self.topicCount,1))

      # Break if converged...
      if self.delta<self.epsilon: break

    return passes

  def solveHuman(self, step=32):
    """Does the exact same thing as solve except it prints out status reports and allows you to hit 'd' to cause it to exit - essentially an interactive version that reports progress so a human can decide to break early if need be. Uses curses."""
    import time
    import curses

    screen = curses.initscr()
    curses.noecho()
    screen.nodelay(1)
    screen.clear()
    screen.addstr(1,2,'Press d for early termination')
    screen.refresh()
    totalIters = 0
    start = time.clock()

    timeSeq = []
    iterSeq = []
    deltaSeq = []
    count = 5

    try:
      while True:
        totalIters += self.solve(step)
        timeSeq.append(time.clock()-start)
        iterSeq.append(totalIters)
        deltaSeq.append(self.delta)

        timeSeq = timeSeq[-count:]
        iterSeq = iterSeq[-count:]
        deltaSeq = deltaSeq[-count:]
        
        screen.clear()
        screen.addstr(1,2,'Press d for early termination')

        offset = 3
        for ti, it, de in zip(timeSeq, iterSeq, deltaSeq):
          screen.addstr(offset,2,'total iters = %i; time = %.1fs'%(it, ti))
          screen.addstr(offset+1,2,'delta = %.6f; target = %.6f'%(de, self.epsilon))
          offset += 3
        
        screen.refresh()
        key = screen.getch()
        if key==ord('d'): break
        if self.delta<self.epsilon: break
        
    finally:
      curses.endwin()
    return totalIters

  def getDelta(self):
    """Returns the maximum change seen for any z multinomial in the most recent iteration. Useful if using maxIter to see if it has got close enough."""
    return self.delta


  def getBeta(self, topic):
    """Returns the parameters for the Dirichlet distribution over the beta multinomial for the given topic, i.e. the DP from which the multinomial that words for the topic are drawn from."""
    return self.beta[topic,:]

  def getTheta(self, doc):
    """Returns the parameter vector for the DP over the theta variable associated with the requested document, i.e. the DP from which the per-document multinomial over topics is drawn. Returns None if it has not been calculated."""
    if doc in self.theta: return self.theta[doc]
    else: return None

  def getDoc(self, doc):
    """Given a document identifier returns a reconstruction of the dictionary originally provided to the add method."""
    if doc not in self.docs: return None
    d = self.docs[doc]

    ret = dict()
    for i in xrange(d.shape[0]): ret[d[i,0]] = d[i,1]
    return ret
    
  def getZ(self, doc):
    """Returns for the provided document a dictionary that is indexed by word index and obtains multinomials over the value of Z for the words with that value in the document. See getDoc for how many times you would need to draw from each distribution. Will only include multinomials for words that exist in the document (There is a hack involving putting a word count of zero in the input to get other words however.). Returns None if it has not been calculated."""
    if doc not in self.z: return None
    d = self.docs[doc]
    z = self.z[doc]

    ret = dict()
    for i in xrange(d.shape[0]): ret[d[i,0]] = z[i,:]
    return ret

  def getNLL(self, doc):
    """Given a document identifier this returns the probability of the document, given the model that has been fitted. Specifically, it returns the negative log likelyhood of the words given the model that has been fitted to it, using the expected values for theta and beta. Returns None if the value can't be calculated, i.e. solve needs to be called."""
    if doc not in self.theta: return None

    d = self.docs[doc]
    theta = self.theta[doc]
    ret = 0.0

    # Calculate the probability of drawing each word given the expected values of beta and theta...
    self.wordMN[:] = (self.beta * theta.reshape((theta.shape[0],1))).sum(axis=0)
    self.wordMN /= self.wordMN.sum()

    # Calculate the nll of drawing the given number of each word...
    ret -= (numpy.log(self.wordMN[:,d[:,0]]) * d[:,1]).sum()

    # Normalisation constant from word ordering being arbitary...
    logInt = numpy.log(numpy.arange(d[:,1].max()+1))
    logInt[0] = 0.0
    for i in xrange(1,logInt.shape[0]): logInt[i] += logInt[i-1]

    ret -= scipy.special.gammaln(d[:,1].sum()+1)
    ret += logInt[d[:,1]].sum()
      
    # All done!..
    return ret

  def getNewNLL(self, dic, lock = True):
    """A helper method, that replicates getNLL but for a document that is not in the corpus. Takes as input a dictionary, as you would provide to the add method - it then clones self, adds the document, solves the model and return getNLL. If lock is True, the default, it locks all the existing model parameters, which is computationally useful, but if its False it lets them all change."""

    # Copy this object and add the document to the new one, after locking the existing documents...
    other = self.clone()
    other.lockAllDoc()
    other.lockBeta()
    d = other.add(dic)

    # Solve it, atking into account the locking request, though always converging with everything but the new guy locked first to save time...
    other.solve()
    if not lock:
      other.lockAllDoc(False)
      other.lockBeta(False)
      other.solve()

    # Calculate and return the value...
    return other.getNLL(d)
