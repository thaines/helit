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
import numpy
import scipy.special
import collections

import solvers

from smp.smp import FlagIndexArray
from utils.mp_map import *



class DocSample:
  """Stores the sample information for a given document - the DP from which topics are drawn and which cluster it is a member of. Also calculates and stores the negative log liklihood of the document."""
  def __init__(self, doc):
    """Given the specific DocState object this copies the relevant information. Note that it doesn't calculate the nll - another method will do that. It also supports cloning."""
    if isinstance(doc, DocSample): # Code for clonning
      self.cluster = doc.cluster
      self.dp = doc.dp.copy()
      self.conc = doc.conc
      self.samples = doc.samples.copy()
      self.behFlags = doc.behFlags.copy()
      self.nll = doc.nll
      self.ident = doc.ident
    else:
      # Extract the model information...
      self.cluster = doc.cluster
      self.dp = doc.use.copy()
      self.conc = doc.conc
      self.samples = doc.samples.copy()
      self.behFlags = doc.behFlags.copy()
      self.nll = 0.0
      self.ident = doc.ident

  def calcNLL(self, doc, state):
    """Calculates the negative log likelihood of the document, given the relevant information. This is the DocState object again, but this time with the entire state object as well. Probability (Expressed as negative log likelihood.) is specificly calculated using all terms that contain a variable in the document, but none that would be identical for all documents. That is, it contains the probability of the cluster, the probability of the dp given the cluster, and the probability of the samples, which factors in both the drawing of the topic and the drawing of the word. The ordering of the samples is considered irrelevant, with both the topic and word defining uniqueness. Some subtle approximation is made - see if you can spot it in the code!"""
    self.nll = 0.0

    # Probability of drawing the cluster...
    self.nll -= math.log(state.clusterUse[doc.cluster])
    self.nll += math.log(state.clusterUse.sum()+state.clusterConc)


    # Probability of drawing the documents dp from its cluster, taking into account the abnormal entrys...
    cl = state.cluster[doc.cluster]
    logBMN = numpy.log(cl[2] / (cl[2]*numpy.asfarray(doc.behFlags)).sum())

    behInstCounts = numpy.zeros(doc.behFlags.shape[0], dtype=numpy.int32)
    instCounts = numpy.zeros(cl[0].shape[0], dtype=numpy.int32)
    for ii in xrange(doc.use.shape[0]):
      behInstCounts[doc.use[ii,0]] += 1
      if doc.use[ii,0]==0: instCounts[doc.use[ii,1]] += 1

    self.nll -= (logBMN * behInstCounts).sum()
    self.nll -= scipy.special.gammaln(behInstCounts.sum() + 1.0)
    self.nll += scipy.special.gammaln(behInstCounts + 1.0).sum()

    norm = cl[0][:,1].sum() + cl[1]
    self.nll -= (numpy.log(numpy.asfarray(cl[0][:,1])/norm)*instCounts).sum()
    self.nll -= scipy.special.gammaln(instCounts.sum() + 1.0) # Cancels with a term from the above - can optimise, but would rather have neat code.
    self.nll += scipy.special.gammaln(instCounts + 1.0).sum()


    # Count the numbers of word/topic instance pairs in the data structure - sum using a dictionary...
    samp_count = collections.defaultdict(int) # [instance,word]
    for s in xrange(doc.samples.shape[0]):
      samp_count[doc.samples[s,0],doc.samples[s,1]] += 1

    # Calculate the probability distribution of drawing each topic instance and the probability of drawing each word/topic assignment...
    inst = numpy.asfarray(doc.use[:,2])
    inst /= inst.sum() + doc.conc
    
    topicWord = numpy.asfarray(state.topicWord) + state.beta
    topicWord = (topicWord.T/topicWord.sum(axis=1)).T

    abnormTopicWord = numpy.asfarray(state.abnormTopicWord) + state.beta
    abnormTopicWord = (abnormTopicWord.T/abnormTopicWord.sum(axis=1)).T

    instLog = numpy.log(inst)
    wordLog = numpy.log(topicWord)
    abnormLog = numpy.log(abnormTopicWord)


    # Now sum into nll the probability of drawing the samples that have been drawn - gets a tad complex as includes the probability of drawing the topic from the documents dp and then the probability of drawing the word from the topic, except I've merged them such that it doesn't look like that is what is happening...
    self.nll -= scipy.special.gammaln(doc.samples.shape[0]+1.0)
    for pair, count in samp_count.iteritems():
      inst, word = pair
      beh = doc.use[inst,0]
      if beh==0:
        topic = cl[0][doc.use[inst,1],0]
        self.nll -= count * (wordLog[topic,word] + instLog[inst])
      else:
        self.nll -= count * (abnormLog[beh,word] + instLog[inst])
      self.nll += scipy.special.gammaln(count+1.0)


  def getCluster(self):
    """Returns the sampled cluster assignment."""
    return self.cluster

  def getInstCount(self):
    """Returns the number of cluster instances in the documents model."""
    return self.dp.shape[0]

  def getInstBeh(self, i):
    """Returns the behaviour index for the given instance."""
    return self.dp[i,0]

  def getInstTopic(self, i):
    """Returns the topic index for the given instance."""
    return self.dp[i,1]

  def getInstWeight(self, i):
    """Returns the number of samples that have been assigned to the given topic instance."""
    return self.dp[i,2]

  def getInstAll(self):
    """Returns a 2D numpy array of integers where the first dimension indexes the topic instances for the document and the the second dimension has three entries, the first (0) the behaviour index, the second (1) the topic index and the third (2) the number of samples assigned to the given topic instance. Do not edit the return value for this method - copy it first."""
    return self.dp

  def getInstConc(self):
    """Returns the sampled concentration parameter."""
    return self.conc

  def getBehFlags(self):
    """Returns the behavioural flags - a 1D array of {0,1} as type unsigned char where 1 indicates that it has the behaviour with that index, 0 that it does not. Entry 0 will map to normal behaviour, and will always be 1. Do not edit - copy first."""
    return self.behFlags

  def getNLL(self):
    """Returns the negative log liklihood of the document given the model."""
    return self.nll

  def getIdent(self):
    """Returns the ident of the document, as passed through from the input document so they may be matched up."""
    return self.ident



class Sample:
  """Stores a single sample drawn from the model - the topics, clusters and each document being sampled over. Stores counts and parameters required to make them into distributions, rather than final distributions. Has clonning capability."""
  def __init__(self, state, calcNLL = True, priorsOnly = False):
    """Given a state this draws a sample from it, as a specific parametrisation of the model. Also a copy constructor, with a slight modification - if the priorsOnly flag is set it will only copy across the priors, and initialise to an empty model."""
    if isinstance(state, Sample): # Code for clonning.
      self.alpha = state.alpha
      self.beta = state.beta.copy()
      self.gamma = state.gamma
      self.rho = state.rho
      self.mu = state.mu
      self.phi = state.phi.copy()

      if not priorsOnly:
        self.topicWord = state.topicWord.copy()
        self.topicUse = state.topicUse.copy()
      else:
        self.topicWord = numpy.zeros((0,state.topicWord.shape[1]), dtype=numpy.int32)
        self.topicUse = numpy.zeros(0,dtype=numpy.int32)
      self.topicConc = state.topicConc

      self.abnormTopicWord = state.abnormTopicWord.copy()
      self.abnorms = dict(state.abnorms)
      self.fia = FlagIndexArray(state.fia)

      if not priorsOnly:
        self.cluster = map(lambda t: (t[0].copy(),t[1],t[2].copy(),t[3].copy()), state.cluster)
        self.clusterUse = state.clusterUse.copy()
      else:
        self.cluster = []
        self.clusterUse = numpy.zeros(0,dtype=numpy.int32)
      self.clusterConc = state.clusterConc

      if not priorsOnly:
        self.doc = map(lambda ds: DocSample(ds), state.doc)
      else:
        self.doc = []
    else: # Normal initialisation code.
      self.alpha = state.alpha
      self.beta = state.beta.copy()
      self.gamma = state.gamma
      self.rho = state.rho
      self.mu = state.mu
      self.phi = state.phi.copy()

      # Topic stuff...
      self.topicWord = state.topicWord.copy()
      self.topicUse = state.topicUse.copy()
      self.topicConc = state.topicConc

      # Abnormality stuff...
      self.abnormTopicWord = state.abnormTopicWord.copy()
      self.abnorms = dict(state.abnorms)
      self.fia = FlagIndexArray(state.fia)

      # Cluster stuff...
      self.cluster = map(lambda t: (t[0].copy(),t[1],t[2].copy(),t[3].copy()), state.cluster)
      self.clusterUse = state.clusterUse.copy()
      self.clusterConc = state.clusterConc

      # The details for each document...
      self.doc = []
      for d in xrange(len(state.doc)):
        self.doc.append(DocSample(state.doc[d]))

      # Second pass through documents to fill in the negative log liklihoods - need some data structures for this...
      if calcNLL:
        for d in xrange(len(state.doc)):
          self.doc[d].calcNLL(state.doc[d],state)


  def merge(self, other):
    """Given a sample this merges it into this sample. Works under the assumption that the new sample was learnt with this sample as its only prior, and ends up as though both the prior and the sample were drawn whilst simultaneously being modeled. Trashes the given sample - do not continue to use."""

    # Update the old documents - there are potentially more behaviours in the new sample, which means adjusting the behaviour flags...
    if self.fia.getLength()!=other.fia.getLength():
      for doc in self.doc:
        newBehFlags = numpy.zeros(other.fia.getLength(), dtype=numpy.uint8)
        newBehFlags[0] = doc.behFlags[0]

        for abnorm, index in self.abnorms:
          newIndex = other.abnorms[abnorm]
          newBehFlags[newIndex] = doc.behFlags[index]
        
        doc.behFlags = newBehFlags

    # Replace the basic parameters...
    self.alpha = other.alpha
    self.beta = other.beta
    self.gamma = other.gamma
    self.rho = other.rho
    self.mu = other.mu
    self.phi = other.phi

    self.topicWord = other.topicWord
    self.topicUse = other.topicUse
    self.topicConc = other.topicConc

    self.abnormTopicWord = other.abnormTopicWord
    self.abnorms = other.abnorms
    self.fia = other.fia

    self.cluster = other.cluster
    self.clusterUse = other.clusterUse
    self.clusterConc = other.clusterConc

    # Add in the (presumably) new documents...
    for doc in other.doc:
      self.doc.append(doc)


  def getAlphaPrior(self):
    """Returns the PriorConcDP that was used for the alpha parameter, which is the concentration parameter for the DP in each document."""
    return self.alpha

  def getBeta(self):
    """Returns the beta prior, which is a vector representing a Dirichlet distribution from which the multinomials for each topic are drawn, from which words are drawn."""
    return self.beta

  def getGammaPrior(self):
    """Returns the PriorConcDP that was used for the gamma parameter, which is the concentration parameter for the global DP from which topics are drawn."""
    return self.gamma
    
  def getRhoPrior(self):
    """Returns the PriorConcDP that was used for the rho parameter, which is the concentration parameter for each specific clusters DP."""
    return self.rho

  def getMuPrior(self):
    """Returns the PriorConcDP that was used for the mu parameter, which is the concentration parameter for the DP from which clusters are drawn."""
    return self.mu

  def getPhi(self):
    """Returns the phi Dirichlet distribution prior on the behavioural multinomial for each cluster."""
    return self.phi


  def getTopicCount(self):
    """Returns the number of topics in the sample."""
    return self.topicWord.shape[0]

  def getWordCount(self):
    """Returns the number of words in the topic multinomial."""
    return self.topicWord.shape[1]

  def getTopicUseWeight(self, t):
    """Returns how many times the given topic has been instanced in a cluster."""
    return self.topicUse[t]

  def getTopicUseWeights(self):
    """Returns an array, indexed by topic id, that contains how many times each topic has been instanciated in a cluster. Do not edit the return value - copy it first."""
    return self.topicUse

  def getTopicConc(self):
    """Returns the sampled concentration parameter for drawing topic instances from the global DP."""
    return self.topicConc
    
  def getTopicWordCount(self, t):
    """Returns the number of samples assigned to each word for the given topic, as an integer numpy array. Do not edit the return value - make a copy first."""
    return self.topicWord[t,:]

  def getTopicWordCounts(self, t):
    """Returns the number of samples assigned to each word for all topics, indexed [topic, word], as an integer numpy array. Do not edit the return value - make a copy first."""
    return self.topicWord

  def getTopicMultinomial(self, t):
    """Returns the calculated multinomial for a given topic ident."""
    ret = self.beta.copy()
    ret += self.topicWord[t,:]
    ret /= ret.sum()
    return ret

  def getTopicMultinomials(self):
    """Returns the multinomials for all topics, in a single array - indexed by [topic, word] to give P(word|topic)."""
    ret = numpy.vstack([self.beta]*self.topicWord.shape[0])
    ret += self.topicWord
    ret = (ret.T / ret.sum(axis=1)).T
    return ret


  def getBehCount(self):
    """Returns the number of behaviours, which is the number of abnormalities plus 1, and the entry count for the indexing variable for abnormals in the relevant methods."""
    return self.abnormTopicWord.shape[0]

  def getAbnormWordCount(self, b):
    """Returns the number of samples assigned to each word for the given abnormal topic. Note that entry 0 equates to normal behaviour and is a dummy that should be ignored."""
    return self.abnormTopicWord[b,:]

  def getAbnormWordCounts(self):
    """Returns the number of samples assigned to each word in each abnormal behaviour. An integer 2D array indexed with [behaviour, word], noting that behaviour 0 is a dummy for normal behaviour. Do not edit the return value - make a copy first."""
    return self.abnormTopicWord

  def getAbnormMultinomial(self, b):
    """Returns the calculated multinomial for a given abnormal behaviour."""
    ret = self.beta.copy()
    ret += self.abnormTopicWord[b,:]
    ret /= ret.sum()
    return ret

  def getAbnormMultinomials(self):
    """Returns the multinomials for all abnormalities, in a single array - indexed by [behaviour, word] to give P(word|topic associated with behaviour). Entry 0 is a dummy to fill in for normal behaviour, and should be ignored."""
    ret = numpy.vstack([self.beta]*self.abnormTopicWord.shape[0])
    ret += self.abnormTopicWord
    ret = (ret.T / ret.sum(axis=1)).T
    return ret


  def getAbnormDict(self):
    """Returns a dictionary that takes each abnormalities user provided token to the behaviour index used for it. Allows the use of the getAbnorm* methods, amung other things."""
    return self.abnorms


  def getClusterCount(self):
    """Returns how many clusters there are."""
    return len(self.cluster)

  def getClusterDrawWeight(self, c):
    """Returns how many times the given cluster has been instanced by a document."""
    return self.clusterUse[c]

  def getClusterDrawWeights(self):
    """Returns an array, indexed by cluster id, that contains how many times each cluster has been instanciated by a document. Do not edit the return value - copy it first."""
    return self.clusterUse

  def getClusterDrawConc(self):
    """Returns the sampled concentration parameter for drawing cluster instances for documents."""
    return self.clusterConc

  def getClusterInstCount(self, c):
    """Returns how many instances of topics exist in the given cluster."""
    return self.cluster[c][0].shape[0]
    
  def getClusterInstWeight(self, c, ti):
    """Returns how many times the given cluster topic instance has been instanced by a documents DP."""
    return self.cluster[c][0][ti,1]
    
  def getClusterInstTopic(self, c, ti):
    """Returns which topic the given cluster topic instance is an instance of."""
    return self.cluster[c][0][ti,0]

  def getClusterInstDual(self, c):
    """Returns a 2D array, where the first dimension is indexed by the topic instance, and the second contains two columns - the first the topic index, the second the weight. Do not edit return value - copy before use."""
    return self.cluster[c][0]

  def getClusterInstConc(self, c):
    """Returns the sampled concentration that goes with the DP from which the members of each documents DP are drawn."""
    return self.cluster[c][1]

  def getClusterInstBehMN(self, c):
    """Returns the multinomial on drawing behaviours for the given cluster."""
    return self.cluster[c][2]

  def getClusterInstPriorBehMN(self, c):
    """Returns the prior on the behaviour multinomial, as an array of integer counts aligned with the flag set."""
    return self.cluster[c][3]


  def docCount(self):
    """Returns the number of documents stored within. Should be the same as the corpus from which the sample was drawn."""
    return len(self.doc)

  def getDoc(self,d):
    """Given a document index this returns the appropriate DocSample object. These indices should align up with the document indices in the Corpus from which this Sample was drawn, assuming no documents have been deleted."""
    return self.doc[d]


  def delDoc(self, ident):
    """Given a document ident this finds the document with the ident and removes it from the model, completly - i.e. all the variables in the sample are also updated. Primarilly used to remove documents for resampling prior to using the model as a prior. Note that this can potentially leave entities with no users - they get culled when the model is loaded into the C++ data structure so as to not cause problems."""
    # Find and remove it from the document list...
    index = None
    for i in xrange(len(self.doc)):
      if self.doc[i].getIdent()==ident:
        index = i
        break
    if index==None: return

    victim = self.doc[index]
    self.doc = self.doc[:index] + self.doc[index+1:]
    

    # Update all the variables left behind by subtracting the relevant terms...
    cluster = self.cluster[victim.cluster]
    self.clusterUse[victim.cluster] -= 1

    ## First pass through the dp and remove its influence; at the same time note the arrays that need to be updated by each user when looping through...
    dp_ext = []
    for i in xrange(victim.dp.shape[0]):
      beh = victim.dp[i,0]
      #count = victim.dp[i,2]

      if beh==0: # Normal behaviour
        cluInst = victim.dp[i,1]

        # Update the instance, and topic use counts if necessary...
        topic = cluster[0][cluInst,0]
        cluster[0][cluInst,1] -= 1
        if cluster[0][cluInst,1]==0:
          self.topicUse[topic] -= 1

        # Store the entity that needs updating in correspondence with this dp instance in the next step...
        dp_ext.append((self.topicWord, topic))

      else: # Abnormal behaviour.
        # Store the entity that needs updating in correspondence with the dp...
        dp_ext.append((self.abnormTopicWord, beh))
    
    ## Go through the samples array and remove their influnce - the hard part was done by the preceding step...
    for si in xrange(victim.samples.shape[0]):
      inst = victim.samples[si,0]
      word = victim.samples[si,1]
      mat, topic = dp_ext[inst]
      mat[topic,word] -= 1

    # Clean up all zeroed items...
    self.cleanZeros()


  def cleanZeros(self):
    """Goes through and removes anything that has a zero reference count, adjusting all indices accordingly."""

    # Remove the zeros from this object, noting the changes...

    ## Topics...
    newTopicCount = 0
    topicMap = dict()
    for t in xrange(self.topicUse.shape[0]):
      if self.topicUse[t]!=0:
        topicMap[t] = newTopicCount
        newTopicCount += 1

    if newTopicCount!=self.topicUse.shape[0]:
      newTopicWord = numpy.zeros((newTopicCount, self.topicWord.shape[1]), dtype=numpy.int32)
      newTopicUse = numpy.zeros(newTopicCount,dtype=numpy.int32)

      for origin, dest in topicMap.iteritems():
        newTopicWord[dest,:] = self.topicWord[origin,:]
        newTopicUse[dest] = self.topicUse[origin]
      
      self.topicWord = newTopicWord
      self.topicUse = newTopicUse

    ## Clusters...
    newClusterCount = 0
    clusterMap = dict()
    for c in xrange(self.clusterUse.shape[0]):
      if self.clusterUse[c]!=0:
        clusterMap[c] = newClusterCount
        newClusterCount += 1

    if newClusterCount!=self.clusterUse.shape[0]:
      newCluster = [None]*newClusterCount
      newClusterUse = numpy.zeros(newClusterCount, dtype=numpy.int32)

      for origin, dest in clusterMap.iteritems():
        newCluster[dest] = self.cluster[origin]
        newClusterUse[dest] = self.clusterUse[origin]

      self.cluster = newCluster
      self.clusterUse = newClusterUse

    ## Cluster instances...
    # (Change is noted by a 2-tuple of (new length, dict) where new length is the new length and dict goes from old indices to new indices.)
    cluInstAdj = []
    for ci in xrange(len(self.cluster)):
      newInstCount = 0
      instMap = dict()
      for i in xrange(self.cluster[ci][0].shape[0]):
        if self.cluster[ci][0][i,1]!=0:
          instMap[i] = newInstCount
          newInstCount += 1

      cluInstAdj.append((newInstCount, instMap))

      if newInstCount!=self.cluster[ci][0].shape[0]:
        newInst = numpy.zeros((newInstCount,2), dtype=numpy.int32)

        for origin, dest in instMap.iteritems():
          newInst[dest,:] = self.cluster[ci][0][origin,:]

        self.cluster[ci] = (newInst, self.cluster[ci][1], self.cluster[ci][2], self.cluster[ci][3])


    # Iterate and update the topic indices of the cluster instances...
    for ci in xrange(len(self.cluster)):
      for i in xrange(self.cluster[ci][0].shape[0]):
        self.cluster[ci][0][i,0] = topicMap[self.cluster[ci][0][i,0]]

    # Now iterate the documents and update their cluster and cluster instance indices...
    for doc in self.doc:
      doc.cluster = clusterMap[doc.cluster]
      _, instMap = cluInstAdj[doc.cluster]

      for di in xrange(doc.dp.shape[0]):
        if doc.dp[di,0]==0:
          doc.dp[di,1] = instMap[doc.dp[di,1]]


  def nllAllDocs(self):
    """Returns the negative log likelihood of all the documents in the sample - a reasonable value to compare various samples with."""
    return sum(map(lambda d: d.getNLL(),self.doc))

  def logNegProbWordsGivenClusterAbnorm(self, doc, cluster, particles = 16, cap = -1):
    """Uses wallach's 'left to right' method to calculate the negative log probability of the words in the document given the rest of the model. Both the cluster (provided as an index) and the documents abnormalities vector are fixed for this calculation. Returns the average of the results for each sample contained within model. particles is the number of particles to use in the left to right estimation algorithm. This is implimented using scipy.weave."""
    return solvers.leftRightNegLogProbWord(self, doc, cluster, particles, cap)

  def logNegProbWordsGivenAbnorm(self, doc, particles = 16, cap = -1):
    """Uses logNegProbWordsGivenClusterAbnorm and simply sums out the cluster variable."""

    # Get the probability of each with the dependence with clusters...
    cluScores = map(lambda c: solvers.leftRightNegLogProbWord(self, doc, c, particles, cap), xrange(self.getClusterCount()))

    # Multiply each by the probability of the cluster, so it can be summed out...
    cluNorm = float(self.clusterUse.sum()) + self.clusterConc
    cluScores = map(lambda c,s: s - math.log(float(self.clusterUse[c])/cluNorm), xrange(len(cluScores)), cluScores)

    # Also need to include the probability of a new cluster, even though it is likelly to be a neglible contribution...
    newVal = solvers.leftRightNegLogProbWord(self, doc, -1, particles, cap)
    newVal -= math.log(self.clusterConc/cluNorm)
    cluScores.append(newVal)

    # Sum out the cluster variable, in a numerically stable way given that we are dealing with negative log likelihood values that will map to extremelly low probabilities...
    minScore = min(cluScores)
    cluPropProb = map(lambda s: math.exp(minScore-s), cluScores)
    return minScore - math.log(sum(cluPropProb))



class Model:
  """Simply contains a list of samples taken from the state during Gibbs iterations. Has clonning capability."""
  def __init__(self, obj=None, priorsOnly = False):
    """If provided with a Model will clone it."""
    self.sample = []
    if isinstance(obj, Model):
      for sample in obj.sample:
        self.sample.append(Sample(sample, priorsOnly))

  def add(self, sample):
    """Adds a sample to the model."""
    self.sample.append(sample)

  def sampleState(self, state):
    """Samples the state, storing the sampled model within."""
    self.sample.append(Sample(state))

  def absorbModel(self, model):
    """Given another model this absorbs all its samples, leaving then given model baren."""
    self.sample += model.sample
    model.sample = []

  def sampleCount(self):
    """Returns the number of samples."""
    return len(self.sample)

  def getSample(self, s):
    """Returns the sample associated with the given index."""
    return self.sample[s]

  def sampleList(self):
    """Returns a list of samples, for iterating."""
    return self.sample


  def delDoc(self, ident):
    """Calls the delDoc method for the given ident on all samples contained within."""
    for sample in self.sample:
      sample.delDoc(ident)


  def bestSampleOnly(self):
    """Calculates the document nll for each sample and prunes all but the one with the highest - very simple way of 'merging' multiple samples together."""
    score = map(lambda s: s.nllAllDocs(),self.sample)
    best = 0
    for i in xrange(1,len(self.sample)):
      if score[i]>score[best]:
        best = i

    self.sample = [self.sample[best]]


  def fitDoc(self, doc, params = None, callback=None, mp = True):
    """Given a document this returns a DocModel calculated by Gibbs sampling the document with the samples in the model as priors. Returns a DocModel. Note that it samples using params for *each* sample in the Model, so you typically want to use less than the defaults in Params, typically only a single run and sample, which is the default. mp can be set to False to force it to avoid multi-processing behaviour"""
    if mp and len(self.sample)>1 and hasattr(solvers,'gibbs_doc_mp'):
      return solvers.gibbs_doc_mp(self, doc, params, callback)
    else:
      return solvers.gibbs_doc(self, doc, params, callback)


  def logNegProbWordsGivenAbnorm(self, doc, particles = 16, cap = -1, mp = True):
    """Calls the function of the same name for each sample and returns the average of the various return values."""
    if mp and len(self.sample)>1:
      sampNLL = mp_map(lambda s,d,p,c: s.logNegProbWordsGivenAbnorm(d,p,c), self.sample, repeat(doc), repeat(particles), repeat(cap))
    else:
      sampNLL = map(lambda s: s.logNegProbWordsGivenAbnorm(doc,particles,cap), self.sample)
    

    ret = 0.0 # Negative log prob
    retCount = 0.0
    for nll in sampNLL:
      retCount += 1.0
      if retCount < 1.5:
        ret = nll
      else:
        ret -= math.log(1.0 + (math.exp(ret-nll) - 1.0)/retCount)

    return ret

  def logNegProbAbnormGivenWords(self, doc, epsilon = 0.1, particles = 16, cap = -1):
    """Returns the probability of the documents current abnormality flags - uses Bayes rule on logNegProbAbnormGivenWords. Does not attempt to calculate the normalising constant, so everything is with proportionality - you can compare flags for a document, but can't compare between different documents. You actually provide epsilon to the function, as its not calculated anywhere. You can either provide a number, in which case that is the probability of each abnormality, or you can provide a numpy vector of probabilities, noting that the first entry must correspond to normal and be set to 1.0"""

    # Handle the conveniance input of providing a single floating point value for epsilon rather than a numpy array...
    if not isinstance(epsilon,numpy.ndarray):
      # Assume its a floating point number and build epsilon as an array...
      value = epsilon
      epsilon = numpy.ones(self.sample[0].phi.shape[0], dtype=numpy.float32)
      epsilon *= value
      epsilon[0] = 1.0

    # Generate flags for the document...
    flags = numpy.zeros(self.sample[0].phi.shape[0], dtype=numpy.uint8)
    flags[0] = 1
    for abnorm in doc.getAbnorms():
      flags[self.sample[0].abnorms[abnorm]] = 1
    
    # Apply Bayes - hardly hard!..
    ret = self.logNegProbWordsGivenAbnorm(doc, particles, cap)
    for i in xrange(1,epsilon.shape[0]):
      if flags[i]!=0: ret -= math.log(epsilon[i])
      else: ret -= math.log(1.0-epsilon[i])
    return ret

  def mlDocAbnorm(self, doc, lone = False, epsilon = 0.1, particles = 16, cap = -1):
    """Decides which abnormalities most likelly exist in the document, using the logNegProbAbnormGivenWords method. Returns the list of abnormalities that are most likelly to exist. It does a greedy search of the state space - by default it considers all states, but setting the lone flag to true it will only consider states with one abnormality."""
    
    # A dictionary that contains True to indicate that the indexed tuple of abnormalities has already been tried (And, effectivly, rejected.)...
    tried = dict()

    # Starting state...
    best = []
    doc.setAbnorms(best)
    bestNLL = self.logNegProbAbnormGivenWords(doc, epsilon, particles, cap)
    tried[tuple(best)] = True

    # Iterate until no change...
    while True:
      newBest = best
      newBestNLL = bestNLL

      for abnorm in self.sample[0].abnorms.iterkeys():
        test = best[:]
        if abnorm in test: test = filter(lambda a:a!=abnorm, test)
        else: test.append(abnorm)
        if lone and len(test)>1: continue
        doc.setAbnorms(test)

        if tuple(test) not in tried:
          tried[tuple(test)] = True

          doc.setAbnorms(test)
          testNLL = self.logNegProbAbnormGivenWords(doc, epsilon, particles, cap)

          if testNLL<newBestNLL:
            newBest = test
            newBestNLL = testNLL

      if best==newBest:
        doc.setAbnorms(best)
        return best
      else:
        best = newBest
        bestNLL = newBestNLL



class DocModel:
  """A Model that just contains DocSample-s for a single document. Obviously incomplete without a full Model, this is typically used when sampling a document relative to an already trained Model, such that the topic/cluster indices will match up with the original Model. Note that if the document has enough data to justify the creation of an extra topic/cluster then that could exist with an index above the indices of the topics/clusters in the source Model."""
  def __init__(self, obj=None):
    """Supports cloning."""
    self.sample = []
    if isinstance(obj, DocModel):
      for sample in obj.sample:
        self.sample.append(DocSample(sample))

  def addFrom(self, model, index=0):
    """Given a model and a document index number extracts all the relevant DocSample-s, adding them to this DocModel. It does not edit the Model but the DocSample-s transfered over are the same instances."""
    for s in xrange(model.sampleCount()):
      self.sample.append(model.getSample(s).getDoc(index))

  def absorbModel(self, dModel):
    """Absorbs samples from the given DocModel, leaving it baren."""
    self.sample += dModel.sample
    dModel.sample = []

  def sampleCount(self):
    """Returns the number of samples contained within."""
    return len(self.sample)

  def getSample(self, s):
    """Returns the sample with the given index, in the range 0..sampleCount()-1"""
    return self.sample[s]

  def sampleList(self):
    """Returns a list of samples, for iterating."""
    return self.sample

  def hasAbnormal(self, name, abnormDict):
    """Given the key for an abnormality (Typically a string - as provided to the Document object orginally.) returns the probability this document has it, by looking through the samples contained within. Requires an abnorm dictionary, as obtained from the getAbnormDict method of a sample."""
    if name not in abnormDict: return 0.0
    index = abnormDict[name]

    count = 0
    for s in self.sample:
      if s.getBehFlags()[index]!=0: count += 1

    return float(count)/float(len(self.sample))

  def getNLL(self):
    """Returns the average nll of all the contained samples - does a proper mean of the probability of the samples."""
    minSam = min(map(lambda s:s.getNLL(),self.sample))
    probMean = sum(map(lambda s:math.exp(minSam-s.getNLL()),self.sample))
    probMean /= float(len(self.sample))
    return minSam - math.log(probMean)
