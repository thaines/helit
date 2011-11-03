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

from dp_conc import PriorConcDP
from params import Params
from smp.smp import FlagIndexArray



class DocState:
  """Helper class to contain the State of Gibbs sampling for a specific document."""
  def __init__(self, doc, alphaConc = None, abnormDict = None):
    if isinstance(doc, DocState):
      self.cluster = doc.cluster
      self.use = doc.use.copy()
      self.conc = doc.conc
      self.samples = doc.samples.copy()
      self.behFlags = doc.behFlags.copy()
      self.behFlagsIndex = doc.behFlagsIndex
      self.behCounts = doc.behCounts.copy()
      self.ident = doc.ident
    else:
      # Index of the cluster its assigned to, initialised to -1 to indicate it is not currently assigned...
      self.cluster = -1

      # Definition of the documents DP, initialised to be empty, which contains instances of cluster instances. The use array is, as typical, indexed by instance in the first dimension and {0,1,2} in the second, where 0 gives the behaviour, 1 the index of the cluster instance it is instancing and 2 gives the number of users, which at this level will be the number of words. conc provides a sample of the concentration value for the DP...
      self.use = numpy.empty((0,3), dtype=numpy.int32)
      self.conc = alphaConc.conc

      # Contains the documents samples - a 2D array where the first dimension indexes each sample. There are then two columns - the first contains the instance index, which indexes the use array, and the second the word index, which indexes the multinomial assigned to each topic. We default to -1 in the instance index column to indicate that it is unassigned...
      self.samples = numpy.empty((doc.getSampleCount(),2), dtype=numpy.int32)
      self.samples[:,0] = -1
      si = 0
      for word, count in map(lambda i: doc.getWord(i), xrange(doc.getWordCount())):
        for _ in xrange(count):
          self.samples[si,1] = word
          si += 1
      assert(si==doc.getSampleCount())

      # Create the documents behaviour flags, which are an array of {0,1} values indicating if the given behaviour is in the document or not - entry 0 is reserved for normal behaviour, whilst the rest are reserved for abnormalities...
      self.behFlags = numpy.zeros(1+len(abnormDict), dtype=numpy.uint8)
      self.behFlags[0] = 1 # Normal behaviour always exists.
      for abnorm in doc.getAbnorms():
        self.behFlags[abnormDict[abnorm]] = 1

      # Index associated with the above behFlags - set on initialisation to match the corpus's FlagIndexArray...
      self.behFlagsIndex = -1

      # We also need the behaviour counts - how many samples have been assigned to each behaviour...
      self.behCounts = numpy.zeros(1+len(abnormDict), dtype=numpy.int32)

      self.ident = doc.getIdent()



class State:
  """State object, as manipulated by a Gibbs sampler to get samples of the unknown parameters of the model."""
  def __init__(self, obj, params = None):
    """Constructs a state object given either another State object (clone), or a Corpus and a Params object. If the Params object is omitted it uses the default. Also supports construction from a single Document, where it uses lots of defaults but is basically identical to a Corpus with a single Document in - used as a shortcut when fitting a Document to an already learnt model."""
    if isinstance(obj, State):
      # Cloning time...
      self.dnrDocInsts = obj.dnrDocInsts
      self.dnrCluInsts = obj.dnrCluInsts
      self.seperateClusterConc = obj.seperateClusterConc
      self.seperateDocumentConc = obj.seperateDocumentConc
      self.oneCluster = obj.oneCluster
      self.calcBeta = obj.calcBeta
      self.calcCluBmn = obj.calcCluBmn
      self.calcPhi = obj.calcPhi
      self.resampleConcs = obj.resampleConcs
      self.behSamples = obj.behSamples
      
      self.alpha = PriorConcDP(obj.alpha)
      self.beta = obj.beta.copy()
      self.gamma = PriorConcDP(obj.gamma)
      self.rho = PriorConcDP(obj.rho)
      self.mu = PriorConcDP(obj.mu)
      self.phi = obj.phi.copy()

      self.topicWord = obj.topicWord.copy()
      self.topicUse = obj.topicUse.copy()
      self.topicConc = obj.topicConc

      self.abnormTopicWord = obj.abnormTopicWord.copy()

      self.cluster = map(lambda t: (t[0].copy(),t[1],t[2].copy()),obj.cluster)
      self.clusterUse = obj.clusterUse.copy()
      self.clusterConc = obj.clusterConc

      self.doc = map(lambda d: DocState(d), obj.doc)
      self.abnorms = dict(obj.abnorms)

      self.fia = FlagIndexArray(obj.fia)

      self.params = Params(obj.params)
      self.model = Model(obj.model)

    elif isinstance(obj, Document):
      # Construct from a single document...

      self.dnrDocInsts = False
      self.dnrCluInsts = False
      self.seperateClusterConc = False
      self.seperateDocumentConc = False
      self.oneCluster = False
      self.calcBeta = False
      self.calcCluBmn = False
      self.calcPhi = False
      self.resampleConcs = False
      self.behSamples = 1024

      wordCount = obj.getWord(obj.getWordCount()-1)[0]

      self.alpha = PriorConcDP()
      self.beta = numpy.ones(wordCount, dtype=numpy.float32)
      self.gamma = PriorConcDP()
      self.rho = PriorConcDP()
      self.mu = PriorConcDP()
      self.phi = numpy.ones(1+len(obj.getAbnorms()), dtype=numpy.float32)
      self.phi[0] *= 10.0
      self.phi /= self.phi.sum()

      self.topicWord = numpy.zeros((0,wordCount), dtype=numpy.int32)
      self.topicUse = numpy.zeros(0,dtype=numpy.int32)
      self.topicConc = self.gamma.conc

      self.abnormTopicWord = numpy.zeros((1+len(obj.getAbnorms()), wordCount), dtype=numpy.int32)

      self.cluster = []
      self.clusterUse = numpy.zeros(0,dtype=numpy.int32)
      self.clusterConc = self.mu.conc

      abnormDict = dict()
      for i, abnorm in enumerate(obj.getAbnorms()):
        abnormDict[abnorm] = i+1
        
      self.doc = [DocState(obj,self.alpha,abnormDict)]
      self.abnorms = dict()
      for num, abnorm in enumerate(obj.getAbnorms()):
        self.abnorms[abnorm] = num+1

      self.fia = FlagIndexArray(len(self.abnorms)+1)
      self.fia.addSingles()

      for doc in self.doc:
        doc.behFlagsIndex = self.fia.flagIndex(doc.behFlags)

      if params!=None: self.params = params
      else: self.params = Params()

      self.model = Model()
    else:
      # Construct from a corpus, as that is the only remaining option...

      # Behaviour flags...
      self.dnrDocInsts = obj.getDocInstsDNR()
      self.dnrCluInsts = obj.getCluInstsDNR()
      self.seperateClusterConc = obj.getSeperateClusterConc()
      self.seperateDocumentConc = obj.getSeperateDocumentConc()
      self.oneCluster = obj.getOneCluster()
      self.calcBeta = obj.getCalcBeta()
      self.calcCluBmn = obj.getCalcClusterBMN()
      self.calcPhi = obj.getCalcPhi()
      self.resampleConcs = obj.getResampleConcs()
      self.behSamples = obj.getBehSamples()

      # Concentration parameters - these are all constant...
      self.alpha = PriorConcDP(obj.getAlpha())
      self.beta = numpy.ones(obj.getWordCount(),dtype=numpy.float32)
      self.beta *= obj.getBeta()
      self.gamma = PriorConcDP(obj.getGamma())
      self.rho = PriorConcDP(obj.getRho())
      self.mu = PriorConcDP(obj.getMu())

      self.phi = numpy.ones(1+len(obj.getAbnormDict()), dtype=numpy.float32)
      self.phi[0] *= obj.getPhiRatio()
      self.phi *= obj.getPhiConc()*self.phi.shape[0] / self.phi.sum()

      # The topics in the model - consists of three parts - first an array indexed by [topic,word] which gives how many times each word has been drawn from the given topic - this alongside beta allows the relevant Dirichlet posterior to be determined. Additionally we have topicUse, which counts how many times each topic has been instanced in a cluster - this alongside topicConc, which is the sampled concentration, defines the DP from which topics are drawn for inclusion in clusters...
      self.topicWord = numpy.zeros((0,obj.getWordCount()),dtype=numpy.int32)
      self.topicUse = numpy.zeros(0,dtype=numpy.int32)
      self.topicConc = self.gamma.conc

      # A second topicWord-style matrix, indexed by behaviour and containing the abnormal topics. Entry 0, which is normal, is again an empty dummy...
      self.abnormTopicWord = numpy.zeros((1+len(obj.getAbnormDict()), obj.getWordCount()), dtype=numpy.int32)

      # Defines the clusters, as a list of (inst, conc, bmn, bmnPrior). inst is a 2D array, containing all the topic instances that make up the cluster - whilst the first dimension of the array indexes each instance the second has two entrys only, the first the index number for the topic, the second the number of using document instances. conc is the sampled concentration that completes the definition of the DP defined for each cluster. bmn is the multinomial on behaviours associated with the cluster - a 1D array of floats. bmnPrior is the flagSet aligned integer array that is the prior on bmn. Additionally we have the DDP from which the specific clusters are drawn - this is defined by clusterUse and clusterConc, just as for the topics...
      self.cluster = []
      self.clusterUse = numpy.zeros(0, dtype=numpy.int32)
      self.clusterConc = self.mu.conc

      # List of document objects, to contain the documents - whilst declared immediatly below as an empty list we then proceed to fill it in with the information from the given Corpus...
      self.doc = []

      for doc in obj.documentList():
        self.doc.append(DocState(doc, self.alpha, obj.getAbnormDict()))

      # The abnormality dictionary - need a copy so we can convert from flags to the user provided codes after fitting the model...
      self.abnorms = dict(obj.getAbnormDict())

      # The flag index array - converts each flag combination to an index - required for learning the per-cluster behaviour multinomials...
      self.fia = FlagIndexArray(len(self.abnorms)+1)
      self.fia.addSingles()

      for doc in self.doc:
        doc.behFlagsIndex = self.fia.flagIndex(doc.behFlags)

      # Store the parameters...
      if params!=None: self.params = params
      else: self.params = Params()

      # Create a model object, for storing samples into...
      self.model = Model()


  def setGlobalParams(self, sample):
    """Sets a number of parameters for the State after initialisation, taking them from the given Sample object. Designed for use with the addPrior method this allows you to extract all relevant parameters from a Sample. Must be called before any Gibbs sampling takes place."""
    self.alpha = PriorConcDP(sample.alpha)
    self.beta = sample.beta.copy()
    self.gamma = PriorConcDP(sample.gamma)
    self.rho = PriorConcDP(sample.rho)
    self.mu = PriorConcDP(sample.mu)

    # No correct way of combining - the below seems reasonable enough however, and is correct if they have the same entrys...
    for key,fromIndex in sample.abnorms.iteritems():
      if key in self.abnorms:
        toIndex = self.abnorms[key]
        self.phi[toIndex] = sample.phi[fromIndex]
    self.phi /= self.phi.sum()

    self.topicConc = sample.topicConc
    self.clusterConc = sample.clusterConc
    for doc in self.doc:
      doc.conc = self.alpha.conc
  
  def addPrior(self, sample):
    """Given a Sample object this uses it as a prior - this is primarilly used to sample a single or small number of documents using a model already trainned on another set of documents. It basically works by adding the topics, clusters and behaviours from the sample into this corpus, with the counts all intact so they have the relevant weight and can't be deleted. Note that you could in principle add multiple priors, though that would be quite a strange scenario. If only called once then the topic indices will line up. Note that all the prior parameters are not transfered, though often you would want to - setGlobalParams is provided to do this. Must be called before any Gibbs sampling takes place."""

    # Below code has evolved into spagetti, via several other tasty culinary dishes, and needs a rewrite. Or to never be looked at or edited ever again. ###################
    
    # Do the topics...
    offset = self.topicWord.shape[0]
    if self.topicWord.shape[0]!=0:
      self.topicWord = numpy.vstack((self.topicWord,sample.topicWord))
    else:
      self.topicWord = sample.topicWord.copy()
    self.topicUse = numpy.hstack((self.topicUse,sample.topicUse))

    # Calculate the new abnormalities dictionary...
    newAbnorms = dict(sample.abnorms)
    for key,_ in self.abnorms.iteritems():
      if key not in newAbnorms:
        val = len(newAbnorms)+1
        newAbnorms[key] = val

    # Transfer over the abnormal word counts...
    newAbnormTopicWord = numpy.zeros((1+len(newAbnorms), max((self.abnormTopicWord.shape[1], sample.abnormTopicWord.shape[1]))), dtype=numpy.int32)

    for abnorm,origin in self.abnorms.iteritems():
      dest = newAbnorms[abnorm]
      limit = self.abnormTopicWord.shape[1]
      newAbnormTopicWord[dest,:limit] += self.abnormTopicWord[origin,:limit]

    for abnorm,origin in sample.abnorms.iteritems():
      dest = newAbnorms[abnorm]
      limit = sample.abnormTopicWord.shape[1]
      newAbnormTopicWord[dest,:limit] += sample.abnormTopicWord[origin,:limit]

    # Update the document flags/counts for behaviours...
    for doc in self.doc:
      newFlags = numpy.zeros(1+len(newAbnorms), dtype=numpy.uint8)
      newCounts = numpy.zeros(1+len(newAbnorms), dtype=numpy.int32)
      newFlags[0] = doc.behFlags[0]
      newCounts[0] = doc.behCounts[0]

      for abnorm,origin in self.abnorms.iteritems():
        dest = newAbnorms[abnorm]
        newFlags[dest] = doc.behFlags[origin]
        newCounts[dest] = doc.behCounts[origin]
      
      doc.behFlags = newFlags
      doc.behCounts = newCounts

    # Update the old clusters behaviour arrays...
    def mapOldCluster(c):
      c2 = numpy.ones(1+len(newAbnorms), dtype=numpy.float32)
      c2 /= c2.sum()
      
      c2[0] *= c[2][0]
      for abnorm,origin in self.abnorms.iteritems():
        dest = newAbnorms[abnorm]
        c2[dest] *= c[2][origin]
      c2 /= c2.sum()
      
      return (c[0],c[1],c2,c[3])
      
    self.cluster = map(mapOldCluster ,self.cluster)
    origCluCount = len(self.cluster)
    
    # Add the new clusters, updating their behaviour arrays and topic indices, plus getting their priors updated with their associated documents...
    def mapCluster(pair):
      ci, c = pair
      
      c0 = c[0].copy()
      c0[:,0] += offset

      c2 = numpy.ones(1+len(newAbnorms), dtype=numpy.float32)
      c2 /= c2.sum()

      c2[0] *= c[2][0]
      for abnorm,origin in sample.abnorms.iteritems():
        dest = newAbnorms[abnorm]
        c2[dest] *= c[2][origin]
      c2 /= c2.sum()

      c3 = c[3].copy()
      for doc in filter(lambda doc: doc.cluster==ci, sample.doc):
        fi = sample.fia.flagIndex(doc.behFlags, False)
        if fi>=len(doc.behFlags): # Only bother if the document has abnormalities, of which this is a valid test.
          total = 0
          for i in xrange(doc.dp.shape[0]):
            c3[doc.dp[i,0]] += doc.dp[i,2]
            total += doc.dp[i,2]
          c3[fi] -= total + 1
      
      return (c0,c[1],c2,c3)
      
    self.cluster += map(mapCluster, enumerate(sample.cluster))
    self.clusterUse = numpy.hstack((self.clusterUse, sample.clusterUse))
    
    # Update phi...
    newPhi = numpy.ones(len(newAbnorms)+1,dtype=numpy.float32)
    newPhi[0] = 0.5*(self.phi[0]+sample.phi[0])
    
    for abnorm,origin in self.abnorms.iteritems():
      dest = newAbnorms[abnorm]
      newPhi[dest] = self.phi[origin]
    for abnorm,origin in sample.abnorms.iteritems():
      dest = newAbnorms[abnorm]
      if abnorm not in self.abnorms:
        newPhi[dest] = sample.phi[origin]
      else:
        newPhi[dest] = 0.5*(newPhi[dest] + sample.phi[origin])
      
    self.phi = newPhi
    self.phi /= self.phi.sum()

    # Recreate the flag index array...
    remapOrig = dict() # Old flag positions to new flag positions.
    remapOrig[0] = 0
    for abnorm,origin in self.abnorms.iteritems():
      remapOrig[origin] = newAbnorms[abnorm]

    remapSam = dict() # sample flag positions to new flag positions.
    remapSam[0] = 0
    for abnorm,origin in sample.abnorms.iteritems():
      remapSam[origin] = newAbnorms[abnorm]
    
    newFia = FlagIndexArray(len(newAbnorms)+1)
    newFia.addSingles()
    behIndAdjOrig = newFia.addFlagIndexArray(self.fia,remapOrig)
    behIndAdjSam  = newFia.addFlagIndexArray(sample.fia,remapSam)

    for doc in self.doc:
      doc.behFlagsIndex = behIndAdjOrig[doc.behFlagsIndex]

    # Update cluster priors on bmn arrays...
    for c in xrange(len(self.cluster)):
      clu = self.cluster[c]
      newBmn = numpy.zeros(newFia.flagCount(),dtype=numpy.int32)
      oldBmn = clu[3].copy()

      # Transilate from old set...
      for b in xrange(oldBmn.shape[0]):
        index = behIndAdjOrig[b] if c<origCluCount else behIndAdjSam[b]
        newBmn[index] += oldBmn[b]

      self.cluster[c] = (clu[0], clu[1], clu[2], newBmn)

    # Replace the old abnormality and fia stuff...
    self.abnormTopicWord = newAbnormTopicWord
    self.abnorms = newAbnorms
    self.fia = newFia


  def sample(self):
    """Samples the current state, storing the current estimate of the model parameters."""
    self.model.sampleState(self)

  def absorbClone(self,clone):
    """Given a clone absorb all its samples - used for multiprocessing."""
    self.model.absorbModel(clone.model)


  def getParams(self):
    """Returns the parameters object."""
    return self.params
    
  def getModel(self):
    """Returns the model constructed from all the calls to sample()."""
    return self.model



# Includes at tail of file to resolve circular dependencies...
from document import Document
from model import Model
