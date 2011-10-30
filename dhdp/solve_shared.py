# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import math
import numpy

from dp_conc import PriorConcDP
from params import Params



class DocState:
  """Helper class to contain the State of Gibbs sampling a specific document."""
  def __init__(self, doc, alphaConc = None):
    if isinstance(doc, DocState):
      self.cluster = doc.cluster
      self.use = doc.use.copy()
      self.conc = doc.conc
      self.samples = doc.samples.copy()
    else:
      # Index of the cluster its assigned to, initialised to -1 to indicate it is not currently assigned...
      self.cluster = -1

      # Definition of the documents DP, initialised to be empty, which contains instances of cluster instances. The use array is, as typical, indexed by instance in the first dimension and 0 or 1 in the second, where 0 gives the index of what it is instancing and 1 gives the number of users, which at this level will be the number of words. conc provides a sample of the concentration value for the DP...
      self.use = numpy.empty((0,2), dtype=numpy.int32)
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
      
      self.alpha = PriorConcDP(obj.alpha)
      self.beta = obj.beta.copy()
      self.gamma = PriorConcDP(obj.gamma)
      self.rho = PriorConcDP(obj.rho)
      self.mu = PriorConcDP(obj.mu)

      self.topicWord = obj.topicWord.copy()
      self.topicUse = obj.topicUse.copy()
      self.topicConc = obj.topicConc

      self.cluster = map(lambda t: (t[0].copy(),t[1]),obj.cluster)
      self.clusterUse = obj.clusterUse.copy()
      self.clusterConc = obj.clusterConc

      self.doc = map(lambda d: DocState(d), obj.doc)

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

      wordCount = obj.getWord(obj.getWordCount()-1)[0]

      self.alpha = PriorConcDP()
      self.beta = numpy.ones(wordCount,dtype=numpy.float32)
      self.gamma = PriorConcDP()
      self.rho = PriorConcDP()
      self.mu = PriorConcDP()

      self.topicWord = numpy.zeros((0,wordCount), dtype=numpy.int32)
      self.topicUse = numpy.zeros(0,dtype=numpy.int32)
      self.topicConc = self.gamma.conc

      self.cluster = []
      self.clusterUse = numpy.zeros(0,dtype=numpy.int32)
      self.clusterConc = self.mu.conc

      self.doc = [DocState(obj,self.alpha)]

      if params!=None: self.params = params
      else: self.params = Params()

      self.model = Model()
    else:
      # Construct from a corpus, as that is the only left out option...

      # Behaviour flags...
      self.dnrDocInsts = obj.getDocInstsDNR()
      self.dnrCluInsts = obj.getCluInstsDNR()
      self.seperateClusterConc = obj.getSeperateClusterConc()
      self.seperateDocumentConc = obj.getSeperateDocumentConc()
      self.oneCluster = obj.getOneCluster()
      self.calcBeta = obj.getCalcBeta()

      # Concentration parameters - these are all constant...
      self.alpha = obj.getAlpha()
      self.beta = numpy.ones(obj.getWordCount(),dtype=numpy.float32)
      self.beta *= obj.getBeta()
      self.gamma = obj.getGamma()
      self.rho = obj.getRho()
      self.mu = obj.getMu()

      # The topics in the model - consists of three parts - first an array indexed by [topic,word] which gives how many times each word has been drawn from the given topic - this alongside beta allows the relevant Dirichlet posterior to be determined. Additionally we have topicUse, which counts how manny times each topic has been instanced in a cluster - this alongside topicConc, which is the sampled concentration, defines the DP from which topics are drawn for inclusion in clusters.
      self.topicWord = numpy.zeros((0,obj.getWordCount()),dtype=numpy.int32)
      self.topicUse = numpy.zeros(0,dtype=numpy.int32)
      self.topicConc = self.gamma.conc

      # Defines the clusters, as a list of (inst, conc). inst is a 2D array, containing all the topic instances that make up the cluster - whilst the first dimension of the array indexes each instance the second has two entrys only, the first the index number for the topic, the second the number of using document instances. conc is the sampled concentration that completes the definition of the DP defined for each cluster. Additionally we have the DDP from which the specific clusters are drawn - this is defined by clusterUse and clusterConc, just as for the topics.
      self.cluster = []
      self.clusterUse = numpy.zeros(0,dtype=numpy.int32)
      self.clusterConc = self.mu.conc

      # List of document objects, to contain the documents - whilst declared immediatly below as an empty list we then proceed to fill it in with the information from the given Corpus...
      self.doc = []

      for doc in obj.documentList():
        self.doc.append(DocState(doc,self.alpha))

      # Store the parameters...
      if params!=None: self.params = params
      else: self.params = Params()

      # Create a model object, for storing samples into...
      self.model = Model()


  def setGlobalParams(self, sample):
    """Sets a number of parameters for the State after initialisation, taking them from the given Sample object. Designed for use with the addPrior method this allows you to extract all relevant parameters from a Sample. Must be called before any Gibbs sampling takes place."""
    self.alpha = sample.alpha
    self.beta = sample.beta.copy()
    self.gamma = sample.gamma
    self.rho = sample.rho
    self.mu = sample.mu

    self.topicConc = sample.topicConc
    self.clusterConc = sample.clusterConc
    for doc in self.doc:
      doc.conc = self.alpha.conc
  
  def addPrior(self, sample):
    """Given a Sample object this uses it as a prior - this is primarilly used to sample a single or small number of documents using a model already trainned on another set of documents. It basically works by adding the topics and clusters from the sample into this corpus, with the counts all intact so they have the relevant weight and can't be deleted. Note that you could in principle add multiple priors, though that would be quite a strange scenario. If only called once then the topic indices will line up. Note that all the prior parameters are not transfered, though often you would want to - setGlobalParams is provided to do this. Must be called before any Gibbs sampling takes place."""
    offset = self.topicWord.shape[0]
    if self.topicWord.shape[0]!=0:
      self.topicWord = numpy.vstack((self.topicWord,sample.topicWord))
    else:
      self.topicWord = sample.topicWord.copy()
    self.topicUse = numpy.hstack((self.topicUse,sample.topicUse))

    def mapCluster(c):
      c0 = c[0].copy()
      c0[:,0] += offset
      return (c0,c[1])
    
    self.cluster += map(mapCluster,sample.cluster)
    self.clusterUse = numpy.hstack((self.clusterUse,sample.clusterUse))


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
