# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import solvers
from solve_shared import Params, State
from dp_conc import PriorConcDP



class Corpus:
  """Contains a set of Document-s, plus parameters for the graphical models priors - everything required as input to build a model, except a Params object."""
  def __init__(self):
    """Basic setup - sets a whole bunch of stuff to sensible defaults."""
    
    # Create the array of documents and support variables...
    self.docs = []
    self.sampleCount = 0 # How many samples exist in all the documents.
    self.wordCount = 0 # How many types of words exist.

    # Behavoural flags...
    self.dnrDocInsts = False
    self.dnrCluInsts = False
    self.seperateClusterConc = False
    self.seperateDocumentConc = False
    self.oneCluster = False
    self.calcBeta = False

    # Parameters for the priors in the graphical model...
    self.alpha = PriorConcDP() # Document instance DP
    self.beta = 1.0 # Topic multinomial symmetric Dirichlet distribution prior.
    self.gamma = PriorConcDP() # Topic DP
    self.rho = PriorConcDP() # Cluster instance DP
    self.mu = PriorConcDP() # Cluster creating DP


  def setDocInstsDNR(self, val):
    """False to resample the document instances, True to not. Defaults to False, but can be set True to save a bit of computation. Not recomended to be changed as convergance is poor without it."""
    self.dnrDocInsts = val

  def getDocInstsDNR(self):
    """Returns False if the document instances are going to be resampled, True if they are not."""
    return self.dnrDocInsts

  def setCluInstsDNR(self, val):
    """False to resample the cluster instances, True to not. Defaults to False, but can be set True to save quite a bit of computation. Its debatable if switching this to True causes the results to degrade in any way, but left on by default as indicated in the paper."""
    self.dnrCluInsts = val

  def getCluInstsDNR(self):
    """Returns False if the cluster instances are going to be resampled, True if they are not."""
    return self.dnrCluInsts

  def setSeperateClusterConc(self, val):
    """True if you want clusters to each have their own concentration parameter, False, the default, if you want a single concentration parameter shared between all clusters. Note that setting this True doesn't really work in my experiance."""
    self.seperateClusterConc = val

  def getSeperateClusterConc(self):
    """True if each cluster has its own seperate concentration parameter, false if they are shared."""
    return self.seperateClusterConc

  def setSeperateDocumentConc(self, val):
    """True if you want each document to have its own concentration value, False if you want a single value shared between all documents. Experiance shows that the default, False, is the only sensible option most of the time, though when single cluster is on True can give advantages."""
    self.seperateDocumentConc = val

  def getSeperateDocumentConc(self):
    """True if each document has its own concentration parameter, False if they all share a single concentration parameter."""
    return self.seperateDocumentConc

  def setOneCluster(self, val):
    """Leave as False to keep the default cluster behaviour, but set to True to only have a single cluster - this results in a HDP implimentation that has an extra pointless layer, making a it a bit inefficient, but not really affecting the results relative to a normal HDP implimentation."""
    self.oneCluster = val

  def getOneCluster(self):
    """Returns False for normal behaviour, True if only one cluster will be used - this forces the algorithm to be normal HDP, with an excess level, rather than dual HDP."""
    return self.oneCluster

  def setCalcBeta(self, val):
    """Leave as False to have beta constant as the algorithm runs, True if you want it recalculated based on the topic multinomials drawn from it. Defaults to False."""
    self.calcBeta = val

  def getCalcBeta(self):
    """Returns False to leave the beta prior on topic word multinomials as is, True to indicate that it should be optimised"""
    return self.calcBeta


  def setAlpha(self, alpha, beta, conc):
    """Sets the concentration details for the per-document DP from which the topics for words are drawn."""
    self.alpha.alpha = alpha
    self.alpha.beta  = beta
    self.alpha.conc  = conc

  def getAlpha(self):
    """Returns the PriorConcDP for the alpha parameter."""
    return self.alpha

  def setBeta(self, beta):
    """Parameter for the symmetric Dirichlet prior on the multinomial distribution from which words are drawn, one for each topic. (Symmetric therefore a single float as input.)"""
    assert(beta>0.0)
    self.beta = beta

  def getBeta(self):
    """Returns the current beta value. Defaults to 1.0."""
    return self.beta
    
  def setGamma(self, alpha, beta, conc):
    """Sets the concentration details for the topic DP, from which topics are drawn"""
    self.gamma.alpha = alpha
    self.gamma.beta  = beta
    self.gamma.conc  = conc

  def getGamma(self):
    """Returns the PriorConcDP for the gamma parameter."""
    return self.gamma

  def setRho(self, alpha, beta, conc):
    """Sets the concentration details used for each cluster instance."""
    self.rho.alpha = alpha
    self.rho.beta  = beta
    self.rho.conc  = conc

  def getRho(self):
    """Returns the PriorConcDP for the rho parameter."""
    return self.rho

  def setMu(self, alpha, beta, conc):
    """Sets the concentration details used for the DP from which clusters are drawn for documents."""
    self.mu.alpha = alpha
    self.mu.beta  = beta
    self.mu.conc  = conc

  def getMu(self):
    """Returns the PriorConcDP for the mu parameter."""
    return self.mu


  def add(self, doc):
    """Adds a document to the corpus."""
    doc.ident = len(self.docs)
    self.docs.append(doc)

    self.sampleCount += doc.getSampleCount()
    self.wordCount = max((self.wordCount, doc.words[-1,0]+1))

  def getDocumentCount(self):
    """Number of documents."""
    return len(self.docs)

  def getDocument(self, ident):
    """Returns the Document associated with the given ident."""
    return self.docs[ident]

  def documentList(self):
    """Returns a list of all documents."""
    return self.docs


  def setWordCount(self, wordCount):
    """Because the system autodetects words as being the identifiers 0..max where max is the largest identifier seen it is possible for you to tightly pack words but to want to reserve some past the end. Its also possible for a data set to never contain the last word, creating problems. This allows you to set the number of words, forcing the issue. Note that setting the number less than actually exists will be ignored."""
    self.wordCount = max((self.wordCount, wordCount))

  def getWordCount(self):
    """Number of words as far as a fitter will be concerned; doesn't mean that they have all actually been sampled within documents however."""
    return self.wordCount


  def getSampleCount(self):
    """Returns the number of samples stored in all the documents contained within."""
    return self.sampleCount


  def sampleModel(self, params=None, callback=None, mp=True):
    """Given parameters to run the Gibbs sampling with this does the sampling, and returns the resulting Model object. If params is not provided it uses the default. callback can be a function to report progress, and mp can be set to False if you don't want to make use of multiprocessing."""
    if params==None: params = Params()
    state = State(self, params)
    if mp and params.runs>1 and hasattr(solvers,'gibbs_all_mp'):
      solvers.gibbs_all_mp(state, callback)
    else:
      solvers.gibbs_all(state, callback)
    return state.getModel()
