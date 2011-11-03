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



import solvers
from solve_shared import Params, State
from document import Document
from dp_conc import PriorConcDP



class Corpus:
  """Contains a set of Document-s, plus parameters for the graphical models priors - everything required as input to build a model, except a Params object."""
  def __init__(self, other = None):
    """Basic setup, sets a whole bunch of stuff to sensible parameters, or a copy constructor if provided with another Corpus."""
    if other!=None:
      # Create the array of documents and support variables...
      self.docs = map(lambda d: Document(d), other.docs)
      self.sampleCount = other.sampleCount
      self.wordCount = other.wordCount

      # Behavoural flags/parameters...
      self.dnrDocInsts = other.dnrDocInsts
      self.dnrCluInsts = other.dnrCluInsts
      self.seperateClusterConc = other.seperateClusterConc
      self.seperateDocumentConc = other.seperateDocumentConc
      self.oneCluster = other.oneCluster
      self.calcBeta = other.calcBeta
      self.calcCluBmn = other.calcCluBmn
      self.calcPhi = other.calcPhi
      self.resampleConcs = other.resampleConcs
      self.behSamples = other.behSamples

      self.alpha = PriorConcDP(other.alpha)
      self.beta = other.beta
      self.gamma = PriorConcDP(other.gamma)
      self.rho = PriorConcDP(other.rho)
      self.mu = PriorConcDP(other.mu)
      self.phiConc = other.phiConc
      self.phiRatio = other.phiRatio

      self.abnorms = dict(other.abnorms)
    else:
      # Create the array of documents and support variables...
      self.docs = []
      self.sampleCount = 0 # How many samples exist in all the documents.
      self.wordCount = 0 # How many types of words exist.

      # Behavoural flags/parameters...
      self.dnrDocInsts = False
      self.dnrCluInsts = True
      self.seperateClusterConc = False
      self.seperateDocumentConc = False
      self.oneCluster = False
      self.calcBeta = True
      self.calcCluBmn = True
      self.calcPhi = True
      self.resampleConcs = True
      self.behSamples = 1024

      # Parameters for the priors in the graphical model...
      self.alpha = PriorConcDP() # Document instance DP
      self.beta = 1.0 # Topic multinomial symmetric Dirichlet distribution prior.
      self.gamma = PriorConcDP() # Topic DP
      self.rho = PriorConcDP() # Cluster instance DP
      self.mu = PriorConcDP() # Cluster creating DP
      self.phiConc = 2.0 # For the prior on behaviour multinomials. (conc is multiplied by the number of entrys.)
      self.phiRatio = 10.0 # "

      # Abnormalities - a dictionary taking each abnormality to a natural number which is its index in the various arrays...
      self.abnorms = dict()


  def setDocInstsDNR(self, val):
    """False to resample the document instances, True to not. Defaults to False, but can be set True to save a bit of computation. Not recomended to be changed, as convergance is poor without it."""
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
    """True if each cluster has its own seperate concentration parameter, False if they are shared."""
    return self.seperateClusterConc

  def setSeperateDocumentConc(self, val):
    """True if you want each document to have its own concentration value, False if you want a single value shared between all documents. Experiance shows that the default, False, is the only sensible option most of the time, though when single cluster is on True can give advantages."""
    self.seperateDocumentConc = val

  def getSeperateDocumentConc(self):
    """True if each document has its own concetration parameter, False if they all share a single concentration parameter."""
    return self.seperateDocumentConc

  def setOneCluster(self, val):
    """Leave as False to keep the default cluster behaviour, but set to True to only have a single cluster - this results in a HDP implimentation that has an extra pointless layer, making a it a bit inefficient, but not really affecting the results relative to a normal HDP implimentation."""
    self.oneCluster = val

  def getOneCluster(self):
    """Returns False for normal behaviour, True if only one cluster will be used - this forces the algorithm to be normal HDP, with an excess level, rather than dual HDP."""
    return self.oneCluster

  def setCalcBeta(self, val):
    """Set False to have beta constant as the algorithm runs, leave as True if you want it recalculated based on the topic multinomials drawn from it."""
    self.calcBeta = val

  def getCalcBeta(self):
    """Returns False to leave the beta prior on topic word multinomials as is, True to indicate that it should be optimised"""
    return self.calcBeta

  def setCalcClusterBMN(self, val):
    """Sets if the per-cluster behaviour multinomial should be resampled."""
    self.calcCluBmn = val

  def getCalcClusterBMN(self):
    """Returns True if it is going to recalculate the per-cluster behaviour distribution, False otherwise."""
    return self.calcCluBmn

  def setCalcPhi(self, val):
    """Set False to have phi constant as the algorithm runs, leave True if you want it recalculated based on the cluster multinomials over behaviour drawn from it."""
    self.calcPhi = val

  def getCalcPhi(self):
    """Returns False if it is going to leave the phi prior as is, True to indicate that it will be optimised."""
    return self.calcPhi

  def setResampleConcs(self, val):
    """Sets True, the default, to resample concentration parameters, False to not."""
    self.resampleConcs = val
    
  def getResampleConcs(self):
    """Returns True if it will be resampling the concentration parameters, False otherwise."""
    return self.resampleConcs
    
  def setBehSamples(self, samples):
    """Sets the number of samples to use when integrating the prior over each per-cluster behaviour multinomial."""
    self.behSamples = samples

  def getBehSamples(self):
    """Returns the number of samples to be used  by the behaviour multinomial estimator. Defaults to 1024."""
    return self.behSamples


  def setAlpha(self, alpha, beta, conc):
    """Sets the concentration details for the per-document DP from which the topics for words are drawn."""
    self.alpha.alpha = alpha
    self.alpha.beta  = beta
    self.alpha.conc  = conc

  def getAlpha(self):
    """Returns the PriorConcDP for the alpha parameter."""
    return self.alpha

  def setBeta(self, beta):
    """Parameter for the symmetric Dirichlet prior on the multinomial distribution from which words are drawn, one for each topic."""
    assert(beta>=0.0)
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

  def setPhi(self, conc, ratio):
    """Sets the weight and ratio for Phi, which is a Dirichlet distribution prior on the multinomial over which behaviour each word belongs to, as stored on a per-cluster basis. conc is the concentration for the distribution, whilst ratio is how many times more likelly normal behaviour is presumed to be than any given abnormal behaviour."""
    self.phiConc  = conc
    self.phiRatio = ratio

  def getPhiConc(self):
    """Returns the concentration parameter for the phi prior. Defaults to 1."""
    return self.phiConc
    
  def getPhiRatio(self):
    """Returns the current phi ratio, which is the ratio of how many times more likelly normal words are than any given abnormal class of words in the prior. Defaults to 10."""
    return self.phiRatio


  def add(self, doc, igIdent = False):
    """Adds a document to the corpus."""
    if igIdent==False: doc.ident = len(self.docs)
    self.docs.append(doc)

    self.sampleCount += doc.getSampleCount()
    self.wordCount = max((self.wordCount, doc.words[-1,0]+1))

    for abnorm in doc.getAbnorms():
      if abnorm not in self.abnorms:
        num = len(self.abnorms)+1 # +1 to account for normal behaviour being entry 0.
        self.abnorms[abnorm] = num

  def getDocumentCount(self):
    """Number of documents."""
    return len(self.docs)

  def getDocument(self,ident):
    """Returns the Document associated with the given ident."""
    return self.docs[ident]

  def documentList(self):
    """Returns a list of all documents."""
    return self.docs

  def getAbnormDict(self):
    """Returns a dictionary indexed by the abnormalities seen in all the documents added so far. The values of the dictionary are unique natural numbers, starting from 1, which index the abnormalities in the arrays used internally for simulation."""
    return self.abnorms


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
    state = State(self,params)
    if mp and params.runs>1 and hasattr(solvers,'gibbs_all_mp'):
      solvers.gibbs_all_mp(state, callback)
    else:
      solvers.gibbs_all(state, callback)
    return state.getModel()
