# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import math
import numpy
import scipy.special
import collections

import solvers



class DocSample:
  """Stores the sample information for a given document - the DP from which topics are drawn and which cluster it is a member of. Also calculates and stores the negative log liklihood of the document."""
  def __init__(self, doc):
    """Given the specific DocState object this copies the relevant information. Note that it doesn't calculate the nll - a method does that. Also supports cloning."""
    if isinstance(doc, DocSample): # Code for clonning
      self.cluster = doc.cluster
      self.dp = doc.dp.copy()
      self.conc = doc.conc
      self.nll = doc.nll
    else:
      # Extract the model information...
      self.cluster = doc.cluster
      self.dp = doc.use.copy()
      self.conc = doc.conc
      self.nll = 0.0

  def calcNLL(self, doc, state):
    """Calculates the negative log likelihood of the document, given the relevant information. This is the DocState object again (It needs the samples, which are not copied into this object by the constructor.), but this time with the entire state object as well. Probability (Expressed as negative log likelihood.) is specificly calculated using all terms that contain a variable in the document, but none that would be identical for all documents. That is, it contains the probability of the cluster, the probability of the DP given the cluster, and the probability of the samples, which factor in both the drawing of the topic and the drawing of the word. The ordering of the samples is considered irrelevant, with both the topic and word defining uniqueness. Some subtle approximation is made - see if you can spot it in the code!"""
    self.nll = 0.0

    # Probability of drawing the cluster...
    self.nll -= math.log(state.clusterUse[doc.cluster])
    self.nll += math.log(state.clusterUse.sum()+state.clusterConc)


    # Probability of drawing the documents dp from its cluster...
    cl = state.cluster[doc.cluster]
    instCounts = numpy.zeros(cl[0].shape[0], dtype=numpy.int32)
    for ii in xrange(doc.use.shape[0]):
      instCounts[doc.use[ii,0]] += 1

    norm = cl[0][:,1].sum() + cl[1]
    self.nll -= (numpy.log(numpy.asfarray(cl[0][:,1])/norm)*instCounts).sum()
    self.nll -= scipy.special.gammaln(instCounts.sum() + 1.0)
    self.nll += scipy.special.gammaln(instCounts + 1.0).sum()


    # Count the numbers of word/topic instance pairs in the data structure - sum using a dictionary...
    samp_count = collections.defaultdict(int) # [instance,word]
    for s in xrange(doc.samples.shape[0]):
      samp_count[doc.samples[s,0],doc.samples[s,1]] += 1

    # Calculate the probability distribution of drawing each topic instance and the probability of drawing each word/topic assignment...
    inst = numpy.asfarray(doc.use[:,1])
    inst /= inst.sum() + doc.conc
    
    topicWord = numpy.asfarray(state.topicWord) + state.beta
    topicWord = (topicWord.T/topicWord.sum(axis=1)).T

    instLog = numpy.log(inst)
    wordLog = numpy.log(topicWord)

    # Now sum into nll the probability of drawing the samples that have been drawn - gets a tad complex as includes the probability of drawing the topic from the documents dp and then the probability of drawing the word from the topic, except I've merged them such that it doesn't look like that is what is happening...
    self.nll -= scipy.special.gammaln(doc.samples.shape[0]+1.0)
    for pair, count in samp_count.iteritems():
      inst, word = pair
      topic = cl[0][doc.use[inst,0],0]
      self.nll -= count * (wordLog[topic,word] + instLog[inst])
      self.nll += scipy.special.gammaln(count+1.0)


  def getCluster(self):
    """Returns the sampled cluster assignment."""
    return self.cluster

  def getInstCount(self):
    """Returns the number of cluster instances in the documents model."""
    return self.dp.shape[0]

  def getInstTopic(self, i):
    """Returns the topic index for the given instance."""
    return self.dp[i,0]

  def getInstWeight(self, i):
    """Returns the number of samples that have been assigned to the given topic instance."""
    return self.dp[i,1]

  def getInstDual(self):
    """Returns a 2D numpy array of integers where the first dimension indexes the topic instances for the document and the the second dimension has two entrys, the first (0) the topic index, the second (1) the number of samples assigned to the given topic instance. Do not edit the return value for this method - copy it first."""
    return self.dp

  def getInstConc(self):
    """Returns the sampled concentration parameter, as used by the document DP."""
    return self.conc

  def getNLL(self):
    """Returns the negative log liklihood of the document given the model, if it has been calculated."""
    return self.nll



class Sample:
  """Stores a single sample drawn from the model - the topics, clusters and each document being sampled over. Stores counts and parameters required to make them into distributions, rather than final distributions. Has clonning capability."""
  def __init__(self, state, calcNLL = True):
    """Given a state this draws a sample from it, as a specific parametrisation of the model."""
    if isinstance(state, Sample): # Code for clonning
      self.alpha = state.alpha
      self.beta = state.beta.copy()
      self.gamma = state.gamma
      self.rho = state.rho
      self.mu = state.mu    

      self.topicWord = state.topicWord.copy()
      self.topicUse = state.topicUse.copy()
      self.topicConc = state.topicConc

      # Cluster stuff...
      self.cluster = map(lambda t: (t[0].copy(),t[1]), state.cluster)
      self.clusterUse = state.clusterUse.copy()
      self.clusterConc = state.clusterConc
      
      self.doc = map(lambda ds: DocSample(ds), state.doc)
    else:
      self.alpha = state.alpha
      self.beta = state.beta.copy()
      self.gamma = state.gamma
      self.rho = state.rho
      self.mu = state.mu

      # Topic stuff...
      self.topicWord = state.topicWord.copy()
      self.topicUse = state.topicUse.copy()
      self.topicConc = state.topicConc

      # Cluster stuff...
      self.cluster = map(lambda t: (t[0].copy(),t[1]), state.cluster)
      self.clusterUse = state.clusterUse.copy()
      self.clusterConc = state.clusterConc

      # The details for each document...
      self.doc = []
      for d in xrange(len(state.doc)):
        self.doc.append(DocSample(state.doc[d]))

      # Second pass through documents to fill in the negative log liklihoods...
      if calcNLL:
        for d in xrange(len(state.doc)):
          self.doc[d].calcNLL(state.doc[d],state)


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
    """Returns the PriorConcDP that was used for the mu parameter, which is the concentration parameter for the Dp from which clusters are drawn."""
    return self.mu


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
    """Returns the number of samples assigned to each word for all topics, indexed [topic,word], as an integer numpy array. Do not edit the return value - make a copy first."""
    return self.topicWord

  def getTopicMultinomial(self, t):
    """Returns the calculated multinomial for a given topic ident."""
    ret = self.beta.copy()
    ret += self.topicWord[t,:]
    ret /= ret.sum()
    return ret

  def getTopicMultinomials(self):
    """Returns the multinomials for all topics, in a single array - indexed by [topic,word] to give P(word|topic)."""
    ret = numpy.vstack([self.beta]*self.topicWord.shape[0])
    ret += self.topicWord
    ret = (ret.T / ret.sum(axis=1)).T
    return ret


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
    """Returns a 2D array, where the first dimension is indexed by the topic instance, and  the second contains two columns - the first the topic index, the second the weight. Do not edit the return value - copy before use."""
    return self.cluster[c][0]

  def getClusterInstConc(self, c):
    """Returns the sampled concentration that goes with the DP from which members of each documents DP are drawn."""
    return self.cluster[c][1]


  def docCount(self):
    """Returns the number of documents stored within. Should be the same as the corpus from which the sample was drawn."""
    return len(self.doc)

  def getDoc(self,d):
    """Given a document index this returns the appropriate DocSample object. These indices should align up with the document indices in the Corpus from which this Sample was drawn."""
    return self.doc[d]


  def nllAllDocs(self):
    """Returns the negative log likelihood of all the documents in the sample - a reasonable value to compare various samples with."""
    return sum(map(lambda d: d.getNLL(),self.doc))



class Model:
  """Simply contains a list of samples taken from the state during Gibbs iterations. Has clonning capability."""
  def __init__(self, obj=None):
    self.sample = []
    if isinstance(obj, Model):
      for sample in obj.sample:
        self.sample.append(Sample(sample))

  def sampleState(self, state):
    """Samples the state, storing the sampled model within."""
    self.sample.append(Sample(state))

  def absorbModel(self, model):
    """Given another model this absorbs all its samples, leaving the given model baren."""
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

  def bestSampleOnly(self):
    """Calculates the document nll for each sample and prunes all but the one with the highest - very simple way of 'merging' multiple samples together."""
    score = map(lambda s: s.nllAllDocs(),self.sample)
    best = 0
    for i in xrange(1,len(self.sample)):
      if score[i]>score[best]:
        best = i

    self.sample = [self.sample[best]]


  def fitDoc(self, doc, params = None, callback=None, mp = True):
    """Given a document this returns a DocModel calculated by Gibbs sampling the document with the samples in the model as priors. Returns a DocModel. Note that it samples using params for *each* sample in the Model, so you typically want to use less than the defaults in Params, typically only a single run and sample, which is the default. mp can be set to False to force it to avoid multi-processing behaviour."""
    if mp and len(self.sample)>1 and hasattr(solvers,'gibbs_doc_mp'):
      return solvers.gibbs_doc_mp(self, doc, params, callback)
    else:
      return solvers.gibbs_doc(self, doc, params, callback)



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

  def getNLL(self):
    """Returns the average nll of all the contained samples - does a proper mean of the probability of the samples."""
    minSam = min(map(lambda s:s.getNLL(),self.sample))
    probMean = sum(map(lambda s:math.exp(minSam-s.getNLL()),self.sample))
    probMean /= float(len(self.sample))
    return minSam - math.log(probMean)
