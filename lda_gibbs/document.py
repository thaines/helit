# -*- coding: utf-8 -*-

# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import numpy
import scipy

import sys
if sys.modules.has_key('lda'):
  import lda
elif sys.modules.has_key('lda_nmp'):
  import lda_nmp as lda
else:
  raise Exception('This module is not meant to be imported directly - import lda/lda_nmp instead.')

import solve_shared



class Document:
  """Representation of a document used by the system. Consists of two parts: a) A list of words; each is referenced by a natural number and is associated with a count of how many of that particular word exist in the document. Stored in a matrix. b) The vector parameterising the multinomial distribution from which topics are drawn for the document, if this has been calculated."""
  def __init__(self, dic):
    """Constructs a document given a dictionary dic[ident] = count, where ident is the natural number that indicates which word and count is how many times that word exists in the document. Excluded entries are effectivly assumed to have a count of zero. Note that the solver will construct an array 0..{max word ident} and assume all words in that range exist, going so far as smoothing in words that are never actually seen."""
    
    # Create data store...
    self.words = numpy.empty((len(dic),2), dtype=numpy.uint)
    
    # Copy in the data...
    index = 0
    self.wordCount = 0 # Total number of words is sometimes useful - stored to save computation.
    for key,value in dic.iteritems():
      self.words[index,0] = key
      self.words[index,1] = value
      self.wordCount += value
      index += 1
    assert(index==self.words.shape[0])
    
    # Sorts the data - experiance shows this is not actually needed as iteritems kicks out integers sorted, but as that is not part of the spec (As I know it.) this can not be assumed, and so this step is required, incase it ever changes (Or indeed another type that pretends to be a dictionary is passed in.)...
    self.words = self.words[self.words[:,0].argsort(),:]
    
    # Set the model variable to None, so it can be filled in later. It will ultimatly contain a numpy.array parametrising the multinomial distribution from which topics are drawn...
    self.model = None
    
    # Ident for the document, stored in here for conveniance. Only assigned when the document is stuffed into a Corpus...
    self.ident = None

    # Topic index, for (semi/weakly-) supervised classification problems...
    self.topic = None


  def getDic(self):
    """Returns a dictionary object that represents the document, basically a recreated version of the dictionary handed in to the constructor."""
    ret = dict()

    for i in xrange(self.words.shape[0]):
      ret[self.words[i,0]] = self.words[i,1]

    return ret


  def getIdent(self):
    """Ident - just the offset into the array in the corpus where this document is stored, or None if its yet to be stored anywhere."""
    return self.ident


  def setTopic(self, topic = None):
    """Allows you to 'set the topic' for the document, which is by default not set. This simply results in an increase in the relevant entry of the prior dirichlet distribution, the size of which is decided by a parameter in the Corpus object. The purpose of this is to allow (semi/weakly-) supervised classification problems to be done, rather than just unsupervised. Defaults to None, which is no topic bias. This is of course not really setting - it is only a prior, and the algorithm could disagree with you. This is arguably an advantage, for if there are mistakes in your trainning set. Note that this is only used for trainning a complete topic model - for fitting a document to an existing model this is ignored. The input should be None to unset (The default) or an integer offset into the topic list."""
    self.topic = topic

  def getTopic(self):
    """Returns the pre-assigned topic, as in integer offset into the topic list, or None if not set."""
    return self.topic


  def dupWords(self):
    """Returns the number of words in the document, counting duplicates."""
    return self.wordCount

  def uniqueWords(self):
    """Returns the number of unique words in the document, i.e. not counting duplicates."""
    return self.words.shape[0]
    
  def getWord(self, word):
    """Given an index 0..uniqueWords()-1 this returns the tuple (ident,count) for that word."""
    return (self.words[word,0], self.words[word,1])


  def getModel(self):
    """Returns the vector defining the multinomial from which topics are drawn, P(topic), if it has been calculated, or None if it hasn't."""
    return self.model
  
  def setModel(self,model):
    self.model = model

  def probTopic(self,topic):
    """Returns the probability of the document emitting the given topic, where topics are represented by their ident. Do not call if model not calculated."""
    assert(self.model!=None)
    return self.model[topic]
  
  
  def fit(self,topicsWords,alpha = 1.0,params = solve_shared.Params()):
    """Calculates a model for this document given a topics-words array, alpha value and a Params object. Note that the topic-words array is technically a point approximation of what is really a prior over a multinomial distribution, so this is not technically correct, but it is good enough for most purposes."""
    
    # Call the fitDoc function provided by the solver system...
    lda.fitDoc(self,topicsWords,alpha,params)
  
  
  def negLogLikelihood(self,topicsWords):
    """Returns the negative log likelihood of the document given a topics-words array. (This document can be in the corpus that generated the list or not, just as long as it has a valid model. Can use fit if need be.) Ignores the priors given by alpha and beta - just the probability of the words given the topic multinomials and the documents multinomial. Note that it is assuming that the topics-words array and document model are both exactly right, rather than averages of samples taken from the distribution over these parameters, i.e. this is not corrrect, but is generally a good enough approximation."""

    # Normalise the rows of the topicsArray, to get P(word|topic)...
    tw = (topicsWords.T/topicsWords.sum(axis=1)).T
    
    # First create a multinomial distribution for the sample of words that we have...
    mn = (tw.T * self.model).sum(axis=1)
    mn /= mn.sum()
    
    # Now calculate the negative log likelihood of generating the sample given...
    ret = -(numpy.log(mn[:,self.words[:,0]])*self.words[:,1]).sum()
    
    # Need to factor in the normalising constant before returning...
    # (Have to be careful here - these numbers can get silly sized.)
    logInt = -numpy.log(numpy.arange(self.wordCount+1))
    logInt[0] = 0.0
    for i in xrange(1,logInt.shape[0]): logInt[i] += logInt[i-1]
    
    ret += logInt[-1]
    for i in xrange(self.words.shape[0]):
      ret -= logInt[self.words[i,1]]
    
    return ret
