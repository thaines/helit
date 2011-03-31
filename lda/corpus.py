# -*- coding: utf-8 -*-

# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



from topic import *

import sys
if sys.modules.has_key('lda'):
  import lda
elif sys.modules.has_key('lda_nmp'):
  import lda_nmp as lda
else:
  raise Exception('This module is not meant to be imported directly - import lda/lda_nmp instead.')

import solve_shared



class Corpus:
  """Contains a set of Document-s and a set of Topic-s associated with those Document-s. Also stores the alpha and beta parameters associated with the model."""
  def __init__(self, topicCount):
    """Basic setup, only input is the number of topics. Chooses default values for alpha and beta which you can change later before fitting the model."""
    
    # Create the array of documents and support variables...
    self.docs = []
    self.totalWords = 0
    self.maxWordIdentifier = 0
    
    # Create the array of topics...
    self.topics = []
    for i in xrange(topicCount):
      self.topics.append(Topic(len(self.topics)))
    
    # And then store some parameters...
    self.alpha = 1.0
    self.alphaMult = 10.0
    self.beta = 1.0
    
    
  def setTopicCount(self,topicCount):
    """Sets the number of topics. Note that this will reset the model, so after doing this all the model variables will be None etc."""
    # Recreate topics...
    self.topics = []
    for i in xrange(topicCount):
      self.topics.append(Topic(len(self.topics)))
      
    # Remove models from docuemnts...
    for doc in self.docs:
      doc.model = None



  def setAlpha(self, alpha):
    """Sets the alpha value - 1 is more often than not a good value, and is the default."""
    self.alpha = alpha

  def setAlphaMult(self, alphaMult):
    """Sets a multiplier of the alpha parameter used when the topic of a document is given - for increasing the prior for a given entry - can be used for semi-supervised classification. Defaults to a factor of 10.0"""
    self.alphaMult = alphaMult
  
  def setBeta(self, beta):
    """The authors of the paper observe that this is effectivly a scale parameter - use a low value to get a fine grained division into topics, or a high value to get just a few topics. Defaults to 1.0, which is a good number for most situations."""
    self.beta = beta
  
  def getAlpha(self):
    """Returns the current alpha value."""
    return self.alpha

  def getAlphaMult(self):
    """Returns the current alpha multiplier."""
    return self.alphaMult
    
  def getBeta(self):
    """Returns the current beta value."""
    return self.beta
    
    
  def add(self, doc):
    """Adds a document to the corpus."""
    doc.ident = len(self.docs)
    self.docs.append(doc)
    
    maxDocIdent = int(doc.words[-1,0])
    if maxDocIdent>self.maxWordIdentifier:
      self.maxWordIdentifier = maxDocIdent
      
    self.totalWords += doc.dupWords()
  
  def setWordCount(self, wordCount):
    """Because the system autodetects words as being the identifiers 0..max where max is the largest identifier seen it is possible for you to tightly pack words but to want to reserve some past the end. Its also possible for a data set to never contain the last word, creating problems. This allows you to set the number of words, forcing the issue. Note that setting the number less than actually exist is a guaranteed crash, at a later time."""
    self.maxWordIdentifier = wordCount-1


  def fit(self, params = solve_shared.Params(), callback = None):
    """Fits a model to this Corpus. params is a Params object from solve-shared. callback if provided should take two numbers - the first is the number of iterations done, the second the number of iterations that need to be done; used to report progress. Note that it will probably not be called for every iteration for reasons of efficiency."""
    lda.fit(self, params, callback)


  def maxWordIdent(self):
    """Returns the maximum word ident currently in the system; note that unlike Topic-s and Document-s this can have gaps in as its user set. Only a crazy user would do that though as it affects the result due to the system presuming that the gap words exist."""
    return self.maxWordIdentifier
  
  def maxDocumentIdent(self):
    """Returns the highest ident; documents will then be found in the range {0..max ident}. Returns -1 if no documents exist."""
    return len(self.docs)-1
  
  def maxTopicIdent(self):
    """Returns the highest ident; topics will then be found in the range {0..max ident}. Returns -1 if no topics exist."""
    return len(self.topics)-1


  def wordCount(self):
    """Number of words as far as a fitter will be concerned; doesn't mean that they all actually exist however."""
    return self.maxWordIdentifier+1

  def documentCount(self):
    """Number of documents."""
    return len(self.docs)

  def topicCount(self):
    """Number of topics."""
    return len(self.topics)


  def getDocument(self, ident):
    """Returns the Document associated with the given ident."""
    return self.docs[ident]
    
  def getTopic(self, ident):
    """Returns the Topic associated with the given ident."""
    return self.topics[ident]
    
  
  def documentList(self):
    """Returns a list of all documents."""
    return self.docs

  def topicList(self):
    """Returns a list of all topics."""
    return self.topics


  def topicsWords(self):
    """Constructs and returns a topics X words array that represents the learned models key part. Simply an array topics X words of P(topic,word). This is the data best saved for analysing future data - you can use the numpy.save/.load functions. Note that you often want P(word|topic), which you can obtain by normalising the rows - (a.T/a.sum(axis=1)).T"""
    ret = numpy.empty((len(self.topics), self.maxWordIdentifier+1), dtype=numpy.float_)
    for topic in self.topics:
      ret[topic.getIdent(),:] = topic.getModel()
    return ret


  def totalWordCount(self):
    """Returns the total number of words used by all the Document-s - is used by the solver, but may be of interest to curious users."""
    return self.totalWords
