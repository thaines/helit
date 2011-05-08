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



import numpy

import rlda



class Corpus:
  """Defines a corpus, i.e. the input to the rLDA algorithm. Consists of documents, identifiers and words, plus counts of how many regions and topics a fitted model should have. Has a method to fit a model, after which you can retrieve the models parameters."""
  def __init__(self, regions, topics):
    """Construct a corpus - you are required to provide the number of regions and topics to be used by the fitting model."""
    # Model size parameters...
    self.regions = regions
    self.topics = topics

    # Create the array of documents, plus support variables...
    self.docs = []
    self.sampleCount = 0 # Number of samples - summed from all documents.
    self.maxIdentNum = -1
    self.maxWordNum = -1

    # Parameters for the priors...
    self.alpha = 1.0
    self.beta = 1.0
    self.gamma = 1.0

    # Multinomial distribution indexed as [word,region,topic] - part of the fitted model, None when not fitted... (Not normalised.)
    self.wrt = None

    # Second part of model - multinomial distribution indexed as [identifier,region]... (Not normalised.)
    self.ir = None
    

  def setRegionTopicCounts(self, regions, topics):
    """Sets the number of regions and topics. Note that this will reset the model, so after doing this all the model variables will be None."""
    self.regions = regions
    self.topics = topics

    self.wrt = None
    self.ir = None

    # Remove models from documents...
    for doc in self.docs:
      doc.model = None

  def getRegionCount(self):
    """Returns the number of regions that will be used."""
    return self.regions

  def getTopicCount(self):
    """Returns the number of topics that will be used."""
    return self.topics


  def setAlpha(self, alpha):
    """Sets the alpha value - 1 is more often than not a good value, and is the default."""
    self.alpha = alpha

  def setBeta(self, beta):
    """Sets the beta value. Defaults to 1.0."""
    self.beta = beta

  def setGamma(self, gamma):
    """Sets the gamma value. Defaults to 1.0. One will note that it doesn't actually get used in the formulation, so in a slight abuse it is used in place of beta during the r-step - this provides a touch more control to the user."""
    self.gamma = gamma

  def getAlpha(self):
    """Returns the current alpha value."""
    return self.alpha

  def getBeta(self):
    """Returns the current beta value."""
    return self.beta

  def getGamma(self):
    """Returns the current gamma value."""
    return self.gamma


  def add(self, doc):
    """Adds a document to the corpus."""
    doc.num = len(self.docs)
    self.docs.append(doc)

    self.sampleCount += doc.getSampleCount()
    self.maxIdentNum = max((self.maxIdentNum,doc.getMaxIdentNum()))
    self.maxWordNum = max((self.maxWordNum,doc.getMaxWordNum()))

  def setIdentWordCounts(self, identCount, wordCount):
    """Because the system autodetects identifiers and words as being the range 0..max where max is the largest number seen it is possible for you to tightly pack words but to want to reserve some past the end. Its also possible for a data set to never contain the last entity, creating problems. This allows you to set the numbers, forcing the issue. Note that setting the number less than actually exist is a guaranteed crash, at a later time."""
    self.maxIdentNum = identCount-1
    self.maxWordNum = wordCount-1


  def getSampleCount(self):
    """Returns the number of identifier-word pairs in all the documents, counting duplicates."""
    return self.sampleCount

  def getMaxIdentNum(self):
    """Returns the largest ident number it has seen."""
    return self.maxIdentNum

  def getMaxWordNum(self):
    """Returns the largest word number it has seen."""
    return self.maxWordNum

  def documentList(self):
    """Returns a list of all documents."""
    return self.docs


  def fit(self, params = rlda.Params(), callback = None):
    """Fits a model to this Corpus."""
    rlda.fit(self, params, callback)


  def setModel(self, wrt, ir):
    """Sets the model, in terms of the wrt and ir count matrices. For internal use only really."""
    self.wrt = wrt
    self.ir = ir

  def getIR(self):
    """Returns an unnormalised multinomial distribution indexed by [identifier,region]"""
    return self.ir

  def getWRT(self):
    """Returns an unnormalised multinomial distribution indexed by [word,region,topic]"""
    return self.wrt
