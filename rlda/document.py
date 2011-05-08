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



import math
import numpy

import rlda



class Document:
  """A document, consists of a list of all the word/identifier pairs in the document."""
  def __init__(self, dic):
    """Constructs a document given a dictionary dic[(identifier num, word num)] = count, where identifier num is the natural number that indicates which identifier, and word num the natural number which indicates which word. Count is how many times that identifier-word pair exist in the document. Excluded entries are effectivly assumed to have a count of zero."""

    # Create data store with columns (identifier,word,count)...
    self.words = numpy.empty((len(dic),3),dtype=numpy.uint)

    # Copy in the data...
    index = 0
    self.sampleCount = 0
    self.maxIdentNum = -1
    self.maxWordNum = -1
    for key, value in dic.iteritems():
      self.words[index,0] = key[0]
      self.words[index,1] = key[1]
      self.words[index,2] = value
      self.maxIdentNum = max((self.maxIdentNum,key[0]))
      self.maxWordNum = max((self.maxWordNum,key[1]))
      self.sampleCount += value
      index += 1

    # Sorts the data...
    self.words = self.words[self.words[:,0].argsort(),:]


    # Set the model variable to None, so it can be filled in later. It will ultimatly contain a numpy.array parametrising the multinomial distribution from which topics are drawn...
    self.model = None

    # Array indexed by regions of their negative log likelihood, plus average region size...
    self.nllRegion = None
    self.sizeRegion = None

    # Document number, stored in here for conveniance. Only assigned when the document is stuffed into a Corpus...
    self.num = None

  def getDic(self):
    """Returns a dictionary object that represents the document, basically a recreated version of the dictionary handed in to the constructor."""
    ret = dict()

    for i in xrange(self.words.shape[0]):
      ret[(self.words[i,0],self.words[i,1])] = self.words[i,2]

    return ret

  def getNum(self):
    """Number - just the offset into the array in the corpus where this document is stored, or None if its yet to be stored anywhere."""
    return self.num


  def getSampleCount(self):
    """Returns the number of identifier-word pairs in the document, counting duplicates."""
    return self.sampleCount

  def getMaxIdentNum(self):
    """Returns the largest ident number it has seen."""
    return self.maxIdentNum

  def getMaxWordNum(self):
    """Returns the largest word number it has seen."""
    return self.maxWordNum

  def getWords(self):
    """Returns an array of all the words in the document, row per word with the columns [identifier,word,count]."""
    return self.words


  def fit(self, ir, wrt, params = rlda.Params(), alpha = 1.0, norm = 100.0):
    """Given the model provided by a corpus (ir and wrt.) this fits the documents model, independent of the corpus itself. Uses Gibbs sampling as you would expect."""
    rlda.fitDoc(self, ir, wrt, alpha, params, norm)


  def getModel(self):
    """Returns the vector defining the multinomial from which topics are drawn, P(topic), if it has been calculated, or None if it hasn't."""
    return self.model

  def setModel(self, model):
    """Sets the model for the document. For internal use only really."""
    self.model = model

  def probTopic(self, topic):
    """Returns the probability of the document emitting the given topic, where topics are represented by their ident. Do not call if model not calculated."""
    assert(self.model!=None)
    return self.model[topic]


  def regionSize(self, region):
    """Returns the average size of the region, as sampled when sampling the region probabilities."""
    return self.sizeRegion[region]

  def regionSizeVec(self):
    """Returns a vector of the average size of each region, as sampled when sampling the region probabilities."""
    return self.sizeRegion


  def negLogLikeRegion(self, region):
    """Returns the negative log likelihood of the words being drawn in the region, sampled and calculated during a call of fit. Do not call if fit has not been run, noting that fitting an entire corpus does not count."""
    return self.nllRegion[region]

  def negLogLikeRegionVec(self):
    """Returns a vector of negative log likelihhods for each region in the document."""
    return self.nllRegion

  def negLogLikeRegionAlt(self, region, ir, wrt, sampleCount=64):
    """Returns the negative log likelihood of the given region, alternate calculation - designed to decide if a region is being normal or not. Assuming that the documents model and the given model are all correct, rather than averages of samples from a distribution. This is obviously incorrect, but gives a good enough approximation for most uses. Has to use sampling in part, hence the sampleCount parameter."""

    # Normalise ir to get P(r|i)...
    ir = ir / ir.sum(axis=0)
    
    # Get a multinomial distribution on the words by fixing the region, multiplying with the distribution on topics and then marginalising topics out. Just for kicks make it the negative log, to save on later calculation...
    lmn = (wrt[:,region,:] * self.model).sum(axis=1)
    lmn /= lmn.sum()
    lmn = -numpy.log(lmn)

    # Construct an array for quickly calculating normalising constants...
    logInt = numpy.log(numpy.arange(self.sampleCount+1))
    logInt[0] = 0.0
    for i in xrange(1,logInt.shape[0]): logInt[i] += logInt[i-1]

    # Iterate and collect samples - samples are needed to deal with the 'r' distribution on each identifier...
    samples = []
    for s in xrange(sampleCount):
      wordCount = numpy.zeros(wrt.shape[0],dtype=numpy.int_)

      # Draw a sample of identifiers that match the region - 1 if accepted, 0 if not...
      vim = (ir[:,region]>numpy.random.random(ir.shape[0])).astype(numpy.int_)
      
      # Draw samples using the region mapping...
      sample = 0.0
      for i in xrange(self.words.shape[0]): # Iterate all identifier/word/count sets...
        sample += vim[self.words[i,0]] * self.words[i,2] * lmn[self.words[i,1]]
        wordCount[self.words[i,1]] += vim[self.words[i,0]] * self.words[i,2]

      # Normalising constant (Tell me this isn't a slick way to write this...)...
      sample += logInt[wordCount.sum()]
      sample -= logInt[wordCount].sum()

      # Store
      samples.append(sample)

    # Average them in a stable way, return...
    offset = min(samples)
    res = 0.0
    for i,sample in enumerate(samples):
      prob = math.exp(offset-sample)
      res += (prob-res)/float(i+1)
    return offset - math.log(res)
