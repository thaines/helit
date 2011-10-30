# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import numpy



class Document:
  """Representation of a document as used by the system. Consists of a list of words - each is referenced by a natural number and is associated with a count of how many of that particular word exist in the document."""
  def __init__(self, dic):
    """Constructs a document given a dictionary (Or equivalent) dic[ident] = count, where ident is the natural number that indicates which word and count is how many times that word exists in the document. Excluded entries are effectivly assumed to have a count of zero. Note that the solver will construct an array 0..{max word ident} and assume all words in that range exist, going so far as smoothing in words that are never actually seen."""
    
    # Create data store...
    self.words = numpy.empty((len(dic),2), dtype=numpy.uint)
    
    # Copy in the data...
    index = 0
    self.sampleCount = 0 # Total number of words is sometimes useful - stored to save computation.
    for key, value in dic.iteritems():
      self.words[index,0] = key
      self.words[index,1] = value
      self.sampleCount += value
      index += 1
    assert(index==self.words.shape[0])
    
    # Sorts the data - experiance shows this is not actually needed as iteritems kicks out integers sorted, but as that is not part of the spec (As I know it.) this can not be assumed, and so this step is required, incase it ever changes (Or indeed another type that pretends to be a dictionary is passed in.)...
    self.words = self.words[self.words[:,0].argsort(),:]
    
    # Ident for the document, stored in here for conveniance. Only assigned when the document is stuffed into a Corpus...
    self.ident = None


  def getDic(self):
    """Returns a dictionary object that represents the document, basically a recreated version of the dictionary handed in to the constructor."""
    ret = dict()
    for i in xrange(self.words.shape[0]): ret[self.words[i,0]] = self.words[i,1]
    return ret

  def getIdent(self):
    """Ident - just the offset into the array in the corpus where this document is stored, or None if its yet to be stored anywhere."""
    return self.ident


  def getSampleCount(self):
    """Returns the number of samples in the document, which is equivalent to the number of words, counting duplicates."""
    return self.sampleCount

  def getWordCount(self):
    """Returns the number of unique words in the document, i.e. not counting duplicates."""
    return self.words.shape[0]
    
  def getWord(self, index):
    """Given an index 0..getWordCount()-1 this returns the tuple (ident,count) for that word."""
    return (self.words[index,0],self.words[index,1])
