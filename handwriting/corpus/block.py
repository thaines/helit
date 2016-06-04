# Copyright 2016 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import string
import re

import numpy



sentence_re = re.compile(r'\..', re.DOTALL)
new_line_re = re.compile(r'\n')
words_re = re.compile(r"[a-zA-Z']+[^a-zA-Z]", re.DOTALL)
letters_re = re.compile(r"[a-zA-Z]")
digits_re = re.compile(r"[0-9]")



class Block:
  """Defines a block of text that may be presented to the user, for them to write in their own handwritting. We have a database of these, a very large database."""
  __slots__ = ['text', 'attribution', 'sentences', 'lines', 'words', 'letters', 'digits']
  
  def __init__(self, text, attribution):
    self.text = text
    self.attribution = attribution
    
    self.sentences = 1 + len(sentence_re.findall(self.text))
    self.lines = 1 + len(new_line_re.findall(self.text))
    self.words = len(words_re.findall(self.text))
    self.letters = len(letters_re.findall(self.text))
    self.digits = len(digits_re.findall(self.text))
  
  
  def stats(self, out_counts, out_adj, adj_index = string.ascii_letters + ' '):
    """Given two input arrays this adds to them the statistics of the contained text. The first array is of length 256, and counts the instances of character codes. The second array is 2D, with ['a', 'b'] being the number of times a 'b' follows an 'a'. It is indexed by adj_index however, and character pairs that contain a character not included are not counted."""
    
    # Counts are relativly easy - convert and histogram...
    text_codes = numpy.fromstring(self.text.encode('utf8'), dtype=numpy.uint8)
    out_counts += numpy.bincount(text_codes, minlength=256)
    
    # Adjacencies require a little more sneakyness...
    # First convert the codes array into an index into the adj_index, with entrys that are not in it set to -1...
    adj_codes = numpy.fromstring(adj_index, dtype=numpy.uint8)
    
    cap = len(adj_index) * len(adj_index)
    conversion = numpy.empty(256, dtype=numpy.int64)
    conversion[:] = cap
    conversion[adj_codes] = numpy.arange(adj_codes.shape[0])
    
    text_codes = conversion[text_codes]
    
    # Now take adjacent pairs, and calculate the 1D index in out_adj matrix...
    pos = (text_codes[:-1] * len(adj_index)) + text_codes[1:]
    
    # Lose values that are too large - they are pairs we do not record...
    pos = pos[pos < cap]
    
    # Histogram and sum into the adjacency matrix...
    if pos.shape[0]>0:
      out_adj += numpy.bincount(pos, minlength=cap).reshape((len(adj_index),len(adj_index)))
