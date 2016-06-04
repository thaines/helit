# Copyright 2016 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import os.path
import string
import numpy

from ply2.ply2 import read
from block import Block



class Corpus:
  """Loads the corpus into memory on creation, and provides access to it. Constructs basic statistics whilst it is at it."""
  
  def __init__(self, adjStats = string.ascii_letters + ' ', noisy = False):
    # Load it...
    if noisy: print 'Reading in ply2 corpus file...'
    data = read(os.path.join(os.path.dirname(__file__),'corpus.ply2'))
    
    # Convert contents into a list of blocks...
    self.blocks = []
    
    t_arr = data['element']['document']['text']
    a_arr = data['element']['document']['attribution']
    
    if noisy:
      print 'Creating blocks...'
    
    for i in xrange(t_arr.shape[0]):
      b = Block(t_arr[i], a_arr[i])
      self.blocks.append(b)
      
    # Construct all statistics - this gets complicated for reasons of efficiency...
    if noisy:
      print 'Collecting statistics...'
    
    self.counts = numpy.zeros(256, dtype=numpy.int32)
    self.adj = numpy.zeros((len(adjStats),len(adjStats)), dtype=numpy.int32)
    self.adj_index = adjStats
    
    for b in self.blocks:
      b.stats(self.counts, self.adj, self.adj_index)

  
  def __getitem__(self, i):
    return self.blocks[i]
  
  def __iter__(self):
    for block in self.blocks:
      yield block
    
  def __len__(self):
    return len(self.blocks)
  
  
  def get_counts(self):
    """Returns an array, indexed by character code, of how many of each appear."""
    return self.counts
  
  def get_adj(self):
    """Returns the adjacency array, indexed by [first character, second character] as counts of then pairs in the string. Uses the indexing in get_adj_index()."""
    return self.adj
  
  def get_adj_index(self):
    """Returns a string where the position in the string indicates which character is represented by each dimension in the adjacency array."""
    return self.adj_index
