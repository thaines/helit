# Copyright 2014 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import bz2



try:
  from utils.make import make_mod
  import os.path

  make_mod('frf_c', os.path.dirname(__file__), ['philox.h', 'philox.c', 'data_matrix.h', 'data_matrix.c', 'summary.h', 'summary.c', 'information.h', 'information.c', 'learner.h', 'learner.c', 'index_set.h', 'index_set.c', 'tree.h', 'tree.c', 'frf_c.h', 'frf_c.c'], numpy=True)
except: pass


from frf_c import *



def save_forest(fn, forest):
  """Saves a forest - the actual Forest interface is fairly flexible, so this is just one way of doing it - streams the Forest header followed by each Tree into a bzip2 compressed file. Suggested extension is '.rf'. The code for this is in Python, and forms a good reference if you need to write your own i/o for this module."""
  f = bz2.BZ2File(fn, 'w')
  f.write(forest.save())
  
  for i in xrange(len(forest)):
    f.write(forest[i]) # The Tree objects returned by forest[i] have the memoryview interface, so write knows what to do!
  
  f.close()

  
  
def load_forest(fn):
  """Loads a forest that was previous saved using the save_forest function. The code for this is in Python, and forms a good reference if you need to write your own i/o for this module."""
  # Prepare...
  ret = Forest()
  f = bz2.BZ2File(fn, 'r')
  
  # The Forest header, which comes with a fixed size initial part and then a variable size second part...
  initial_head = f.read(Forest.initial_size())
  head_size = Forest.size_from_initial(initial_head)
  head = initial_head + f.read(head_size - len(initial_head))
  
  trees = ret.load(head)
  
  # Each tree in return...
  for _ in xrange(trees):
    header = f.read(Tree.head_size())
    size = Tree.size_from_head(header)
    
    tree = Tree(size)
    memoryview(tree)[:len(header)] = header
    memoryview(tree)[len(header):] = f.read(size - len(header))
    
    ret.append(tree)
  
  # Cleanup and return...
  f.close()
  return ret
