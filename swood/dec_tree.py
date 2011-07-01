# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import math
import numpy
import numpy.random
import scipy.stats.distributions



class DecTree:
  """A decision tree, uses id3 with the c4.5 extension for continuous attributes. Fairly basic - always grows fully and stores a distribution of children at every node so it can fallback for previously unseen attribute categories. Allows the trainning vectors to be weighted and can be pickled. An effort has been made to keep this small, due to the fact that its not unusual to have millions in memory."""
  __slots__ = ['dist_cat', 'dist_weight', 'leaf', 'discrete', 'index', 'children', 'threshold', 'low', 'high'] # One tends to make a lot of these - best to keep 'em small.
   
  def __init__(self, int_dm, real_dm, cat, weight = None, index = None, rand = None, minimum_size = 1):
    """Input consists of upto 5 arrays and one parameter. The first two parameters are data matrices, where each row contains the attributes for an example. Two are provided, one of numpy.int32 for the discrete features, another of numpy.float32 for the continuous features - one can be set to None to indicate none of that type, but obviously at least one has to be provided. The cat vector is then aligned with the data matrices and gives the category for each exemplar, as a numpy.int32. weight optionally provides a numpy.float32 vector that also aligns with the data matrices, and effectivly provides a continuous repeat count for each example, so some can be weighted as being more important. By default all items in the data matrices are used, however, instead an index vector can be provided that indexes the examples to be used by the tree - this not only allows a subset to be used but allows samples to be repeated if desired (This feature is actually used for building the tree recursivly, such that each DecTree object is in fact a node; also helps for creating a collection of trees with random trainning sets.). Finally, by default it considers all features at each level of the tree, however, if an integer rather than None is provided to the rand parameter it instead randomly selects a subset of attributes, of size rand, and then selects the best of this subset, with a new draw for each node in the tree. minimum_size gives a minimum number of samples in a node for it to be split - it needs more samples than this otherwise it will become a leaf. A simple tree prunning method; defaults to 1 which is in effect it disabled."""

    # If weight and/or index have not been provided create them - makes the code neater...
    if weight==None: weight = numpy.ones(cat.shape[0], dtype=numpy.float32)
    if index==None: index = numpy.arange(cat.shape[0], dtype=numpy.int32)

    # Collect the category statistics for this node - incase its a leaf or splits on discrete values and encounters a new value, or just for general curiosity...
    cats = numpy.unique(cat[index])
    self.dist_cat = numpy.empty(cats.shape[0], dtype=numpy.int32)
    self.dist_weight = numpy.empty(cats.shape[0], dtype=numpy.float32)
    
    for i,c in enumerate(cats):
      self.dist_cat[i] = c
      self.dist_weight[i] = weight[index[numpy.where(cat[index]==c)]].sum()

    # Decide if its worth subdividing this node or not (Might change our mind later)...
    if self.dist_cat.shape[0]<=1 or index.shape[0]<=minimum_size: self.leaf = True
    else:
      # Its subdivision time!..
      self.leaf = False

      # Select the set of attributes to consider splitting on...
      int_size = int_dm.shape[1] if int_dm!=None else 0
      real_size = real_dm.shape[1] if real_dm!=None else 0
      if rand==None: options = numpy.arange(int_size+real_size)
      else: options = numpy.random.permutation(int_size+real_size)[:rand]

      # We need to find the optimal split out of all options...
      ent = self.entropy()
      choice = (None, (ent,))
      for c in options:
        if c<int_size: ch = (c, self.__entropy_discrete(int_dm, c, cat, weight, index))
        else: ch = (c, self.__entropy_continuous(real_dm, c-int_size, cat, weight, index))
        if ch[1][0]<choice[1][0]: choice = ch

      # Check its worth doing the split - needs to be at least a tiny improvement...
      if choice[1][0]+1e-5>ent: self.leaf = True
      else:
        # Recurse to generate the children nodes, which happen to just be more DecTree objects - code depends on if its discrete or continuous...
        if choice[0]<int_size:
          self.discrete = True
          self.index = choice[0]
          self.children = []
          for category, c_index in choice[1][1].iteritems():
            self.children.append((category, DecTree(int_dm, real_dm, cat, weight, c_index, rand, minimum_size)))
          self.children = tuple(self.children)
        else:
          self.discrete = False
          self.index = choice[0] - int_size
          self.threshold = choice[1][1]
          self.low = DecTree(int_dm, real_dm, cat, weight, choice[1][2], rand, minimum_size)
          self.high = DecTree(int_dm, real_dm, cat, weight, choice[1][3], rand, minimum_size)


  def entropy(self):
    """Returns the entropy of the data that was used to train this node. Really an internal method, exposed in case of rampant curiosity. Note that it is in nats, not bits."""
    return scipy.stats.distributions.entropy(self.dist_weight)

  def __entropy_discrete(self, int_dm, column, cat, weight, index):
    """Internal method - works out the entropy after a discrete division. Also returns the indices for the children nodes as values in a dictionary indexed by the class they have for the relevant column. Returns a tuple (entropy, index dict.)"""
    
    # Generate the index dictionary...
    ind_dict = dict()
    values = numpy.unique(int_dm[index,column])
    for v in values: ind_dict[v] = index[numpy.where(int_dm[index,column]==v)]

    # Calculate the entropy - make use of the index dictionary...
    entropy = 0.0
    p_div = weight[index].sum()
    for c_index in ind_dict.itervalues():
      c_div = weight[c_index].sum()
      cats = numpy.unique(cat[c_index])
      dist = numpy.empty(cats.shape[0], dtype=numpy.float32)
      for i,c in enumerate(cats):
        dist[i] = weight[c_index[numpy.where(cat[c_index]==c)]].sum()
      entropy += c_div/p_div * scipy.stats.distributions.entropy(dist)
      
    # Return...
    return (entropy, ind_dict)

  def __entropy_continuous(self, real_dm, column, cat, weight, index):
    """Internal method - works out the entropy after a continuous division. Also returns the optimal split point and the indices for the children nodes - as a tuple (entropy, split point, low, high)."""

    # Generate a copy of the index sorted by the matching value in the selected real_dm column...
    s_index = index[numpy.argsort(real_dm[index,column])]

    # Generate a culumative array containing a sum of the weights for each exemplar less than the split point, which is defined as being between the index in the array and the next index, as aligned with the sorted weighted array...
    cats = numpy.unique(cat[index])
    cum = numpy.zeros((s_index.shape[0],cats.shape[0]), dtype=numpy.float32)
    for i,c in enumerate(cats):
      ind = numpy.where(cat[s_index]==c)
      cum[ind,i] += weight[s_index[ind]]
    cum = numpy.cumsum(cum, axis=0)

    # Calculate the entropy for each split point...
    entLow = scipy.stats.distributions.entropy(cum[:-1,:].T)
    entHigh = scipy.stats.distributions.entropy((numpy.reshape(cum[-1,:], (1,-1))-cum[:-1,:]).T)
    weight = cum[:-1,:].sum(axis=1) / cum[-1,:].sum()
    ent = weight*entLow + (1.0-weight)*entHigh

    # Select the lowest, return that split point and other relevant information...
    i = numpy.argmin(ent)
    split = 0.5*(real_dm[s_index[i],column] + real_dm[s_index[i+1],column])
    return (ent[i], split, s_index[:i+1], s_index[i+1:])


  def classify(self, int_vec, real_vec):
    """Given a pair of vectors, one for discrete attributes and another for continuous atributes this returns the trees estimated distribution for the exampler. This distribution will take the form of a dictionary, which you must not modify, that is indexed by categories and goes to a count of how many examples with that category were in that leaf node. 99% of the time only one category should exist, though various scenarios can result in there being more than 1."""
    if self.leaf: return self.prob()
    elif self.discrete:
      key = int_vec[self.index]
      for value, child in self.children:
        if value==key:
          return child.classify(int_vec, real_vec)
      return self.prob() # Previously unseen attribute - fallback and return this nodes distribution.
    else: # Its continuous.
      itsLow = real_vec[self.index]<self.threshold
      if itsLow: return self.low.classify(int_vec, real_vec)
      else: return self.high.classify(int_vec, real_vec)


  def prob(self):
    """Returns the distribution over the categories of the trainning examples that went through this node - if this is a leaf its likelly to be non-zero for just one category. Represented as a dictionary category -> weight that only includes entrys if they are not 0. weights are the sum of the weights for the input, and are not normalised."""
    ret = dict()
    for i in xrange(self.dist_cat.shape[0]): ret[self.dist_cat[i]] = self.dist_weight[i]
    return ret

  def size(self):
    """Returns how many nodes make up the tree."""
    if self.leaf: return 1
    elif self.discrete: return 1 + sum(map(lambda c: c[1].size(), self.children))
    else: return 1 + self.low.size() + self.high.size()

  def isLeaf(self):
    """Returns True if it is a leaf node, False otherwise."""
    return self.leaf

  def isDiscrete(self):
    """Returns True if it makes its decision based on a discrete attribute, False if it is continuous or a leaf."""
    if self.leaf: return False
    return self.discrete

  def isContinuous(self):
    """Returns True if it makes a decision by splitting a continuous node, False if its is either discrete or a leaf."""
    if self.leaf: return False
    return not self.discrete

  def getIndex(self):
    """Returns the index of either the discrete column or continuous column which it decides on, or None if it is a leaf."""
    if self.leaf: return None
    return self.index

  def getChildren(self):
    """Returns a dictionary of children nodes indexed by the attribute the decision is being made on if it makes a discrete decision, otherwise None. Note that any unseen attribute value will not be included."""
    if not self.leaf and self.discrete:
      ret = dict()
      for value, child in self.children: ret[value] = child
      return ret
    else: return None

  def getThreshold(self):
    """If it is a continuous node this returns the threshold between going down the low and high branches of the decision tree, otherwise returns None."""
    if not self.leaf and not self.discrete: return self.threshold
    else: return None

  def getLow(self):
    """If it is a continuous decision node this returns the branch down which samples with the attribute less than the threshold go to; otherwise None."""
    if not self.leaf and not self.discrete: return self.low
    else: return None

  def getHigh(self):
    """If it is a continuous decision node this returns the branch down which samples with the attribute higher than or equal to the threshold go to; otherwise None."""
    if not self.leaf and not self.discrete: return self.high
    else: return None
