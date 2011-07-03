# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



from collections import defaultdict
import cPickle as pickle
import bz2

import numpy
import numpy.random

from dec_tree import DecTree



class SWood:
  """A stochastic woodland implimentation (Usually called a random forest:-P). Nothing that fancy - does classification and calculates/provides the out-of-bag error estimate so you can tune the parameters."""
  def __init__(self, int_dm, real_dm, cat, tree_count = 128, option_count = 3, minimum_size = 1, weight = None, index = None, callback = None, compress = False):
    """Constructs and trains the stochastic wood - basically all its doing is constructing lots of trees, each with a different bootstrap sample of the input and calculating the out-of-bound error estimates. The parameters are as follows: int_dm & real_dm - the data matrices, one for discrete attributes and one for continuous; you can set one to None if there are none of that kind. cat - The category vector, aligned with the data matrices, where each category is represented by an integer. tree_count - The number of decision trees to create. option_count - The number of attributes to consider at each level of the decision trees - maps to the rand parameter of the DecTree class. minimum_size - Nodes in the trees do not suffer further splits once they are this size or smaller. weight - Optionally allows you to weight the trainning examples, aligned with data matrices. index - Using this you can optionally tell it which examples to use from the other matrices/vectors, and/or duplicate examples. callback - An optional function of the form (steps done,steps overall) used to report progress during construction. compress - if True trees are stored pickled and compressed, in a bid to make them consume less memory - this will obviously destroy classification performance unless multi_classify is used with suitably large blocks. Allows the algorithm to be run with larger quantities of data, but only use as a last resort."""
    
    # Generate weight/index vectors if not provided, and also put in a dummy callback if needed to avoid if statements...
    if weight==None: weight = numpy.ones(cat.shape[0], dtype=numpy.float32)
    if index==None: index = numpy.arange(cat.shape[0], dtype=numpy.int32)
    if callback==None: callback = lambda a, b: None

    # Create data structure to calculate the oob error rate...
    oob_success = numpy.zeros(cat.shape[0], dtype=numpy.float32)
    oob_total = numpy.zeros(cat.shape[0], dtype=numpy.int32)

    # Iterate and create all the trees...
    self.trees = []
    for itr in xrange(tree_count):
      callback(itr, tree_count)

      # Select the bootstrap sample...
      b_ind = numpy.random.randint(index.shape[0], size=index.shape[0])
      b_ind.sort() # Should improve cache coherance slightly.
      bootstrap = index[b_ind]

      # Train the classifier...
      dt = DecTree(int_dm, real_dm, cat, weight, bootstrap, option_count, minimum_size)
      if compress: self.trees.append(bz2.compress(pickle.dumps(dt)))
      else: self.trees.append(dt)

      # Get the indices of the oob set...
      oob_set = numpy.ones(index.shape[0], dtype=numpy.bool_)
      oob_set[b_ind] = False
      oob_set = index[oob_set]

      # Store the oob info...
      for ind in oob_set:
        dist = dt.classify(int_dm[ind,:], real_dm[ind,:])
        if cat[ind] in dist:
          oob_success[ind] += float(dist[cat[ind]]) / float(sum(dist.itervalues()))
        oob_total[ind] += 1

    # Combine the oob info to calculate the error rate, include being robust to a smaple never being a member of the oob set...
    oob_total[oob_total==0] = 1
    self.success = (oob_success[index] / oob_total[index]).mean()

    del callback # Should not need this, but apparently I do.


  def classify(self, int_vec, real_vec):
    """Classifies an example, given the discrete and continuous feature vectors. Returns a dictionary indexed by categories that goes to the probability of that category being assigned; categories can be excluded, implying they have a value of one, but the returned value is actually a default dict setup to return 0.0 when you request an unrecognised key. The probabilities will of course sum to 1."""
    ret = defaultdict(float)
    for dt in self.trees:
      if isinstance(dt,str): dt = pickle.loads(bz2.decompress(dt))
      dist = dt.classify(int_vec, real_vec)
      distTotal = float(sum(dist.itervalues()))

      for cat, weight in dist.iteritems():
        w = weight / distTotal
        ret[cat] += w

    tWeight = len(self.trees)
    for cat in ret.iterkeys():
      ret[cat] /= tWeight

    return ret

  def multi_classify(self, int_dm, real_dm, callback = None):
    """Identical to classify, except you give it a data matrix and it classifies each entry in turn, returning a list of distributions. Note that in the cass of using the compressed version of this class using this is essential to be computationally reasonable."""

    # Create the output...
    if int_dm!=None: size = int_dm.shape[0]
    else: size = real_dm.shape[0]
    ret = map(lambda _: defaultdict(float), xrange(size))

    # Iterate and sum in the effect of each tree in turn, for all samples...
    for i,dt in enumerate(self.trees):
      if callback: callback(i, len(self.trees))
      
      if isinstance(dt,str): dt = pickle.loads(bz2.decompress(dt))
      
      for ii, prob in enumerate(ret):
        dist = dt.classify(int_dm[ii,:] if int_dm!=None else None, real_dm[ii,:] if real_dm!=None else None)
        distTotal = float(sum(dist.itervalues()))

        for cat, weight in dist.iteritems():
          w = weight / distTotal
          prob[cat] += w

    # Normalise...
    tWeight = len(self.trees)
    for prob in ret:
      for cat in prob.iterkeys():
        prob[cat] /= tWeight

    # Return...
    return ret


  def oob_success(self):
    """Returns the success rate ([0.0,1.0], more is better.) for the tree that has been trained. Calculated using the out-of-bag techneque, and primarilly exists so you can run with multiple values of option_count to find the best parameter, or see the effect of tree_count."""
    return self.success

  def tree_list(self):
    """Returns a list of all the decision trees in the woodland. Note that when in compressed mode these will be strings of bzip2 compressed and pickled trees, which can be resurected using pickle.loads(bz2.decompress(<>))."""
    return self.trees
