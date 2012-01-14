# -*- coding: utf-8 -*-

# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import math
import exceptions
import cPickle as pickle

import numpy



class KMeansShared:
  """Provides a standard interface for k-means, so that all implimentations provide the same one. key issue is that features can be provided in 3 ways - an array where each row is a feature, a list of feature vectors, a list of tuples where the last entry of each tuple is the feature vector. These are then passed through to return values, but the actual implimentation only has to deal with one interface."""
  def __init__(self):
    self.means = None # Matrix of mean points, where each row is the coordinate of the centre of a cluster.


  def parameters(self):
    """Returns how many parameters the currently fitted model has, used for model selection."""
    return self.means.shape[0]*self.means.shape[1]

  def clusterCount(self):
    """returns how many clusters it has been trained with, returns 0 if it has not been trainned."""
    if self.means!=None: return self.means.shape[0]
    else: return 0

  def getCentre(self, i):
    """Returns the centre of the indexed cluster."""
    return self.means[i,:]


  def doTrain(self, feats, clusters):
    """Should train the model given a data matrix, where each row is a feature."""
    raise exceptions.NotImplementedError()


  def doGetCluster(self, feats):
    """feats should arrive as a 2D array, where each row is a feature - should return an integer vector of which cluster each array is assigned to. Default implimentation uses brute force, which is typically fast enough assuming a reasonable number of clusters."""
    ret = numpy.empty(feats.shape[0],dtype=numpy.int_)

    for i in xrange(feats.shape[0]):
      ret[i] = ((self.means-feats[i,:])**2).sum(axis=1).argmin()

    return ret


  def train(self, feats, clusters, *listArgs, **dictArgs):
    """Given the features and number of clusters, plus any implimentation specific arguments, this trains the model. Features can be provided as a data matrix, as a list of feature vectors or as a list of tuples where the last entry of each tuple is a feature vector."""
    if isinstance(feats,numpy.ndarray):
      # Numpy array - just pass through...
      assert(len(feats.shape)==2)
      self.doTrain(feats,clusters,*listArgs,**dictArgs)
      del listArgs,dictArgs
    elif isinstance(feats,list):
      if isinstance(feats[0],numpy.ndarray):
        # List of vectors - glue them all together to create a data matrix and pass on through...
        data = numpy.vstack(feats)
        self.doTrain(data,clusters,*listArgs,**dictArgs)
        del listArgs,dictArgs
      elif isinstance(feats[0],tuple):
        # List of tuples where the last item should be a numpy.array as a feature vector...
        vecs = map(lambda x:x[-1],feats)
        data = numpy.vstack(vecs)
        self.doTrain(data,clusters,*listArgs,**dictArgs)
        del listArgs,dictArgs
      else:
        raise exceptions.TypeError('bad type for features - when given a list it must contain numpy.array vectors or tuples with the last element a vector')
    else:
      raise exceptions.TypeError('bad type for features - expects a numpy.array or a list')


  def getCluster(self, feats):
    """Converts the feature vectors of the input into integers, which represent the cluster which each feature is most likelly a member of. Output will be the same form as the input, but with the features converted to integers. In the case of a feature matrix it will become a vector of integers. For a list of feature vectors it will become a list of integers. For a list of tuples with the last element a feature vector it will return a list of tuples where the last element has been replaced with an integer."""
    if isinstance(feats,numpy.ndarray):
      # Numpy array - just pass through...
      assert(len(feats.shape)==2)
      return self.doGetCluster(feats)
    elif isinstance(feats,list):
      if isinstance(feats[0],numpy.ndarray):
        # List of vectors - glue them all together to create a data matrix and pass on through...
        data = numpy.vstack(feats)
        asVec = self.doGetCluster(data)
        return map(lambda i:asVec[i],xrange(asVec.shape[0]))
      elif isinstance(feats[0],tuple):
        # List of tuples where the last item should be a numpy.array as a feature vector...
        vecs = map(lambda x:x[-1],feats)
        data = numpy.vstack(vecs)
        asVec = self.doGetCluster(data)
        return map(lambda i:feats[i][:-1] + (asVec[i],),xrange(asVec.shape[0]))
      else:
        raise exceptions.TypeError('bad type for features - when given a list it must contain numpy.array vectors or tuples with the last element a vector')
    else:
      raise exceptions.TypeError('bad type for features - expects a numpy.array or a list')


  def doGetNLL(self, feats):
    """Takes feats as a 2D data matrix and calculates the negative log likelihood of those features comming from the model, noting that it assumes a symmetric Gaussian for each centre and calculates the standard deviation using the provided features."""
    
    # Record each features distance from its nearest cluster centre...
    distSqr = numpy.empty(feats.shape[0], dtype=numpy.float_)
    cat = numpy.empty(feats.shape[0], dtype=numpy.int_)

    for i in xrange(feats.shape[0]):
      distsSqr = ((self.means-feats[i,:].reshape((1,-1)))**2).sum(axis=1)
      cat[i] =  distsSqr.argmin()
      distSqr[i] = distsSqr.min()
    
    # Calculate the standard deviation to use for all clusters...
    var = distSqr.sum() / (feats.shape[0] - self.means.shape[0])
    
    # Sum up the log likelihood for all features...
    mult = -0.5/var
    norm = numpy.log(numpy.bincount(cat))
    norm -= numpy.log(feats.shape[0])
    norm -= 0.5 * (numpy.log(2.0*numpy.pi) + self.means.shape[1]*numpy.log(var))
    
    ll = (distSqr*mult).sum()
    ll += norm[cat].sum()
    
    # Return the negative ll...
    return -ll


  def getNLL(self, feats):
    """Given a set of features returns the negative log likelihood of the given features being generated by the model. Note that as the k-means model is not probabilistic this is technically wrong, but it hacks it by treating each cluster as a symmetric Gaussian with a standard deviation calculated using the provided features, with equal probability of selecting each cluster."""
    if isinstance(feats,numpy.ndarray):
      # Numpy array - just pass through...
      assert(len(feats.shape)==2)
      return self.doGetNLL(feats)
    elif isinstance(feats,list):
      if isinstance(feats[0],numpy.ndarray):
        # List of vectors - glue them all together to create a data matrix and pass on through...
        data = numpy.vstack(feats)
        return self.doGetNLL(data)
      elif isinstance(feats[0],tuple):
        # List of tuples where the last item should be a numpy.array as a feature vector...
        vecs = map(lambda x:x[-1],feats)
        data = numpy.vstack(vecs)
        return self.doGetNLL(data)
      else:
        raise exceptions.TypeError('bad type for features - when given a list it must contain numpy.array vectors or tuples with the last element a vector')
    else:
      raise exceptions.TypeError('bad type for features - expects a numpy.array or a list')


  def save(self, filename):
    """Saves the learned parameters to a file."""
    pickle.dump(self.means, open(filename,'wb'), pickle.HIGHEST_PROTOCOL)

  def load(self, filename):
    """Loads the learned parameters from a file."""
    self.means = pickle.load(open(filename,'rb'))

  def getData(self):
    """Returns the data contained within, so it can be serialised with other data. (You can of course serialise this class directly if you want, but the returned object is a numpy array, so less likely to be an issue for any program that loads it.)"""
    return self.means

  def setData(self,data):
    """Sets the data for the object, should be same form as returned from getData."""
    self.means = data
