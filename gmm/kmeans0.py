# -*- coding: utf-8 -*-

# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import math
import random

import numpy
import scipy.cluster.vq

from kmeans_shared import KMeansShared



class KMeans0(KMeansShared):
  """Wraps the kmeans implimentation provided by scipy with the same interface as provided by the other kmeans implimentations in the system. My experiance shows that this is insanely slow - not sure why, but even my brute force C implimentation is faster (Best guess is its coded in python, and not optimised in any way.)."""

  def doTrain(self, feats, clusters, maxIters = 512, restarts = 8, assignOut = None):
    """Given a large number of features as a data matrix this finds a good set of cluster centres. clusters is the number of clusters to create, maxIters is the maximum number of iterations for convergance of a cluster and restarts how many times to run - the most probable result is used. minSize is the smallest cluster size - if a cluster is smaller than this it is reinitialised."""
    
    self.means = numpy.empty((clusters, feats.shape[1]), dtype=numpy.float_)
    self.means[:,:] = scipy.cluster.vq.kmeans(feats, clusters, restarts)[0]
    
    if assignOut!=None: assignOut[:], _ = scipy.cluster.vq.vq(feats, centres)

