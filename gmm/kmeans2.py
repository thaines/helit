# -*- coding: utf-8 -*-

# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import math
import random

import numpy
import scipy.weave as weave

from kmeans_shared import KMeansShared



class KMeans2(KMeansShared):
  """This version of kmeans gets clever with its initialisation, running k-means on a subset of points, repeatedly, then combining the various runs, before ultimatly only doing k-means on the full dataset just once. This optimisation is only valuable for large data sets, e.g. with at least 10k feature vectors, and will cause slow down for small data sets."""
  def __kmeans(self, centres, data, minSize = 3, maxIters = 1024, assignOut = None):
    """Internal method - does k-means on the data set as it is treated internally. Given the initial set of centres and a data matrix - the centres matrix is then updated to the new positions."""
    assignment = numpy.empty(data.shape[0], dtype=numpy.int_)
    assignmentCount = numpy.empty(centres.shape[0], dtype=numpy.int_)

    assignment[:] = -1

    code = """
    for (int i=0;i<maxIters;i++)
    {
     // Reassign features to clusters...
     bool change = false;
     for (int f=0;f<Ndata[0];f++)
     {
      int best = -1;
      float bestDist = 1e100;
      for (int c=0;c<Ncentres[0];c++)
      {
       float dist = 0.0;
       for (int e=0;e<Ndata[1];e++)
       {
        float d = DATA2(f,e) - CENTRES2(c,e);
        dist += d*d;
       }

       if (dist<bestDist)
       {
        best = c;
        bestDist = dist;
       }
      }

      if (best!=ASSIGNMENT1(f))
      {
       ASSIGNMENT1(f) = best;
       change = true;
      }
     }

     // If no reassignments happen break out early...
     if (change==false) break;

     // Recalculate cluster centres with an incrimental mean...
     for (int c=0;c<Ncentres[0];c++)
     {
      for (int e=0;e<Ndata[1];e++) {CENTRES2(c,e) = 0.0;}
      ASSIGNMENTCOUNT1(c) = 0;
     }

     for (int f=0;f<Ndata[0];f++)
     {
      int c = ASSIGNMENT1(f);
      ASSIGNMENTCOUNT1(c) += 1;
      float div = ASSIGNMENTCOUNT1(c);

      for (int e=0;e<Ndata[1];e++)
      {
       CENTRES2(c,e) += (DATA2(f,e) - CENTRES2(c,e)) / div;
      }
     }

     // Reassign puny clusters...
     for (int c=0;c<Ncentres[0];c++)
     {
      if (ASSIGNMENTCOUNT1(c)<minSize)
      {
       int rf = rand() % Ndata[0];
       for (int e=0;e<Ndata[1];e++)
       {
        CENTRES2(c,e) = DATA2(rf,e);
       }
      }
     }
    }
    """

    weave.inline(code,['centres', 'assignment', 'assignmentCount','data','maxIters','minSize'])

    if assignOut!=None: assignOut[:] = assignment


  def doTrain(self, feats, clusters, callback = None, smallCount = 32, smallMult = 3, minSize = 2, maxIters = 256, assignOut = None):
    """Given a features data matrix this finds a good set of cluster centres. clusters is the number of clusters to create."""

    if callback!=None:
      callback(0,smallCount*2+2)

    # Iterate through and create a set of centres derived from small samples...
    smallSize = clusters * smallMult * minSize
    startOptions = []
    for i in xrange(smallCount):
      if callback!=None:
        callback(1+i,smallCount*2+2)

      # Grab a small sample from the data matrix, in randomised order...
      sample = feats[numpy.random.permutation(feats.shape[0])[:smallSize],:]

      # Select the first 'clusters' rows as the initial centres - its already in random order so no point in more randomisation...
      centres = sample[:clusters,:]

      # Run k-means...
      self.__kmeans(centres, sample, minSize, maxIters)

      # Store the set of centres ready for the next bit...
      startOptions.append(centres)

    # Iterate through each of the centres calculated above, using them to initialise kmeans on all the centres combined; select the centres that have the minimum negative log likelihood on this limited data set as the intialisation for the final step (Use a value that is proportional rather than actually calculating -log likelihood.)...
    allStarts = numpy.vstack(startOptions)
    bestStart = None
    bestScore = 1e100
    for i,centres in enumerate(startOptions):
      if callback!=None:
        callback(1+smallCount+i,smallCount*2+2)

      # Run k-means...
      self.__kmeans(centres, allStarts, minSize, maxIters)

      # Calculate the sum of distances from the nearest centres, which is proportional to the negative log likelihood of the model...
      distSum = 0.0
      for i in xrange(allStarts.shape[0]):
        distSum += ((centres-allStarts[i])**2).sum(axis=1).min()

      # If better then previous scores store it for use...
      if distSum<bestScore:
        bestStart = centres
        bestScore = distSum

    # Finally do k-means on the full data set, using the initialisation that has been calculated...
    if callback!=None:
      callback(1+smallCount*2,smallCount*2+2)

    self.__kmeans(bestStart, feats, minSize, maxIters, assignOut)
    self.means = bestStart

    del callback # There is a bug in here somewhere and this fixes it. Fixing the actual bug might be wise, but I just can't see it. (Somehow the functions callspace is kept, but can't see how.)
