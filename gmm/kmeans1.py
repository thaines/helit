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



class KMeans1(KMeansShared):
  """The most basic implimentation of k-means possible - just brute force with multiple restarts."""

  def doTrain(self, feats, clusters, maxIters = 512, restarts = 8, minSize = 4, callback = None, assignOut = None):
    """Given a large number of features as a data matrix this finds a good set of cluster centres. clusters is the number of clusters to create, maxIters is the maximum number of iterations for convergance of a cluster and restarts how many times to run - the most probable result is used. minSize is the smallest cluster size - if a cluster is smaller than this it is reinitialised."""

    if callback!=None:
      callback(0,restarts+1)

    # Required data structures...
    bestModelScore = None # Score of the model currently in self.means
    centres = numpy.empty((clusters,feats.shape[1]), dtype=numpy.float_)
    assignment = numpy.empty(feats.shape[0], dtype=numpy.int_)
    assignmentCount = numpy.empty(clusters, dtype=numpy.int_)

    # Iterate and run the algorithm from scratch a number of times, calculating the log likelihood of the model each time, so we can ultimatly select the best...
    for r in xrange(restarts):
      if callback!=None:
        callback(1+r,restarts+1)

      # Initialisation - fill in the matrix of cluster centres, where centres are initialised by randomly selecting a point from the data set (With duplication - we let the reinitalisation code deal with them.)...
      for c in xrange(centres.shape[0]):
        ri = random.randrange(feats.shape[0])
        centres[c,:] = feats[ri,:]

      assignment[:] = -1

      try:
        code = """
        for (int i=0;i<maxIters;i++)
        {
         // Reassign features to clusters...
         bool change = false;
         for (int f=0;f<Nfeats[0];f++)
         {
          int best = -1;
          float bestDist = 1e100;
          for (int c=0;c<Ncentres[0];c++)
          {
           float dist = 0.0;
           for (int e=0;e<Nfeats[1];e++)
           {
            float d = FEATS2(f,e) - CENTRES2(c,e);
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
          for (int e=0;e<Nfeats[1];e++) {CENTRES2(c,e) = 0.0;}
          ASSIGNMENTCOUNT1(c) = 0;
         }

         for (int f=0;f<Nfeats[0];f++)
         {
          int c = ASSIGNMENT1(f);
          ASSIGNMENTCOUNT1(c) += 1;
          float div = ASSIGNMENTCOUNT1(c);

          for (int e=0;e<Nfeats[1];e++)
          {
           CENTRES2(c,e) += (FEATS2(f,e) - CENTRES2(c,e)) / div;
          }
         }

         // Reassign puny clusters...
         for (int c=0;c<Ncentres[0];c++)
         {
          if (ASSIGNMENTCOUNT1(c)<minSize)
          {
           int rf = rand() % Nfeats[0];
           for (int e=0;e<Nfeats[1];e++)
           {
            CENTRES2(c,e) = FEATS2(rf,e);
           }
          }
         }
        }
        """

        weave.inline(code,['centres', 'assignment', 'assignmentCount','feats','maxIters','minSize'])

      except:
        # Iterate until convergance...
        for i in xrange(maxIters):
          # Re-assign features to clusters...
          beenChange = False
          for i in xrange(feats.shape[0]): # Slow bit
            nearest = ((centres-feats[i,:])**2).sum(axis=1).argmin()
            if nearest!=assignment[i]:
              beenChange = True
              assignment[i] = nearest

          # If no reassignments happen break out early...
          if beenChange==False: break

          # Recalculate cluster centres (Incrimental mean used)...
          centres[:] = 0.0
          assignmentCount[:] = 0
          for i in xrange(feats.shape[0]):
            c = assignment[i]
            assignmentCount[c] += 1
            centres[c,:] += (feats[i,:]-centres[c,:])/float(assignmentCount[c])

          # Find all puny clusters and reinitialise...
          indices = numpy.argsort(assignmentCount)
          for ic in xrange(indices.shape[0]):
            c = indices[ic]
            if assignmentCount[c]>=minSize:
              break
            ri = random.randrange(feats.shape[0])
            centres[c,:] = feats[ri,:]


      # Calculate the models negative log likelihood - if better than the current model replace it (This is done proportionally)...
      if bestModelScore==None:
        self.means = centres.copy()
      else:
        negLogLike = 0.0
        for i in xrange(feats.shape[0]):
          negLogLike += ((feats[i,:]-centres[assignment[i],:])**2).sum()

        if negLogLike<bestModelScore:
          self.means = centres.copy()

      if assignOut!=None: assignOut[:] = assignment
