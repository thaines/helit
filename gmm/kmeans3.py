# -*- coding: utf-8 -*-

# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import math
import random

import numpy
import scipy.weave as weave

from kmeans1 import KMeans1
from kmeans2 import KMeans2



class KMeans3(KMeans2):
  """Takes the initialisation improvememnts of KMeans2 and additionally makes improvements to the kmeans implimentation. Specifically it presumes that distance computations are expensive and avoids doing them where possible, by additionally storing how far each cluster centre moved since the last step. This allows a lot of such computations to be pruned, especially later on when cluster centres are not moving. It does not work if distance computations are cheap however, so only worth it for long feature vectors."""
  def __kmeans(self, centres, data, minSize = 3, maxIters = 1024, assignOut = None):
    """Internal method - does k-means on the data set as it is treated internally. Given the initial set of centres and a data matrix - the centres matrix is then updated to the new positions."""
    assignment = numpy.empty(data.shape[0], dtype=numpy.int_)
    assignmentDist = numpy.empty(data.shape[0], dtype=numpy.float_) # Distance squared
    assignment[:] = -1
    assignmentDist[:] = 0.0

    assignmentCount = numpy.empty(centres.shape[0], dtype=numpy.int_)
    motion = numpy.empty(centres.shape[0], dtype=numpy.float_) # This is distance, rather than distance squared, unlike for assignments.
    motion[:] = 1e100

    tempCentres = numpy.empty(centres.shape, dtype=numpy.float_)

    code = """
    for (int i=0;i<maxIters;i++)
    {
     // Reassign features to clusters...
     bool change = false;
     for (int f=0;f<Ndata[0];f++)
     {
      // Initialise the best to the previous best...
      int best = ASSIGNMENT1(f);
      float bestDist; // Squared distance.
      if (best==-1) bestDist = 1e100;
      else
      {
       if (MOTION1(best)<1e-6) bestDist = ASSIGNMENTDIST1(f);
       else
       {
        bestDist = 0.0;
        for (int e=0;e<Ndata[1];e++)
        {
         float d = DATA2(f,e) - CENTRES2(best,e);
         bestDist += d*d;
        }
       }
      }

      // Check all the centres to see if they are better than the current best...
      float rad = sqrt(ASSIGNMENTDIST1(f)); // Distance to closest center from *previous* iteration - all other centers must of been further away in this iteration.
      for (int c=0;c<Ncentres[0];c++)
      {
       if (c==ASSIGNMENT1(f)) continue;

       // Only do the distance calculation if the node could actually suceded..
       float optDist = rad - MOTION1(c);
       optDist *= optDist; // Distance -> distance squared.
       if (optDist<bestDist)
       {
        // Node stands a chance - do the full calculation...
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
      }

      change |= (best!=ASSIGNMENT1(f));
      ASSIGNMENT1(f) = best;
      ASSIGNMENTDIST1(f) = bestDist;
     }

     // If no reassignments happen break out early...
     if (change==false) break;

     // Recalculate cluster centres with an incrimental mean...
     for (int c=0;c<Ncentres[0];c++)
     {
      for (int e=0;e<Ndata[1];e++) {TEMPCENTRES2(c,e) = 0.0;}
      ASSIGNMENTCOUNT1(c) = 0;
     }

     for (int f=0;f<Ndata[0];f++)
     {
      int c = ASSIGNMENT1(f);
      ASSIGNMENTCOUNT1(c) += 1;
      float div = ASSIGNMENTCOUNT1(c);

      for (int e=0;e<Ndata[1];e++)
      {
       TEMPCENTRES2(c,e) += (DATA2(f,e) - TEMPCENTRES2(c,e)) / div;
      }
     }

     // Transfer over, storing the motion vectors...
     for (int c=0;c<Ncentres[0];c++)
     {
      MOTION1(c) = 0.0;
      for (int e=0;e<Ndata[1];e++)
      {
       float delta = TEMPCENTRES2(c,e) - CENTRES2(c,e);
       MOTION1(c) += delta*delta;
       CENTRES2(c,e) = TEMPCENTRES2(c,e);
      }
      MOTION1(c) = sqrt(MOTION1(c));
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
       MOTION1(c) = 1e100;
      }
     }
    }
    """

    weave.inline(code,['centres', 'assignment', 'assignmentDist', 'assignmentCount', 'motion', 'data', 'tempCentres', 'maxIters', 'minSize'])

    if assignOut!=None: assignOut[:] = assignment
