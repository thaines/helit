# -*- coding: utf-8 -*-

# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import math
import random

import numpy
import scipy.weave as weave

from kmeans import KMeans
from mixture import Mixture



class IsotropicGMM(Mixture):
  """Fits a gaussian mixture model to a data set, using isotropic Gaussians, i.e. so each is parameterised by only a single standard deviation, in addition to position and weight."""
  def __init__(self):
    self.mix = None # Vector of mixing probabilities.
    self.mean = None # 2D array where each row is the mean of a distribution.
    self.sd = None # Vector of standard deviations.


  def clusterCount(self):
    """Returns how many clusters it has been fitted with, or 0 if it is yet to be trainned."""
    if self.mix!=None: return self.mix.shape[0]
    else: return 0

  def parameters(self):
    """Returns how many parameters the currently fitted model has, used for model selection."""
    return self.mean.shape[0]*(self.mean.shape[1]+2)


  def getMix(self,i):
    """Returns the mixing weight for the given cluster."""
    return self.mix[i]

  def getCentre(self,i):
    """Returns the mean/centre of the given cluster."""
    return self.mean[i,:]

  def getSD(self,i):
    """Returns the standard deviation for the given cluster."""
    return self.sd[i]


  def doTrain(self, feats, clusters, maxIters = 1024, epsilon = 1e-4):
    # Initialise using kmeans...
    km = KMeans()
    kmAssignment = numpy.empty(feats.shape[0],dtype=numpy.float_)
    km.train(feats,clusters,assignOut = kmAssignment)
    
    # Create the assorted data structures needed...
    mix = numpy.ones(clusters,dtype=numpy.float_)/float(clusters)
    mean = numpy.empty((clusters,feats.shape[1]),dtype=numpy.float_)
    for c in xrange(clusters): mean[c,:] = km.getCentre(c)
    sd = numpy.zeros(clusters,dtype=numpy.float_)

    tempCount = numpy.zeros(clusters,dtype=numpy.int_)
    for f in xrange(feats.shape[0]):
      c = kmAssignment[f]
      dist = ((feats[f,:] - mean[c,:])**2).sum()
      tempCount[c] += 1
      sd += (dist-sd)/float(tempCount[c])
    sd = numpy.sqrt(sd/float(feats.shape[1]))

    wv = numpy.ones((feats.shape[0],clusters),dtype=numpy.float_) # Weight vectors calculated in e-step.
    pwv = numpy.empty(clusters,dtype=numpy.float_) # For convergance detection.
    norms = numpy.empty(clusters,dtype=numpy.float_) # Normalising constants for the distributions, to save repeated calculation.

    sqrt2pi = math.sqrt(2.0*math.pi)

    # The code...
    code = """
    for (int iter=0;iter<maxIters;iter++)
    {
     // e-step - for all features calculate the weight vector (Also do convergance detection.)...
     for (int c=0;c<Nmean[0];c++)
     {
      norms[c] = pow(sqrt2pi*sd[c], Nmean[1]);
     }
     
     bool done = true;
     for (int f=0;f<Nfeats[0];f++)
     {
      float sum = 0.0;
      for (int c=0;c<Nmean[0];c++)
      {
       float distSqr = 0.0;
       for (int i=0;i<Nmean[1];i++)
       {
        float diff = FEATS2(f,i) - MEAN2(c,i);
        distSqr += diff*diff;
       }
       pwv[c] = WV2(f,c);
       float core = -0.5*distSqr / (sd[c]*sd[c]);
       WV2(f,c) = mix[c]*exp(core); // Unnormalised.
       WV2(f,c) /= norms[c]; // Normalisation
       sum += WV2(f,c);
      }
      for (int c=0;c<Nmean[0];c++)
      {
       WV2(f,c) /= sum;
       done = done && (fabs(WV2(f,c)-pwv[c])<epsilon);
      }
     }

     if (done) break;


     // Zero out mix,mean and sd, ready for filling...
     for (int c=0;c<Nmean[0];c++)
     {
      mix[c] = 0.0;
      for (int i=0;i<Nmean[1];i++) MEAN2(c,i) = 0.0;
      sd[c] = 0.0;
     }

     
     // m-step - update the mixing vector, means and sd...
     // *Calculate mean and mixing vector incrimentally...
     for (int f=0;f<Nfeats[0];f++)
     {
      for (int c=0;c<Nmean[0];c++)
      {
       mix[c] += WV2(f,c);
       if (WV2(f,c)>1e-6) // Msut not update if value is too low due to division in update - NaN avoidance.
       {
        for (int i=0;i<Nmean[1];i++)
        {
         MEAN2(c,i) += WV2(f,c) * (FEATS2(f,i) - MEAN2(c,i)) / mix[c];
        }
       }
      }
     }
     
     // prevent the mix of any given component getting too low - will cause the algorithm to NaN...
     for (int c=0;c<Nmean[0];c++)
     {
      if (mix[c]<1e-6) mix[c] = 1e-6;
     }

     // *Calculate the sd simply, initial calculation is sum of squared differences...
     for (int f=0;f<Nfeats[0];f++)
     {
      for (int c=0;c<Nmean[0];c++)
      {
       float distSqr = 0.0;
       for (int i=0;i<Nmean[1];i++)
       {
        float delta = FEATS2(f,i) - MEAN2(c,i);
        distSqr += delta*delta;
       }
       sd[c] += WV2(f,c) * distSqr;
      }
     }

     // *Final adjustments for the new state...
     float mixSum = 0.0;
     for (int c=0;c<Nmean[0];c++)
     {
      sd[c] = sqrt(sd[c]/(mix[c]*float(Nfeats[1])));
      mixSum += mix[c];
     }
     
     for (int c=0;c<Nmean[0];c++) mix[c] /= mixSum;
    }
    """

    # Weave it...
    weave.inline(code,['feats', 'maxIters', 'epsilon', 'mix', 'mean', 'sd', 'wv', 'pwv', 'norms', 'sqrt2pi'])

    # Store result...
    self.mix = mix
    self.mean = mean
    self.sd = sd


  def doGetUnnormWeight(self,feats):
    ret = numpy.empty((feats.shape[0],self.mix.shape[0]),dtype=numpy.float_)
    norms = numpy.empty(self.mix.shape[0],dtype=numpy.float_) # Normalising constants for the distributions, to save repeated calculation.

    sqrt2pi = math.sqrt(2.0*math.pi)

    code = """
    for (int c=0;c<Nmean[0];c++) norms[c] = pow(sqrt2pi*sd[c],Nmean[1]);

    for (int f=0;f<Nfeats[0];f++)
    {
     float sum = 0.0;
     for (int c=0;c<Nmean[0];c++)
     {
      float distSqr = 0.0;
      for (int i=0;i<Nmean[1];i++)
      {
       float diff = FEATS2(f,i) - MEAN2(c,i);
       distSqr += diff*diff;
      }
      RET2(f,c) = mix[c]*exp(-0.5*distSqr/(sd[c]*sd[c])); // Unnormalised.
      RET2(f,c) /= norms[c]; // Normalisation
     }
    }
    """

    mix = self.mix
    mean = self.mean
    sd = self.sd
    weave.inline(code,['feats', 'ret', 'mix', 'mean', 'sd', 'norms', 'sqrt2pi'])

    return ret
    
  def doGetWeight(self,feats):
    """Returns the probability of it being drawn from each of the clusters, as a vector. It will obviously sum to one."""
    ret = self.doGetUnnormWeight(feats)
    ret = (ret.T/ret.sum(axis=1)).T
    return ret

  def doGetCluster(self,feats):
    """Returns the cluster it is, with greatest probability, a member of."""
    return self.doGetUnnormWeight(feats).argmax(axis=1)

  def doGetNLL(self,feats):
    """returns the negative log likelihood of the given set of features being drawn from the model."""
    return -numpy.log(self.doGetUnnormWeight(feats).sum(axis=1)).sum()


  def getData(self):
    """Returns the data contained within, so it can be serialised with other data. (You can of course serialise this class directly if you want, but the returned object is a tuple of numpy arrays, so less likely to be an issue for any program that loads it.)"""
    return (self.mix,self.mean,self.sd)

  def setData(self,data):
    """Sets the data for the object, should be same form as returned from getData."""
    self.mix = data[0]
    self.mean = data[1]
    self.sd = data[2]
