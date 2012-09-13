# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import math
import numpy

from scipy import weave
from utils.start_cpp import start_cpp
from utils.matrix_cpp import matrix_code

from loo_cov import PrecisionLOO, SubsetPrecisionLOO # Not used below, just for conveniance.
from gmm import GMM



class KDE_INC:
  """Provides an incrimental kernel density estimate system that uses Gaussians. A kernel density estimate system with Gaussian kernels that, on reaching a cap, starts merging kernels to limit the number of kernels to a constant - done in such a way as to minimise error whilst capping computation. (Computation is quite high however - this is not a very efficient implimentation.)"""
  def __init__(self, prec, cap = 32):
    """Initialise with the precision matrix to use for the kernels, which implicitly provides the number of dimensions, and the cap on the number of kernels to allow."""
    self.prec = numpy.asarray(prec, dtype=numpy.float32)
    self.gmm = GMM(prec.shape[0], cap) # Current mixture model.
    self.count = 0 # Number of samples provided so far.

    self.merge = numpy.empty((cap,cap), dtype=numpy.float32) # [i,j]; cost of merging two entrys, only valid when j<i, other values set high to avoid issues.
    self.merge[:,:] = 1e64

    # For holding the temporary merge costs calculated when adding a sample...
    self.mergeT = numpy.empty(cap, dtype=numpy.float32)

    # For the C code...
    self.temp = numpy.empty((2, prec.shape[0], prec.shape[0]), dtype=numpy.float32)

  def setPrec(self, prec):
    """Changes the precision matrix - must be called before any samples are added, and must have the same dimensions as the current one."""
    self.prec = numpy.asarray(prec, dtype=numpy.float32)


  def samples(self):
    """Returns how many samples have been added to the object."""
    return self.count

  def prob(self, sample):
    """Returns the probability of the given sample - must not be called until at least one sample has been added, though it will return a positive constant if called with no samples provided."""
    if self.count!=0: return self.gmm.prob(sample)
    else: return 1.0

  def nll(self, sample):
    """Returns the negative log liklihood of the given sample - must not be called until at least one sample has been added, though it will return a positive constant if called with no samples provided."""
    if self.count!=0: return self.gmm.nll(sample)
    else: return 0.0


  def __merge(self, weightA, meanA, precA, weightB, meanB, precB):
    """Merges two Gaussians and returns the merged result, as (weight, mean, prec)"""
    newWeight = weightA + weightB
    newMean = weightA/newWeight * meanA + weightB/newWeight * meanB

    deltaA = meanA - newMean
    covA = numpy.linalg.inv(precA) + numpy.outer(deltaA, deltaA)

    deltaB = meanB - newMean
    covB = numpy.linalg.inv(precB) + numpy.outer(deltaB, deltaB)

    newCov = weightA/newWeight * covA + weightB/newWeight * covB
    newPrec = numpy.linalg.inv(newCov)

    return (newWeight, newMean, newPrec)


  def __calcMergeCost(self, weightA, meanA, precA, weightB, meanB, precB):
    """Calculates and returns the cost of merging two Gaussians."""
    # (For anyone wondering about the fact we are comparing them against each other rather than against the result of merging them that is because this way tends to get better results.)

    # The log determinants and delta...
    logDetA = math.log(numpy.linalg.det(precA))
    logDetB = math.log(numpy.linalg.det(precB))
    delta = meanA - meanB

    # Kullback-Leibler of representing A using B...
    klA = logDetB - logDetA
    klA += numpy.trace(numpy.dot(precB, numpy.linalg.inv(precA)))
    klA += numpy.dot(numpy.dot(delta, precB), delta)
    klA -= precA.shape[0]
    klA *= 0.5

    # Kullback-Leibler of representing B using A...
    klB = logDetA - logDetB
    klB += numpy.trace(numpy.dot(precA, numpy.linalg.inv(precB)))
    klB += numpy.dot(numpy.dot(delta, precA), delta)
    klB -= precB.shape[0]
    klB *= 0.5

    # Return a weighted average...
    return weightA * klA + weightB * klB


  def add(self, sample):
    """Adds a sample, updating the kde accordingly."""
    global weave

    try:
      weave = None # Below code is actually slowing things down. Am disabling for now.
      if weave==None: raise Exception()
      support =  matrix_code + start_cpp() + """
      // Note - designed so that A and Out pointers can be the same.
       void doMerge(int size, float weightA, float * meanA, float * precA, float weightB, float * meanB, float * precB, float & weightOut, float * meanOut, float * precOut, float * tVec, float * tMat1, float * tMat2)
       {
        // Handle the weight, recording the ratios needed next...
         float wOut = weightA + weightB;
         float ratioA = weightA/wOut;
         float ratioB = weightB/wOut;
         weightOut = wOut;

        // Do the mean - simply a weighted average - store in a temporary for now...
         for (int i=0; i<size; i++)
         {
          tVec[i] = ratioA * meanA[i] + ratioB * meanB[i];
         }

        // Put the covariance of precision A into tMat1...
         for (int i=0; i<size*size; i++) tMat2[i] = precA[i];
         Inverse(tMat2, tMat1, size);

        // Add the outer product of the A delta into tMat1...
         for (int r=0; r<size; r++)
         {
          for (int c=0; c<size; c++)
          {
           tMat1[r*size + c] += (meanA[c] - tVec[c]) * (meanA[r] - tVec[r]);
          }
         }

        // Put the covariance of precision B into tMat2...
         for (int i=0; i<size*size; i++) precOut[i] = precB[i];
         Inverse(precOut, tMat2, size);

        // Add the outer product of the B delta into tMat2...
         for (int r=0; r<size; r++)
         {
          for (int c=0; c<size; c++)
          {
           tMat2[r*size + c] += (meanB[c] - tVec[c]) * (meanB[r] - tVec[r]);
          }
         }

        // Get the weighted average of the covariance matrices into tMat1...
         for (int i=0; i<size*size; i++)
         {
          tMat1[i] = ratioA * tMat1[i] + ratioB * tMat2[i];
         }

        // Dump the inverse of tMat1 into the output precision...
         Inverse(tMat1, precOut, size);

        // Copy from the temporary mean into the output mean...
         for (int i=0; i<size; i++) meanOut[i] = tVec[i];
       }

      float mergeCost(int size, float weightA, float * meanA, float * precA, float weightB, float * meanB, float * precB, float * tVec1, float * tVec2, float * tMat1, float * tMat2)
      {
       // Calculate some shared values...
        float logDetA = log(Determinant(precA, size));
        float logDetB = log(Determinant(precB, size));

        for (int i=0; i<size; i++)
        {
         tVec1[i] = meanA[i] - meanB[i];
        } // tVec1 now contains the delta.

       // Calculate the Kullback-Leibler divergance of substituting B for A...
        float klA = logDetB - logDetA;

        for (int i=0; i<size*size; i++) tMat1[i] = precA[i];
        if (Inverse(tMat1, tMat2, size)==false) return 0.0;
        for (int i=0; i<size; i++)
        {
         for (int j=0; j<size; j++)
         {
          klA += precB[i*size + j] * tMat2[j*size + i];
         }
        }

        for (int i=0; i<size; i++)
        {
         tVec2[i] = 0.0;
         for (int j=0; j<size; j++)
         {
          tVec2[i] += precB[i*size + j] * tVec1[j];
         }
        }
        for (int i=0; i<size; i++) klA += tVec1[i] * tVec2[i];
        klA -= size;
        klA *= 0.5;

       // Calculate the Kullback-Leibler divergance of substituting A for B...
        float klB = logDetA - logDetB;

        for (int i=0; i<size*size; i++) tMat1[i] = precB[i];
        if (Inverse(tMat1, tMat2, size)==false) return 0.0;
        for (int i=0; i<size; i++)
        {
         for (int j=0; j<size; j++)
         {
          klB += precA[i*size + j] * tMat2[j*size + i];
         }
        }

        for (int i=0; i<size; i++)
        {
         tVec2[i] = 0.0;
         for (int j=0; j<size; j++)
         {
          tVec2[i] += precA[i*size + j] * tVec1[j];
         }
        }
        for (int i=0; i<size; i++) klB += tVec1[i] * tVec2[i];
        klB -= size;
        klB *= 0.5;

       // Return a weighted average of the divergances...
        return weightA * klA + weightB * klB;
      }
      """

      code = start_cpp(support) + """
      if (count < Nweight[0])
      {
       // Pure KDE mode - just add the kernel...
        for (int i=0; i<Nsample[0]; i++)
        {
         MEAN2(count, i) = sample[i];
        }

        for (int i=0; i<Nsample[0]; i++)
        {
         for (int j=0; j<Nsample[0]; j++)
         {
          PREC3(count, i, j) = BASEPREC2(i, j);
         }
        }

        assert(Sprec[0]==sizeof(float));
        assert(Sprec[1]==sizeof(float)*Nsample[0]);
        log_norm[count]  = 0.5 * log(Determinant(&PREC3(count, 0, 0), Nsample[0]));
        log_norm[count] -= 0.5 * Nsample[0] * log(2.0*M_PI);

        float w = 1.0 / (count+1);
        for (int i=0; i<=count; i++)
        {
         weight[i] = w;
        }

       // If the next sample will involve merging then we need to fill in the merging costs cache in preperation...
        if (count+1==Nweight[0])
        {
         for (int i=0; i<Nweight[0]; i++)
         {
          for (int j=0; j<i; j++)
          {
           MERGE2(i, j) = mergeCost(Nsample[0], weight[i], &MEAN2(i,0), &PREC3(i,0,0), weight[j], &MEAN2(j,0), &PREC3(j,0,0), &TEMP2(0,0), &TEMP2(1,0), &TEMPPREC3(0,0,0), &TEMPPREC3(1,0,0));
          }
         }
        }
      }
      else
      {
       // We have the maximum number of kernels - need to either merge the new kernel with an existing one, or merge two existing kernels and use the freed up slot for the new kernel...

       // Update the weights, and calculate the weight of the new kernel...
        float adjust = float(count) / float(count+1);

        for (int i=0; i<Nweight[0]; i++) weight[i] *= adjust;
        for (int i=0; i<Nweight[0]; i++)
        {
         for (int j=0; j<i; j++) MERGE2(i, j) *= adjust;
        }

        float w = 1.0 / float(count + 1.0);

       // Calculate the costs of merging the new kernel with each of the old kernels...
        for (int i=0; i<Nweight[0]; i++)
        {
         mergeT[i] = mergeCost(Nsample[0], w, sample, basePrec, weight[i], &MEAN2(i,0), &PREC3(i,0,0), &TEMP2(0,0), &TEMP2(1,0), &TEMPPREC3(0,0,0), &TEMPPREC3(1,0,0));
        }

       // Find the lowest merge cost and act accordingly - either we are merging the new kernel with an old one or merging two existing kernels and putting the new kernel in on its own...
        int lowI = 1;
        int lowJ = 0;

        for (int i=0; i<Nweight[0]; i++)
        {
         for (int j=0; j<i; j++)
         {
          if (MERGE2(i, j) < MERGE2(lowI, lowJ))
          {
           lowI = i;
           lowJ = j;
          }
         }
        }

        int lowN = 0;

        for (int i=1; i<Nweight[0]; i++)
        {
         if (mergeT[i] < mergeT[lowN]) lowN = i;
        }

        if (mergeT[lowN] < MERGE2(lowI, lowJ))
        {
         // We are merging the new kernel with an existing kernel...

         // Do the merge...
          doMerge(Nsample[0], weight[lowN], &MEAN2(lowN,0), &PREC3(lowN,0,0), w, sample, basePrec, weight[lowN], &MEAN2(lowN,0), &PREC3(lowN,0,0), &TEMP2(0,0), &TEMPPREC3(0,0,0), &TEMPPREC3(1,0,0));

         // Update the normalising constant...
          log_norm[lowN]  = 0.5 * log(Determinant(&PREC3(lowN, 0, 0), Nsample[0]));
          log_norm[lowN] -= 0.5 * Nsample[0] * log(2.0*M_PI);

         // Update the array of merge costs...
          for (int i=0; i<Nweight[0]; i++)
          {
           if (i!=lowN)
           {
            float mc = mergeCost(Nsample[0], weight[i], &MEAN2(i,0), &PREC3(i,0,0), weight[lowN], &MEAN2(lowN,0), &PREC3(lowN,0,0), &TEMP2(0,0), &TEMP2(1,0), &TEMPPREC3(0,0,0), &TEMPPREC3(1,0,0));

            if (i<lowN) MERGE2(lowN, i) = mc;
                   else MERGE2(i, lowN) = mc;
           }
          }
        }
        else
        {
         // We are merging two existing kernels then putting the new kernel into the freed up spot...

         // Do the merge...
          doMerge(Nsample[0], weight[lowI], &MEAN2(lowI,0), &PREC3(lowI,0,0), weight[lowJ], &MEAN2(lowJ,0), &PREC3(lowJ,0,0), weight[lowI], &MEAN2(lowI,0), &PREC3(lowI,0,0), &TEMP2(0,0), &TEMPPREC3(0,0,0), &TEMPPREC3(1,0,0));

         // Copy in the new kernel...
          weight[lowJ] = w;
          for (int i=0; i<Nsample[0]; i++) MEAN2(lowJ,i) = sample[i];
          for (int i=0; i<Nsample[0];i++)
          {
           for (int j=0; j<Nsample[0]; j++)
           {
            PREC3(lowJ,i,j) = basePrec[i*Nsample[0] + j];
           }
          }

         // Update both normalising constants...
          log_norm[lowI]  = 0.5 * log(Determinant(&PREC3(lowI, 0, 0), Nsample[0]));
          log_norm[lowI] -= 0.5 * Nsample[0] * log(2.0*M_PI);

          log_norm[lowJ]  = 0.5 * log(Determinant(&PREC3(lowJ, 0, 0), Nsample[0]));
          log_norm[lowJ] -= 0.5 * Nsample[0] * log(2.0*M_PI);

         // Update the array of merge costs...
          for (int i=0; i<Nweight[0]; i++)
          {
           if (i!=lowI)
           {
            float mc = mergeCost(Nsample[0], weight[i], &MEAN2(i,0), &PREC3(i,0,0), weight[lowI], &MEAN2(lowI,0), &PREC3(lowI,0,0), &TEMP2(0,0), &TEMP2(1,0), &TEMPPREC3(0,0,0), &TEMPPREC3(1,0,0));

            if (i<lowI) MERGE2(lowI, i) = mc;
                   else MERGE2(i, lowI) = mc;
           }
          }

          for (int i=0; i<Nweight[0]; i++)
          {
           if ((i!=lowI)&&(i!=lowJ))
           {
            float mc = mergeCost(Nsample[0], weight[i], &MEAN2(i,0), &PREC3(i,0,0), weight[lowJ], &MEAN2(lowJ,0), &PREC3(lowJ,0,0), &TEMP2(0,0), &TEMP2(1,0), &TEMPPREC3(0,0,0), &TEMPPREC3(1,0,0));

            if (i<lowJ) MERGE2(lowJ, i) = mc;
                   else MERGE2(i, lowJ) = mc;
           }
          }
        }
      }
      """

      sample = numpy.asarray(sample, dtype=numpy.float32).flatten()
      basePrec = self.prec
      count = self.count
      merge = self.merge
      mergeT = self.mergeT
      tempPrec = self.temp

      weight = self.gmm.weight
      mean = self.gmm.mean
      prec = self.gmm.prec
      log_norm = self.gmm.log_norm
      temp = self.gmm.temp

      weave.inline(code, ['sample', 'basePrec', 'count', 'merge', 'mergeT', 'tempPrec', 'weight', 'mean', 'prec', 'log_norm', 'temp'], support_code = support)
      self.count += 1

    except Exception, e:
      if weave!=None:
        print e
        weave = None

      if self.count<self.gmm.weight.shape[0]:
        # Pure kde phase...
        self.gmm.mean[self.count,:] = numpy.asarray(sample, dtype=numpy.float32)
        self.gmm.prec[self.count,:,:] = self.prec
        self.gmm.calcNorm(self.count)

        self.count += 1
        self.gmm.weight[:self.count] = 1.0 / float(self.count)

        if self.count==self.gmm.weight.shape[0]:
          # Next sample starts merging - need to prepare by filling in the kl array...
          # (Below is grossly inefficient - calculates the same things more times than is possibly funny. I'll optimise it if I ever decide that I care enough to do so.)
          for i in xrange(self.merge.shape[0]):
            for j in xrange(i):
              self.merge[i,j] = self.__calcMergeCost(self.gmm.weight[i], self.gmm.mean[i,:], self.gmm.prec[i,:,:], self.gmm.weight[j], self.gmm.mean[j,:], self.gmm.prec[j,:,:])
      else:
        # Merging phase...
        sample = numpy.asarray(sample, dtype=numpy.float32)

        # Adjust weights...
        adjust = float(self.count) / float(self.count+1)
        self.gmm.weight *= adjust
        for i in xrange(self.merge.shape[0]): self.merge[i,:i] *= adjust

        self.count += 1
        weight = 1.0 / float(self.count)

        # Calculate the merging costs for the new kernel versus the old kernels...
        for i in xrange(self.merge.shape[0]):
          self.mergeT[i] = self.__calcMergeCost(weight, sample, self.prec, self.gmm.weight[i], self.gmm.mean[i,:], self.gmm.prec[i,:,:])

        # Select the best merge - it either involves the new sample or it does not...
        bestOld = numpy.unravel_index(numpy.argmin(self.merge), self.merge.shape)
        bestNew = numpy.argmin(self.mergeT)
        if self.mergeT[bestNew] < self.merge[bestOld]:
          # Easy scenario - new kernel is being merged with an existing kernel - not too much fiddling involved...

          # Do the merge...
          newWeight, newMean, newPrec = self.__merge(weight, sample, self.prec, self.gmm.weight[bestNew], self.gmm.mean[bestNew,:], self.gmm.prec[bestNew,:,:])

          # Store the result...
          self.gmm.weight[bestNew] = newWeight
          self.gmm.mean[bestNew,:] = newMean
          self.gmm.prec[bestNew,:,:] = newPrec
          self.gmm.calcNorm(bestNew)

          # Update the merge weights...
          for i in xrange(self.merge.shape[0]):
            if i!=bestNew:
              cost = self.__calcMergeCost(self.gmm.weight[i], self.gmm.mean[i,:], self.gmm.prec[i,:,:], self.gmm.weight[bestNew], self.gmm.mean[bestNew,:], self.gmm.prec[bestNew,:,:])
              if i<bestNew: self.merge[bestNew,i] = cost
              else: self.merge[i,bestNew] = cost

        else:
          # We are merging two old kernels, and then putting the new kernel into the slot freed up - this is extra fiddly...
          # Do the merge...
          newWeight, newMean, newPrec = self.__merge(self.gmm.weight[bestOld[0]], self.gmm.mean[bestOld[0],:], self.gmm.prec[bestOld[0],:,:], self.gmm.weight[bestOld[1]], self.gmm.mean[bestOld[1],:], self.gmm.prec[bestOld[1],:,:])

          # Store the result, put the new component in the other slot...
          self.gmm.weight[bestOld[0]] = newWeight
          self.gmm.mean[bestOld[0],:] = newMean
          self.gmm.prec[bestOld[0],:,:] = newPrec
          self.gmm.calcNorm(bestOld[0])

          self.gmm.weight[bestOld[1]] = weight
          self.gmm.mean[bestOld[1],:] = sample
          self.gmm.prec[bestOld[1],:,:] = self.prec
          self.gmm.calcNorm(bestOld[1])

          # Update the merge weights for both the merged and new kernels...
          for i in xrange(self.merge.shape[0]):
            if i!=bestOld[0]:
              cost = self.__calcMergeCost(self.gmm.weight[i], self.gmm.mean[i,:], self.gmm.prec[i,:,:], self.gmm.weight[bestOld[0]], self.gmm.mean[bestOld[0],:], self.gmm.prec[bestOld[0],:,:])
              if i<bestOld[0]: self.merge[bestOld[0],i] = cost
              else: self.merge[i,bestOld[0]] = cost

          for i in xrange(self.merge.shape[0]):
            if i!=bestOld[0] and i!=bestOld[1]:
              cost = self.__calcMergeCost(self.gmm.weight[i], self.gmm.mean[i,:], self.gmm.prec[i,:,:], self.gmm.weight[bestOld[1]], self.gmm.mean[bestOld[1],:], self.gmm.prec[bestOld[1],:,:])
              if i<bestOld[1]: self.merge[bestOld[1],i] = cost
              else: self.merge[i,bestOld[1]] = cost


  def marginalise(self, dims):
    """Returns an object on which you can call prob(), but with only a subset of the dimensions. The set of dimensions is given as something that can be interpreted as a numpy array of integers - it is the dimensions to keep, it marginalises away everything else. The indexing of the returned object will match up with that in dims. Note that you must not have any repetitions in dims - that would create absurdity."""
    ret = self.gmm.clone()
    ret.marginalise(dims)
    return ret
