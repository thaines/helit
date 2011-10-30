# Copyright (c) 2011, Tom SF Haines
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#  * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



from utils.start_cpp import start_cpp



# Provides code for estimating the Dirichlet distribution from which a number of multinomial distributions were drawn from, given those multinomials...
dir_est_code = start_cpp() + """
// Defined as a class - you then add each multinomial before requesting a maximum likelihood update of the Dirichlet distribution. It uses Newton-Raphson iterations, and so needs a starting point - you provide a vector to be updated, which can of course save time if it is already close...
class EstimateDir
{
 public:
   EstimateDir(int vecSize):size(vecSize), samples(0), meanLog(new double[vecSize]), grad(new double[vecSize]), qq(new double[vecSize])
   {
    for (int i=0;i<vecSize;i++) meanLog[i] = 0.0;
   }
  ~EstimateDir() {delete[] meanLog; delete[] grad; delete[] qq;}


  void Add(float * mn)
  {
   samples += 1;
   for (int i=0;i<size;i++)
   {
    meanLog[i] += (log(mn[i]) - meanLog[i]) / double(samples);
   }
  }
  
  void Add(double * mn)
  {
   samples += 1;
   for (int i=0;i<size;i++)
   {
    meanLog[i] += (log(mn[i]) - meanLog[i]) / double(samples);
   }
  }

  
  void Update(float * dir, int maxIter = 64, float epsilon = 1e-3, float cap = 1e6)
  {
   for (int iter=0;iter<maxIter;iter++)
   {
    // We will need the sum of the dir vector...
     double dirSum = 0.0;
     for (int i=0;i<size;i++)
     {
      dirSum += dir[i];
     }

    // Check for Nan/inf - if so reset to basic value...
     if ((dirSum==dirSum) || (dirSum>1e100))
     {
      for (int i=0;i<size;i++) dir[i] = 1.0;
      dirSum = size;
     }

    // Safety - don't let it get too precise, that probably means its being crazy (Can happen with too few samples.)...
     if (dirSum>cap)
     {
      float mult = cap / dirSum;
      for (int i=0;i<size;i++)
      {
       dir[i] *= mult;
      }
      dirSum = cap;
     }
     
    // Calculate the gradiant and the Hessian 'matrix', except its actually diagonal...
     double digDirSum = digamma(dirSum);
     for (int i=0;i<size;i++)
     {
      grad[i] = samples * (digDirSum - digamma(dir[i]) + meanLog[i]);
      qq[i] = -samples * trigamma(dir[i]);
     }

    // Calculate b...
     double b = 0.0;
     double bDiv = 1.0 / (samples*trigamma(dirSum));
     for (int i=0;i<size;i++)
     {
      b += grad[i]/qq[i];
      bDiv += 1.0/qq[i];
     }
     b /= bDiv;

    // Do the update, sum the change...
     double change = 0.0;
     for (int i=0;i<size;i++)
     {
      double delta = (grad[i] - b) / qq[i];
      dir[i] -= delta;
      if (dir[i]<1e-3) dir[i] = 1e-3;
      change += fabs(delta);
     }

    // Break if no change...
     if (change<epsilon) break;
   }
  }
  
  void Update(double * dir, int maxIter = 64, double epsilon = 1e-6, double cap = 1e6)
  {
   for (int iter=0;iter<maxIter;iter++)
   {
    // We will need the sum of the dir vector...
     double dirSum = 0.0;
     for (int i=0;i<size;i++)
     {
      dirSum += dir[i];
     }

    // Check for Nan/inf - if so reset to basic value...
     if ((dirSum==dirSum) || (dirSum>1e100))
     {
      for (int i=0;i<size;i++) dir[i] = 1.0;
      dirSum = size;
     }

    // Safety - don't let it get too precise, that probably means its being crazy (Can happen with too few samples.)...
     if (dirSum>cap)
     {
      float mult = cap / dirSum;
      for (int i=0;i<size;i++)
      {
       dir[i] *= mult;
      }
      dirSum = cap;
     }

    // Calculate the gradiant and the Hessian 'matrix', except its actually diagonal...
     double digDirSum = digamma(dirSum);
     for (int i=0;i<size;i++)
     {
      grad[i] = samples * (digDirSum - digamma(dir[i]) + meanLog[i]);
      qq[i] = -samples * trigamma(dir[i]);
     }

    // Calculate b...
     double b = 0.0;
     double bDiv = 1.0 / (samples*trigamma(dirSum));
     for (int i=0;i<size;i++)
     {
      b += grad[i]/qq[i];
      bDiv += 1.0/qq[i];
     }
     b /= bDiv;

    // Do the update, sum the change...
     double change = 0.0;
     for (int i=0;i<size;i++)
     {
      double delta = (grad[i] - b) / qq[i];
      dir[i] -= delta;
      change += fabs(delta);
     }

    // Break if no change...
     if (change<epsilon) break;
   }
  }
  

 private:
  int size;
  int samples;
  double * meanLog; // Vector of length size, contains the component-wise mean of the log of each of the samples - consititutes the sufficient statistics required to do the update.
  double * grad; // Temporary during update.
  double * qq; // Temporary during update.
};

"""
