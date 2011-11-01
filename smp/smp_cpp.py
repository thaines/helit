# Copyright 2011 Tom SF Haines

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.



from utils.start_cpp import start_cpp
from utils.numpy_help_cpp import numpy_util_code
from dp_utils.sampling_cpp import sampling_code



smp_code = numpy_util_code + sampling_code + start_cpp() + """
#ifndef SMP_CODE
#define SMP_CODE

class SMP
{
 public:
  // Basic constructor - after construction before anything else the Init method must be called...
   SMP()
   :fia(0),priorMN(0),sam(0),samPos(0),samTemp(0),power(0),temp(0)
   {}

  // Constructor that calls the init method...
   SMP(int flagSize, int fliSize)
   :fia(0),priorMN(0),sam(0),samPos(0),samTemp(0),power(0),temp(0)
   {
    Init(flagSize, fliSize);
   }

  // Cleans up...
   ~SMP()
   {
    delete[] temp;
    delete[] power;
    delete[] samTemp;
    delete[] samPos;
    delete[] sam;
    delete[] priorMN;
    delete[] fia;
   }

  // Initialises the Sparse Multinomial Posterior object with the length of each flag sequence as flagSize and the number of such flag sequences in the system with fliSize. Note that the flag list must be provided by a flag index array that has had its addSingles method correctly called...
   void Init(int flagSize, int fliSize)
   {
    // Clean up...
     delete[] temp;
     delete[] power;
     delete[] samTemp;
     delete[] samPos;
     delete[] sam;
     delete[] priorMN;
     delete[] fia;
    
    // Store sizes...
     flagLen = flagSize;
     fliLen = fliSize;

     // Initialise the flag index array - its filled in later...
     fia = new unsigned char[flagLen*fliLen];

    // Setup the prior - by default a uniform...
     priorMN = new float[flagLen];
     for (int f=0;f<flagLen;f++) priorMN[f] = 1.0/flagLen;
     priorConc = flagLen;

    // Zero out the sampling set - user has to add some samples before use...
     samLen = 0;
     sam = 0;
     samPos = 0;
     samTemp = 0;

    // The power counting array - stores the exponent term for each flag list...
     power = new int[fliLen];
     for (int s=0;s<fliLen;s++) power[s] = 0;

    // The temporary vector, which gets so many uses...
     temp = new float[flagLen];
   }


  // Fills in the flag index array - must be called in practise immediatly after the constructor. Input is the output of calling getFlagMatrix on a FlagIndexArray.
   void SetFIA(PyArrayObject * arr)
   {
    for (int s=0;s<fliLen;s++)
    {
     for (int f=0;f<flagLen;f++)
     {
      fia[s*flagLen + f] = Byte2D(arr,s,f);
     }
    }
   }

  // For if you have the fia as an array of unsigned char's instead...
   void SetFIA(unsigned char * arr)
   {
    for (int s=0;s<fliLen;s++)
    {
     for (int f=0;f<flagLen;f++)
     {
      fia[s*flagLen + f] = arr[s*flagLen + f];
     }
    }
   }
   
  // Sets the number of samples to use for the estimation - basically draws a large number of positions from a uniform Dirichlet distribution and then pre-calculates the values required such that the calculation of the mean given samples is trivial. Must be called before sampling occurs.
   void SetSampleCount(int count = 1024)
   {
    // Handle memory...
     samLen = count;
     delete[] sam;
     sam = new float[samLen*fliLen];
     delete[] samPos;
     samPos = new float[samLen*flagLen];
     delete[] samTemp;
     samTemp = new float[samLen];

    // Generate the samples...
     for (int a=0;a<samLen;a++)
     {
      // Draw a distribution from the uniform Dirichlet - we are going to integrate by the classic summing of lots of uniform samples approach...
       float sum = 0.0;
       for (int f=0;f<flagLen;f++)
       {
        temp[f] = -log(1.0 - sample_uniform()); // Identical to sample_gamma(1), but without the code to deal with values other than 1!
        sum += temp[f];
       }
       
       for (int f=0;f<flagLen;f++) temp[f] /= sum;

      // Calculate and store the log of each of the sums implied by the flag array - this makes the integration sampling nice and efficient...
       for (int s=0;s<fliLen;s++)
       {
        float * out = &sam[a*fliLen + s];
        *out = 0.0;

        for (int f=0;f<flagLen;f++)
        {
         if (fia[s*flagLen + f]!=0) *out += temp[f];
        }

        *out = log(*out);
       }

      // Also fill in the samPos array...
       for (int f=0;f<flagLen;f++)
       {
        samPos[a*flagLen + f] = temp[f];
       }
     }
   }

  // Sets the Dirichlet prior, using a vector that sums to unity and a concentration...
   void SetPrior(float * mn, float conc)
   {
    for (int f=0;f<flagLen;f++) priorMN[f] = mn[f];
    priorConc = conc;
   }

  // Sets the Dirichlet prior, using a vector that sums to the concentration...
   void SetPrior(float * dir)
   {
    priorConc = 0.0;
    for (int f=0;f<flagLen;f++)
    {
     priorMN[f] = dir[f];
     priorConc += priorMN[f];
    }
    for (int f=0;f<flagLen;f++) priorMN[f] /= priorConc;
   }

  // This version takes python objects - a numpy array of floats and a python float for the concentration...
   void SetPrior(PyArrayObject * mn, PyObject * conc)
   {
    for (int f=0;f<flagLen;f++) priorMN[f] = Float1D(mn,f);
    priorConc = PyFloat_AsDouble(conc);
   }
   

  // Resets the counts, ready to add a bunch of new samples for a new estimate...
   void Reset()
   {
    for (int s=0;s<fliLen;s++) power[s] = 0;
   }

  // Given a flag list index indicating which counts are valid and a set of counts indicating the sample counts drawn from the unknown multinomial. Updates the model accordingly...
   void Add(int fli, const int * counts)
   {
    int total = 0;
    for (int f=0;f<flagLen;f++)
    {
     if (fia[fli*flagLen + f]!=0)
     {
      power[f] += counts[f];
      total += counts[f];
     }
    }
    power[fli] -= total + 1;
   }

  // An alternate add method - adds the return value of Power(), allowing the combining of samples stored in seperate SMP objects...
   void Add(const int * pow)
   {
    for (int s=0;s<fliLen;s++) power[s] += pow[s];
   }

  // For incase you have a power vector as a numpy array...
   void Add(PyArrayObject * pow)
   {
    for (int s=0;s<fliLen;s++) power[s] += Int1D(pow,s);
   }


  // These return the dimensions of the entities...
   int FlagSize() {return flagLen;}
   int FlagIndexSize() {return fliLen;}

  // These return info about the prior...
   float * GetPriorMN() {return priorMN;}
   float GetPriorConc() {return priorConc;}

  // Returns the power vector - can be used to combine SMP objects, assuming the same fia...
   const int * Power() {return power;}


  // Calculates the mean of the multinomial drawn from the prior - has lots of sexy optimisations to make it fast. out must be of flagSize() and will have the estimate of the mean written into it...
   void Mean(float * out)
   {
    // First calculate the log probability of each sample, including the prior, storing into samTemp...
     float maxVal = -1e100;
     for (int a=0;a<samLen;a++)
     {
      samTemp[a] = 0.0;

      // Prior...
       for (int f=0;f<flagLen;f++)
       {
        samTemp[a] += priorConc * priorMN[f] * sam[a*fliLen + f];
       }

      // Informaiton provided by samples...
       for (int s=0;s<fliLen;s++)
       {
        samTemp[a] += power[s] * sam[a*fliLen + s];
       }

      // Keep the maximum, for the next bit...
       if (samTemp[a]>maxVal) maxVal = samTemp[a];
     }


    // Convert samTemp into an array of weights that sum to one - done in a numerically stable way, as the logs will represent extremelly small probabilities...
     float sum = 0.0;
     for (int a=0;a<samLen;a++)
     {
      samTemp[a] = exp(samTemp[a] - maxVal);
      sum += samTemp[a];
     }

     for (int a=0;a<samLen;a++) samTemp[a] /= sum;


    // Calculate the mean by suming the weights multiplied by the sample draws into the output...
     for (int f=0;f<flagLen;f++) out[f] = 0.0;
     
     for (int a=0;a<samLen;a++)
     {
      for (int f=0;f<flagLen;f++)
      {
       out[f] += samTemp[a] * samPos[a*flagLen + f];
      }
     }
   }


 private:
  int flagLen; // Length of each flag list.
  int fliLen; // Number of flag lists in system.
  unsigned char * fia; // Array indexed [fli * flagLen + flagIndex] of {0,1} indicating inclusion in each flag list.

  float * priorMN; // Multinomial of prior.
  float priorConc; // Concentration of prior.

  int samLen; // Number of samples used when sampling the mean.
  float * sam; // Array indexed by [sam * fliLen + fli], giving the log of each sum of terms of a draw from Dirichlet(1,...,1).
  float * samPos; // Array indexed by [sam * flagLen + flag], giving the not-log of the above for the single flag entries.
  float * samTemp; // Temporary of length samLen.

  int * power; // Count array, indexed by fli, of power terms for current distribution.

  float * temp; // Temporary array of length flagLen.
};

#endif
"""
