// Copyright 2013 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



#include "bessel.h"

#include <math.h>



// Giant cache of the logs of the integers and half integers, to accelerate the below...
int log_array_size;
float * log_array_data;

// Given the maximum desired value * 2, so the halves are integers, this returns an array indexed by value * 2 to get log(value)...
const float * LogHalfArray(int max)
{
 int size = max+1;
 if (size>log_array_size)
 {
  log_array_data = (float*)realloc(log_array_data, size*sizeof(float));
  
  if (log_array_size==0)
  {
   log_array_data[0] = 1e-32;
   log_array_size = 1;
  }
  
  int i;
  for (i=log_array_size; i<size; i++)
  {
   log_array_data[i] = log(0.5 * i);
  }
  
  log_array_size = size;
 }
  
 return log_array_data;
}



// Modified bessel function of teh first kind. I truly hate this function...
float ModBesselFirst(int orderX2, float x, float accuracy, int limit)
{
 // Various simple things...
  float halfX = 0.5 * x;
  float order = 0.5 * orderX2;
  float mainMult = halfX * halfX;

 // Calculate the initial summand, for r==0, putting it straight into ret...  
  float ret = 1.0;
  {
   float temp = 0.0;
   if ((orderX2%2)==1)
   {
    ret = sqrt(2) * sqrt(x) / sqrt(M_PI);
    temp = 0.5;
   }

   int p;
   for (p=1; p<=orderX2/2; p++)
   {
    ret *= halfX;
    ret /= p + temp;
   }
  }

 // Iterate through following summands, calculating each with the aid of the previous...
  float summand = ret;
  int r;
  for (r=1; r<limit; r++)
  {
   summand *= mainMult;
   summand /= order + r;
   summand /= r;
   float ret2 = ret + summand;
   if (isfinite(ret2)==0) break; // If this fires the function hasn't worked, but at least you get a finite answer.
   ret = ret2;

   if (summand<accuracy) break;
  }

 return ret;
}



float LogModBesselFirst(int orderX2, float x, float accuracy, int limit)
{
 if (x<1e-12)
 {
  if (orderX2==0) return 0.0;
             else return -1e32;
 }
 static const int block_size = 16;
 
 // Basic preperation...
  accuracy = log(accuracy);
  const float * log_hi = LogHalfArray(limit + block_size - 1 + orderX2 - 1);
 
 // Calculate values that will get reused a lot...
  float log_half_x = log(0.5 * x);
  float log_main_mult = 2.0 * log_half_x;

 // Calculate the initial summand, for r==0, putting it straight into ret...  
  float log_ret = 0.0;
  {
   char temp = 0;
   if ((orderX2%2)==1)
   {
    log_ret = 0.5 * (log_hi[4] + log(x/M_PI));
    temp = 1;
   }

   int p;
   for (p=1; p<=orderX2/2; p++)
   {
    log_ret += log_half_x;
    log_ret -= log_hi[p*2 + temp];
   }
  }

 // Iterate through following summands, calculating each with the aid of the previous...
  float log_summand = log_ret;
  int r;
  
  float block[block_size];
  
  for (r=1; r<limit; )
  {
   int s;
   float max_bs = log_ret;
   for (s=0; s<block_size; s++, r++)
   {
    log_summand += log_main_mult;
    log_summand -= log_hi[2*r + orderX2];
    log_summand -= log_hi[2*r];
    
    block[s] = log_summand;
    if (log_summand>max_bs) max_bs = log_summand;
   }
   
   float total = exp(log_ret - max_bs);
   for (s=0; s<block_size; s++)
   {
    total += exp(block[s] - max_bs);
   }
   log_ret = max_bs + log(total);

   if ((log_summand<accuracy)&&(block[block_size-2]>=log_summand)) break;
  }
  
 return log_ret;
}



float LogModBesselFirstAlt(int orderX2, float x, float accuracy, int limit)
{
 // Special case problem values...
  if (orderX2==0) return LogModBesselFirst(orderX2, x, accuracy, limit);
  if (x<1e-12) return -1e32;
 
 accuracy = log(accuracy);
 
 // Create the very first term, set ret to be it...
  float term = x - 0.5 * log(2.0 * M_PI * x) - LogGamma(orderX2+1);

  float inc_gam = LogLowerIncompleteGamma(orderX2 + 1, 2.0 * x);
  term += inc_gam;
  int sign = 1;
  
  float ret = term;

 // Keep summing in terms until the desired accuracy is reached, or we obtain a term with zero in, meaning we have obtained 100% accuracy and can stop...
  int n;
  float log2x = log(2.0 * x);
  for (n=1; n<limit; n++)
  {
   // Move to next inc_gam, factoring it into the term...
    term -= inc_gam;
   
    float smo = 0.5 * (orderX2 + 2*n - 1);
    inc_gam += log(smo);
    inc_gam += log(1.0 - exp(smo * log2x - 2.0 * x - inc_gam));
    
    term += inc_gam;
    
   // Update term...
    int mult2X = 1 - orderX2 + 2 * (n - 1);
    if (mult2X==0) break; // Term has zero in - this term and all further add nil.
    
    if (mult2X<0)
    {
     sign *= -1;
     mult2X = -mult2X;
    }
    
    term += log(0.5 * mult2X);
    term -= log(n) + log2x;

   // Add in or subtract in term - we can always assume its less than the previous...
    ret += log(1.0 + sign * exp(term - ret));
   
   // If the accuracy is high enough, terminate...
    if (term<accuracy) break;
  }
 
 // Return...
  return ret;
}



float LogGamma(int x2)
{
 float ret = 0.0;
 int i = 4;
 
 // Deal with the half... 
  if ((x2&1)==1)
  {
   i = 1;
   ret = 0.5 * log(M_PI);
  }
  
 // Incrimentally do the rest...
  for (; i<x2; i+=2)
  {
   ret += log(0.5*i);
  }
 
 return ret; 
}



float ERF(float x)
{
 float t = 1.0 / (1.0 + 0.5 * fabs(x));
 float t2 = t * t;
 float t3 = t2 * t;
 float t4 = t2 * t2;
 float t5 = t2 * t3;
 float t6 = t3 * t3;
 float t7 = t3 * t4;
 float t8 = t4 * t4;
 float t9 = t3 * t6;
 
 float inner = -x * x - 1.26551223 + 1.00002368 * t + 0.37409196 * t2 + 0.09678418 * t3 - 0.18628806 * t4 + 0.27886807 * t5 - 1.13520398 * t6 + 1.48851587 * t7 - 0.82215223 * t8 + 0.17087277 * t9;
 
 if (x>=0.0) return 1.0 - exp(inner + log(t));
        else return exp(inner + log(t)) - 1.0;
}



float LogLowerIncompleteGamma(int x2, float limit)
{
 float ret;
 
 // Zero case...
  if (x2==0) return 0.0 / 0.0; // Not defined.
 
 // Handle the half/integer component...
  int i;
  if (x2%2==0)
  {
   // Integer... 
    ret = log(1.0 - exp(-limit));
    i = 2;
  }
  else
  {
   // Half...
    ret = 0.5 * log(M_PI) + log(ERF(sqrt(limit)));
    i = 1;
  }

 // Iterate to upgrade it to the actual requested value via the recursive relationship between offsets of 1...
  float log_limit = log(limit);
  for (;i<x2; i+=2)
  {
   float smo = 0.5 * i;
   ret += log(smo);
   ret += log(1.0 - exp(smo * log_limit - limit - ret));
  }
 
 return ret;
}



void FreeBesselMemory(PyObject * ignored)
{
 free(log_array_data);
}
