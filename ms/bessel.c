// Copyright 2013 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



#include "bessel.h"

#include <math.h>



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
 accuracy = log(accuracy);
 
 // Various simple things...
  float log_half_x = log(0.5 * x);
  float order = 0.5 * orderX2;
  float log_main_mult = 2.0 * log_half_x;

 // Calculate the initial summand, for r==0, putting it straight into ret...  
  float log_ret = 0.0;
  {
   float temp = 0.0;
   if ((orderX2%2)==1)
   {
    log_ret = 0.5 * (log(2) + log(x) - log(M_PI));
    temp = 0.5;
   }

   int p;
   for (p=1; p<=orderX2/2; p++)
   {
    log_ret += log_half_x;
    log_ret -= log(p + temp);
   }
  }

 // Iterate through following summands, calculating each with the aid of the previous...
  float log_summand = log_ret;
  int r;
  for (r=1; r<limit; r++)
  {
   log_summand += log_main_mult;
   log_summand -= log(order + r);
   log_summand -= log(r);
   
   if (log_ret>log_summand) log_ret += log(1.0 + exp(log_summand - log_ret));
   else log_ret = log_summand + log(1.0 + exp(log_ret - log_summand));

   if (log_summand<accuracy) break;
  }

 return log_ret;
}
