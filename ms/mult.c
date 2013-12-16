// Copyright 2013 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



#include "mult.h"



float mult_area_mci(const Kernel * kernel, KernelConfig config, int dims, int terms, const float ** fv, const float ** scale, float * temp1, float * temp2, int samples, const unsigned int index[2])
{
 float norm = kernel->norm(dims, config);
 
 float ret = 0.0;
 
 unsigned int rng[3];
 rng[0] = index[0];
 rng[1] = index[1];
 
 for (rng[2]=0; rng[2]<samples; rng[2]++)
 {
  // Draw from the first distribution, convert to global space...
   kernel->draw(dims, config, rng, fv[0], temp1);
   
   int i;
   for (i=0; i<dims; i++) temp1[i] /= scale[0][i];
   
  // Calculate its value, by multiplying with the probabilities of the remaining distributions - do everything in the global 'without scale' space for consistancy...
   float prob = 1.0;
   
   int j;
   for (j=1; j<terms; j++)
   {
    // Convert to local space and offset...
     for (i=0; i<dims; i++) temp2[i] = temp1[i] * scale[j][i] - fv[j][i];
     
    // Multiply in the base normalised probability...
     prob *= norm * kernel->weight(dims, config, temp2);
     
    // Factor in scaling effects...
     for (i=0; i<dims; i++) prob *= scale[j][i];
   }
   
  // Update ret with an incrimental average...
   ret += (prob - ret) / (rng[2] + 1);
 }
  
 return ret;
}
