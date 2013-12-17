// Copyright 2013 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



#include "mult.h"

#include "kernels.h"
#include "philox.h"

#include <stdlib.h>



void MultCache_new(MultCache * self)
{
 self->max_dims = 0;
 self->max_terms = 0;
 
 self->temp_dims1 = NULL;
 self->temp_dims2 = NULL;
 self->temp_terms1 = NULL;
 self->temp_terms2 = NULL;
 
 self->rng_index[0] = 0;
 self->rng_index[1] = 0;
 
 self->gibbs_samples = -1;
 self->mci_samples = -1;
 self->mh_proposals = -1;
}


void MultCache_delete(MultCache * self)
{
 free(self->temp_dims1);
 free(self->temp_dims2);
 free(self->temp_terms1);
 free(self->temp_terms2);
}


void MultCache_ensure(MultCache * self, int dims, int terms)
{
 if (self->max_dims<dims)
 {
  self->max_dims = dims;
  self->temp_dims1 = realloc(self->temp_dims1, self->max_dims * sizeof(float));
  self->temp_dims2 = realloc(self->temp_dims2, self->max_dims * sizeof(float));
 }
 
 if (self->max_terms<terms)
 {
  self->max_terms = terms;
  self->temp_terms1 = realloc(self->temp_terms1, self->max_terms * sizeof(float));
  self->temp_terms2 = realloc(self->temp_terms2, self->max_terms * sizeof(float));
 }
}



float mult_area_mci(const Kernel * kernel, KernelConfig config, int dims, int terms, const float ** fv, const float ** scale, MultCache * cache)
{
 MultCache_ensure(cache, dims, terms);
 
 float * draw = cache->temp_dims1;
 float * draw_s = cache->temp_dims2;
 int samples = cache->mci_samples;
 if (samples<1) samples = 1000;
 
 float norm = kernel->norm(dims, config);
 float ret = 0.0;
 
 unsigned int rng[3];
 rng[0] = cache->rng_index[0];
 rng[1] = cache->rng_index[1];
 
 for (rng[2]=0; rng[2]<samples; rng[2]++)
 {
  // Draw from the first distribution, convert to global space...
   kernel->draw(dims, config, rng, fv[0], draw);
   
   int i;
   for (i=0; i<dims; i++) draw[i] /= scale[0][i];
   
  // Calculate its value, by multiplying with the probabilities of the remaining distributions - do everything in the global 'without scale' space for consistancy...
   float prob = 1.0;
   
   int j;
   for (j=1; j<terms; j++)
   {
    // Convert to local space and offset...
     for (i=0; i<dims; i++) draw_s[i] = draw[i] * scale[j][i] - fv[j][i];
     
    // Multiply in the base normalised probability...
     prob *= norm * kernel->weight(dims, config, draw_s);
     
    // Factor in scaling effects...
     for (i=0; i<dims; i++) prob *= scale[j][i];
   }
   
  // Update ret with an incrimental average...
   ret += (prob - ret) / (rng[2] + 1);
 }
  
 return ret;
}



int mult_draw_mh(const Kernel * kernel, KernelConfig config, int dims, int terms, const float ** fv, const float ** scale, float * out, MultCache * cache)
{
 MultCache_ensure(cache, dims, terms);
 
 float * proposal = cache->temp_dims1;
 float * p_temp = cache->temp_dims2;
 float * out_prob = cache->temp_terms1;
 float * proposal_prob = cache->temp_terms1;
 int proposals = cache->mh_proposals;
 if (proposals<1) proposals = terms * 8;
 
 int i, j, k;
 int accepts = 0;
 const float norm = 1.0; //kernel->norm(dims, config); // Always cancels out.
 
 unsigned int rng[3];
 rng[0] = cache->rng_index[0];
 rng[1] = cache->rng_index[1];
 rng[2] = 0;
 
 unsigned int ind[4];
 int ind_use = 4;
 
 // Create an initial proposal - just draw from the first distribution, scale, and fill in the probability values in out_prob...
  kernel->draw(dims, config, rng, fv[0], out);
  rng[2] += 1;
  for (j=0; j<dims; j++) out[j] /= scale[0][j];
  
  for (k=0; k<terms; k++)
  {
   out_prob[k] = norm;
   for (j=0; j<dims; j++)
   {
    //out_prob[k] *= scale[k][j]; // Always cancels out.
    p_temp[j] = out[j] * scale[k][j] - fv[k][j];
   }
   out_prob[k] *= kernel->weight(dims, config, p_temp);
  }
  
 // Loop through creating proposals and accepting/rejecting them - by the end we should have a reasonable draw in out...
  int term = 0; // Which term to draw the next proposal from - cycle through them as one might be extra pointy and have a much higher accept rate than the others. Could get all complex and one armed bandit-y, but just doing this for now.
  for (i=0; i<proposals; i++)
  {
   // Select the term we are going to draw from...
    term = (term+1) % terms;
    
   // Draw from that term, converting the draw to global space...
     kernel->draw(dims, config, rng, fv[term], proposal);
     rng[2] += 1;
     for (j=0; j<dims; j++) proposal[j] /= scale[term][j];
    
   // Go through and calculate the probabilities...
    for (k=0; k<terms; k++)
    {
     proposal_prob[k] = norm;
     for (j=0; j<dims; j++)
     {
      //proposal_prob[k] *= scale[k][j]; // Always cancels out.
      p_temp[j] = proposal[j] * scale[k][j] - fv[k][j];
     }
     proposal_prob[k] *= kernel->weight(dims, config, p_temp);
    }

   // Calculate the accept probability...
    float ap = 1.0;
    for (k=0; k<terms; k++)
    {
     if (k!=term)
     {
      ap *= proposal_prob[k] / out_prob[k];
     }
    }
    
   // Decide if we are going to accept, if so copy it into the output...
    float threshold = 0.0;
    if (ap<1.0)
    {
     // Do a uniform draw for the threshold, but only if needed...
      if (ind_use>3)
      {
       ind[0] = cache->rng_index[0];
       ind[1] = cache->rng_index[1];
       ind[2] = rng[2];
       ind[3] = 0x80808080;
      
       philox(ind);
       rng[2] += 1;
       ind_use = 0;
      }
      
      threshold = uniform(ind[ind_use]);
      ind_use += 1;
    }
    
    if (ap>threshold)
    {
     // Accept - copy into the out variable, including the term probabilities...
      for (j=0; j<dims; j++) out[j] = proposal[j];
      for (k=0; k<terms; k++) out_prob[k] = proposal_prob[k];
      ++accepts;
    }
  }
 
 return accepts;
}
