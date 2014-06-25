// Copyright 2013 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



#include "mult.h"

#include "kernels.h"
#include "philox.h"
#include "bessel.h"

#include <stdlib.h>



void MultCache_new(MultCache * self)
{
 self->max_dims = 0;
 self->max_terms = 0;
 
 self->temp_dims1 = NULL;
 self->temp_dims2 = NULL;
 self->temp_terms1 = NULL;
 self->temp_terms2 = NULL;
 
 self->scaled = NULL;
 self->fv = NULL;
 self->scale = NULL;
 
 self->rng = NULL;
 
 self->gibbs_samples = 1;
 self->mci_samples = 1000;
 self->mh_proposals = 8;
}


void MultCache_delete(MultCache * self)
{
 free(self->temp_dims1);
 free(self->temp_dims2);
 free(self->temp_terms1);
 free(self->temp_terms2);
 
 free(self->scaled);
 free(self->fv);
 free(self->scale);
}


void MultCache_ensure(MultCache * self, int dims, int terms)
{
 if (self->max_dims<dims)
 {
  self->max_dims = dims;
  self->temp_dims1 = (float*)realloc(self->temp_dims1, self->max_dims * sizeof(float));
  self->temp_dims2 = (float*)realloc(self->temp_dims2, self->max_dims * sizeof(float));
  
  self->scaled = (float*)realloc(self->scaled, self->max_dims * sizeof(float));
 }
 
 if (self->max_terms<terms)
 {
  self->max_terms = terms;
  self->temp_terms1 = (float*)realloc(self->temp_terms1, self->max_terms * sizeof(float));
  self->temp_terms2 = (float*)realloc(self->temp_terms2, self->max_terms * sizeof(float));
  
  self->fv = (const float**)realloc(self->fv, self->max_terms * sizeof(float*));
  self->scale = (const float**)realloc(self->scale, self->max_terms * sizeof(float*));
 }
}



float mult_area_mci(const Kernel * kernel, KernelConfig config, int dims, int terms, const float ** fv, const float ** scale, MultCache * cache)
{
 MultCache_ensure(cache, dims, terms);
 
 float * draw = cache->temp_dims1;
 float * draw_s = cache->temp_dims2;
 int samples = cache->mci_samples;
 
 float norm = kernel->norm(dims, config);
 float ret = 0.0;
 
 int s;
 for (s=0; s<samples; s++)
 {
  // Draw from the first distribution, convert to global space...
   kernel->draw(dims, config, cache->rng, fv[0], draw);
   
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
   ret += (prob - ret) / (s + 1);
 }
  
 return ret;
}



float mult_area_gaussian(int dims, int terms, const float ** fv, const float ** scale, MultCache * cache)
{
 MultCache_ensure(cache, dims, terms);
 
 // We have a set of feature vectors that represent the mean of each Gaussian multiplied by its inverse standard deviation, plus the inverse standard deviation (scale)... which is strange. Regardless, the maths to obtain the Gaussian that is the multiple of them is easy enough given that observation, as scale squared gives us the diagonal precison matrix required...
  float * fv_mult = cache->temp_dims1;
  float * scale_mult = cache->temp_dims2;
  
  // Zero out accumilators...
   int i;
   for (i=0; i<dims; i++)
   {
    fv_mult[i] = 0.0; 
    scale_mult[i] = 0.0;
   }
  
  // Sum in...
   int j;
   for (j=0; j<terms; j++)
   {
    for (i=0; i<dims; i++)
    {
     float s = scale[j][i];
     fv_mult[i] += s * fv[j][i];
     scale_mult[i] += s * s;
    }
   }
  
  // Fiddle to get fv_mult as the actual mean (not multiplied by inverse sd) and scale_mult as the inverse standard deviation...
   for (i=0; i<dims; i++)
   {
    fv_mult[i] /= scale_mult[i];
    scale_mult[i] = sqrt(scale_mult[i]);
   }
  
 // We have the parameters of the multiplied Gaussian, with weight 1 - we need to calculate its relative scale compared to the actual multiplication of the Gaussians at a single point to obtain the remaning volume - we use the mean of the multiplied Gaussian, as that should be numerically safe...
  const float wibble = pow(2.0 * M_PI, -0.5*dims);
  float norm;
  float half_exp_me;
  
  // Put in the actual multiplied value...
   norm = 1.0 / wibble;
   for (i=0; i<dims; i++) norm /= scale_mult[i];
   half_exp_me = 0.0;
   
  // Divide through by the input terms...
   for (j=0; j<terms; j++)
   {
    norm *= wibble;
    for (i=0; i<dims; i++)
    {
     norm *= scale[j][i];
     float offset = (fv_mult[i] * scale[j][i]) - fv[j][i];
     half_exp_me -= offset * offset;
    }
   }
  
 return exp(0.5 * half_exp_me - log(norm));
}



float mult_area_fisher(float conc, float log_norm, int dims, int terms, const float ** fv, const float ** scale, MultCache * cache)
{
 MultCache_ensure(cache, dims, terms);
 
 // Calculate the direction and concentration of the multiplication of the terms - simple vector addition followed by normalisation at the end basically...
  float * dir = cache->temp_dims1;
  
  int i;
  for (i=0; i<dims; i++) dir[i] = 0.0;

  int j;
  for (j=0; j<terms; j++)
  {
   for (i=0; i<dims; i++) dir[i] += conc * fv[j][i];
  }
  
  float mult_conc = 0.0;
  for (i=0; i<dims; i++) mult_conc += dir[i] * dir[i];
  mult_conc = sqrt(mult_conc);
  
  for (i=0; i<dims; i++) dir[i] /= mult_conc;
  
 // Choose a point - the mode of the multiplied distributions (dir) is safe - and work out the ratio between the area one value and multiplication value, as this gives us the area under the multiplication of the distributions...
  float exp_me = 0.0;
  
  // Put in the value of the normalised distribution...
   exp_me -= mult_conc;
   exp_me -= (0.5 * dims - 1) * log(mult_conc);
   exp_me += (0.5 * dims) * log(2 * M_PI);
   exp_me += LogModBesselFirst(dims-2, mult_conc, 1e-6, 1024);
   
  // Loop through and divide by each Fisher in turn...
   for (j=0; j<terms; j++)
   {
    float dot = 0.0;
    for (i=0; i<dims; i++) dot += dir[i] * fv[j][i];
    exp_me += conc * dot + log_norm;
   }
 
 return exp(exp_me);
}



int mult_draw_mh(const Kernel * kernel, KernelConfig config, int dims, int terms, const float ** fv, const float ** scale, float * out, MultCache * cache)
{
 MultCache_ensure(cache, dims, terms);
 
 float * proposal = cache->temp_dims1;
 float * p_temp = cache->temp_dims2;
 float * out_prob = cache->temp_terms1;
 float * proposal_prob = cache->temp_terms1;
 int proposals = cache->mh_proposals * terms;
 
 int i, j, k;
 int accepts = 0;
 const float norm = 1.0; //kernel->norm(dims, config); // Always cancels out.
 
 // Create an initial proposal - just draw from the first distribution, scale, and fill in the probability values in out_prob...
  kernel->draw(dims, config, cache->rng, fv[0], out);
  
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
     kernel->draw(dims, config, cache->rng, fv[term], proposal);
     
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
     threshold = PhiloxRNG_uniform(cache->rng);
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



void mult_draw_gaussian(int dims, int terms, const float ** fv, const float ** scale, float * out, MultCache * cache, int mode)
{
 MultCache_ensure(cache, dims, terms);
 
 // We have a set of feature vectors that represent the mean of each Gaussian multiplied by its inverse standard deviation, plus the inverse standard deviation (scale)... which is strange. Regardless, the maths to obtain the Gaussian that is the multiple of them is easy enough given that observation, as scale squared gives us the diagonal precison matrix required...
  float * fv_mult = cache->temp_dims1;
  float * scale_mult = cache->temp_dims2;
  
  // Zero out accumilators...
   int i;
   for (i=0; i<dims; i++)
   {
    fv_mult[i] = 0.0; 
    scale_mult[i] = 0.0;
   }
  
  // Sum in...
   int j;
   for (j=0; j<terms; j++)
   {
    for (i=0; i<dims; i++)
    {
     float s = scale[j][i];
     fv_mult[i] += s * fv[j][i];
     scale_mult[i] += s * s;
    }
   }
   
 // If we are in wrong mode output the mean...
  if (mode!=0)
  {
   for (i=0; i<dims; i++) out[i] = fv_mult[i] / scale_mult[i];
   return;
  }
  
 // Fiddle to get the usual format - mean multiplied by inverse sd and inverse sd...
  for (i=0; i<dims; i++)
  {
   scale_mult[i] = sqrt(scale_mult[i]);
   fv_mult[i] /= scale_mult[i];
  }

 // Do the proper thing - draw from the distribution...
  Gaussian.draw(dims, NULL, cache->rng, fv_mult, out);
  
  for (i=0; i<dims; i++) out[i] /= scale_mult[i];
}



void mult(const Kernel * kernel, KernelConfig config, int terms, Spatial * spatials, float * out, MultCache * cache, int * index, float * prob, float quality, int fake)
{
 // Make sure the cache is large enough, and will not need to be resized at all...
  int dims = DataMatrix_features(Spatial_dm(spatials[0]));
  MultCache_ensure(cache, dims, terms);
  
 // Various assorted values we need...
  int i;
  int samples = cache->gibbs_samples * terms;
  
  float range = 2.0 * kernel->range(dims, config, quality);
  
 // Store the scales into the cache...
  int t;
  for (t=0; t<terms; t++)
  {
   cache->scale[t] = Spatial_dm(spatials[t])->mult;
  }
  
 // Randomly draw our starting point from the first distribution - this is a straight draw...
   
  i = DataMatrix_draw(Spatial_dm(spatials[0]), cache->rng);
  cache->fv[0] = DataMatrix_fv(Spatial_dm(spatials[0]), i, NULL);
 
 // Iterate and sample in turn, noting that the first pass is rigged to incrimentally initialise...
  int s;
  for (s=0; s<samples; s++)
  {
   for (t=0; t<terms; t++)
   {
    if ((s==0)&&(t==0)) continue;
    
    // Work out the probability of selecting each item within range, stored cumulatively - optimise selection using one other sample position to avoid considering samples that are too far away...
     int other = (t+terms-1) % terms;
     for (i=0; i<dims; i++) cache->scaled[i] = cache->fv[other][i] * (cache->scale[t][i] / cache->scale[other][i]);
     
     int pos = 0;
     Spatial_start(spatials[t], cache->scaled, range);
     while (1)
     {
      int targ = Spatial_next(spatials[t]);
      if (targ<0) break;
      index[pos] = targ;
      
      float weight;
      cache->fv[t] = DataMatrix_fv(Spatial_dm(spatials[t]), index[pos], &weight);
      prob[pos] = weight * kernel->mult_mass(dims, config, (s==0)?(t+1):terms, cache->fv, cache->scale, cache);
      if (pos!=0) prob[pos] += prob[pos-1];
      pos += 1;
     }
     
     if (pos==0)
     {
      // Problem - no neighbours found - almost certainly means its gone sideways - attempt to recover by putting something random in...
       i = DataMatrix_draw(Spatial_dm(spatials[t]), cache->rng);
       cache->fv[t] = DataMatrix_fv(Spatial_dm(spatials[t]), i, NULL);
  
      continue;
     }

    // Draw a uniform...
     float loc = prob[pos-1] * PhiloxRNG_uniform(cache->rng);
     
    // Go through and select the item that matches with the uniform - binary search...
     int low = 0;
     int high = pos - 1;
     
     while (low<high)
     {
      // Select half way point...
       int mid = (low + high) / 2;
        
      // Head for the right half...
       if (loc<prob[mid]) high = mid;
                     else low = mid + 1;
     }
    
    // Record it...
     cache->fv[t] = DataMatrix_fv(Spatial_dm(spatials[t]), index[low], NULL);
   }
  }
  
 // For the final state make a draw from the selected configuration - a straight call...
  kernel->mult_draw(dims, config, terms, cache->fv, cache->scale, out, cache, fake);
}



// Dummy function, to make a warning go away because it was annoying me...
void MultModule_IgnoreMe(void)
{
 import_array();  
}
