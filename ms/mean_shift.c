// Copyright 2013 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



#include "mean_shift.h"

#include "eigen.h"



float calc_weight(DataMatrix * dm)
{
 float ret = 0.0;
 
 int i;
 int exemplars = DataMatrix_exemplars(dm);
 
 for (i=0; i<exemplars; i++)
 {
  float w;
  DataMatrix_fv(dm, i, &w);
  ret += w;
 }
  
 return ret;
}



float calc_norm(DataMatrix * dm, const Kernel * kernel, KernelConfig config, float weight)
{
 int features = DataMatrix_features(dm);
 float ret = kernel->norm(features, config) / weight;
 
 int i;
 for (i=0; i<features; i++) ret *= dm->mult[i];
 
 return ret;
}



float prob(Spatial spatial, const Kernel * kernel, KernelConfig config, const float * fv, float norm, float quality)
{
 // Extract a bunch of things...
  DataMatrix * dm = Spatial_dm(spatial);
  
  int feats = DataMatrix_features(dm);
  float range = kernel->range(feats, config, quality);
  
 // Loop and sum the return value...
  float ret = 0.0;
  Spatial_start(spatial, fv, range);
  
  while (1)
  {
   int targ = Spatial_next(spatial);
   if (targ<0) break;
   
   float w;
   float * loc = DataMatrix_fv(dm, targ, &w);

   kernel->to_offset(feats, config, loc, fv);
   w *= kernel->weight(feats, config, loc);
   
   w *= norm;
   ret += w;
  }
  
 return ret;
}



void draw(DataMatrix * dm, const Kernel * kernel, KernelConfig config, PhiloxRNG * rng, float * out)
{
 int feats = DataMatrix_features(dm);
 
 // First draw the exemplar we are to play with...
  int exemplar = DataMatrix_draw(dm, rng);
  
 // Get the exemplars location...
  float * fv = DataMatrix_fv(dm, exemplar, NULL);
  
 // Draw from the kernel in question...
  kernel->draw(feats, config, rng, fv, out);
 
 // Convert from scaled space to normal space...
  int i;
  for (i=0; i<feats; i++) out[i] /= dm->mult[i];
}



float loo_nll(Spatial spatial, const Kernel * kernel, KernelConfig config, float norm, float quality, float limit)
{
 // Extract a bunch of things...
  DataMatrix * dm = Spatial_dm(spatial);
  
  int exemplars = DataMatrix_exemplars(dm);
  int features = DataMatrix_features(dm);
  float range = kernel->range(features, config, quality);
  
 // Loop and do each exemplar in the data set in turn... 
  float ret = 0.0;
  
  float * fvi = (float*)malloc(features * sizeof(float));
  
  int i, j;
  for (i=0; i<exemplars; i++)
  {
   // Get exemplar i, so we can play with it...
    float wi;
    float * fv = DataMatrix_fv(dm, i, &wi);
    for (j=0; j<features; j++) fvi[j] = fv[j];
    
   // Calculate the probability of exemplar i, ignoring entry i...
    float prob = 0.0;
    Spatial_start(spatial, fvi, range);
    
    while (1)
    {
     int targ = Spatial_next(spatial);
     if (targ<0) break;
     if (targ==i) continue; // Skip the one we are currently analysing.
   
     float w;
     float * fv = DataMatrix_fv(dm, targ, &w);

     kernel->to_offset(features, config, fv, fvi);
     w *= kernel->weight(features, config, fv);
   
     w *= norm;
     prob += w;
    }
    
   // Update the return cost...
    if (prob<limit) prob = limit;
    ret -= wi * log(prob);
  }
  
  free(fvi);
  
 return ret;
}



float entropy(Spatial spatial, const Kernel * kernel, KernelConfig config, float norm, float quality)
{
 // Extract a bunch of things...
  DataMatrix * dm = Spatial_dm(spatial);
  
  int exemplars = DataMatrix_exemplars(dm);
  int features = DataMatrix_features(dm);
  float range = kernel->range(features, config, quality);
  
 // Loop and do each exemplar in the data set in turn... 
  float ret = 0.0;
  float samples = 0.0;
  
  float * fvi = (float*)malloc(features * sizeof(float));
  
  int i, j;
  for (i=0; i<exemplars; i++)
  {
   // Get exemplar i, so we can play with it...
    float wi;
    float * fv = DataMatrix_fv(dm, i, &wi);
    for (j=0; j<features; j++) fvi[j] = fv[j];
    
   // Calculate the probability of exemplar i...
    float prob = 0.0;
    Spatial_start(spatial, fvi, range);
    
    while (1)
    {
     int targ = Spatial_next(spatial);
     if (targ<0) break;
   
     float w;
     float * fv = DataMatrix_fv(dm, targ, &w);

     kernel->to_offset(features, config, fv, fvi);
     w *= kernel->weight(features, config, fv);
   
     w *= norm;
     prob += w;
    }
    
   // Update the return cost - incrimental mean...
    samples += wi;
    ret += wi * (log(prob) - ret) / samples;
  }
  
  free(fvi);
  
 return -ret;
}



float kl_divergence(Spatial spatial_p, const Kernel * kernel_p, KernelConfig config_p, float norm_p, float quality_p, Spatial spatial_q, const Kernel * kernel_q, KernelConfig config_q, float norm_q, float quality_q, float limit)
{
 // Extract a bunch of things...
  DataMatrix * dm_p = Spatial_dm(spatial_p);
  DataMatrix * dm_q = Spatial_dm(spatial_q);
  
  int exemplars = DataMatrix_exemplars(dm_p);
  int features = DataMatrix_features(dm_p);
  float range_p = kernel_p->range(features, config_p, quality_p);
  float range_q = kernel_q->range(features, config_q, quality_q);

// Loop and do each exemplar from p in turn... 
  float ret = 0.0;
  float samples = 0.0;
  
  float * fvi = (float*)malloc(features * sizeof(float));
  
  int i, j;
  for (i=0; i<exemplars; i++)
  {
   // Get exemplar i, so we can play with it...
    float wi;
    float * fv = DataMatrix_fv(dm_p, i, &wi);
    for (j=0; j<features; j++) fvi[j] = fv[j];
    
   // Calculate the probability of exemplar i in p...
    float prob_p = 0.0;
    Spatial_start(spatial_p, fvi, range_p);
    
    while (1)
    {
     int targ = Spatial_next(spatial_p);
     if (targ<0) break;
   
     float w;
     float * fv = DataMatrix_fv(dm_p, targ, &w);

     kernel_p->to_offset(features, config_p, fv, fvi);
     w *= kernel_p->weight(features, config_p, fv);
   
     w *= norm_p;
     prob_p += w;
    }
   
   // Calculate the probability of exemplar i in q...
    float prob_q = 0.0;
    Spatial_start(spatial_q, fvi, range_q);
    
    while (1)
    {
     int targ = Spatial_next(spatial_q);
     if (targ<0) break;
   
     float w;
     float * fv = DataMatrix_fv(dm_q, targ, &w);

     kernel_q->to_offset(features, config_q, fv, fvi);
     w *= kernel_q->weight(features, config_q, fv);
   
     w *= norm_q;
     prob_q += w;
    }
    
    if (prob_q<limit) prob_q = limit;
    
   // Update the return cost - incrimental mean...
    samples += wi;
    ret += wi * (log(prob_p / prob_q) - ret) / samples;
  }
  
  free(fvi);
  
 return ret;
}



void mode(Spatial spatial, const Kernel * kernel, KernelConfig config, float * fv, float * temp, float quality, float epsilon, int iter_cap)
{
 // Extract the many things we need... 
  DataMatrix * dm = Spatial_dm(spatial);
  
  int feats = DataMatrix_features(dm);
  float range = kernel->range(feats, config, quality);

 // Loop until convergance...
  float delta = 2.0 * epsilon;
  int iters = 0;
  while ((delta>epsilon)&&(iters<iter_cap))
  {
   // Prepare the temporary for incrimental mean calculation...
    float weight = 0.0;
    int i;
    for (i=0; i<feats; i++) temp[i] = 0.0;
   
   // Iterate all relevant samples, to calculate the mean...
    Spatial_start(spatial, fv, range);
    while (1)
    {
     int targ = Spatial_next(spatial);
     if (targ<0) break;
     
     float w;
     float * loc = DataMatrix_fv(dm, targ, &w);

     kernel->to_offset(feats, config, loc, fv);
     w *= kernel->weight(feats, config, loc);
     
     if (w>1e-6)
     {
      weight += w;
      for (i=0; i<feats; i++) temp[i] += w * (loc[i] - temp[i]) / weight;
     }
    }
   
   // Copy into the fv, calculating delta as well...
    delta = kernel->offset(feats, config, fv, temp);
    
   // We just iterated...
    iters += 1;
  }
}



int mode_merge(Spatial spatial, const Kernel * kernel, KernelConfig config, Balls balls, float * fv, float * temp, float quality, float epsilon, int iter_cap, float merge_range, int check_step)
{
 // Extract some things that we need... 
  DataMatrix * dm = Spatial_dm(spatial);
  int feats = DataMatrix_features(dm);
  float range = kernel->range(feats, config, quality);
  int states = kernel->states(feats, config);
  
 // Converge the provided feature vector, with breaks every check_step-s to find out if its hit a mode or not...
  float delta = 2.0 * epsilon;
  int iters = 0;
  
  int out = -1; // Output
    
  while ((delta>epsilon)&&(iters<iter_cap))
  {
   // Check if we collided with a mode that already exists...
    if ((iters%check_step)==0)
    {
     if (states>1)
     {
      int s;
      for (s=0; s<states; s++)
      {
       out = Balls_within(balls, fv);
       if (out>=0) break;
       kernel->next(feats, config, s, fv);
      }
     }
     else
     {
      out = Balls_within(balls, fv);
     }
       
     if (out>=0) break;
    }
    
   // Prepare the temporary for incrimental mean calculation...
    float weight = 0.0;
    int i;
    for (i=0; i<feats; i++) temp[i] = 0.0;
   
   // Iterate all relevant samples, to calculate the mean...
    Spatial_start(spatial, fv, range);
    while (1)
    {
     int targ = Spatial_next(spatial);
     if (targ<0) break;
     
     float w;
     float * loc = DataMatrix_fv(dm, targ, &w);
       
     kernel->to_offset(feats, config, loc, fv);
     w *= kernel->weight(feats, config, loc);
     
     if (w>1e-6)
     {
      weight += w;
      for (i=0; i<feats; i++) temp[i] += w * (loc[i] - temp[i]) / weight;
     }
    }
   
   // Copy into the fv, calculating delta as well...
    delta = kernel->offset(feats, config, fv, temp);
      
   // We just iterated...
    iters += 1; 
  }
  
 // If it has not merged with an existing mode create a new one to assign it to...
  if (out<0)
  {
   // Check if it got to the point it should merge in the last few iterations...
    if (states>1)
    {
     int s;
     for (s=0; s<states; s++)
     {
      out = Balls_within(balls, fv);
      if (out>=0) break;
      kernel->next(feats, config, s, fv);
     }
    }
    else
    {
     out = Balls_within(balls, fv);
    }
      
   // If not we have no choice but to create a mode...
    if (out<0)
    {
     out = Balls_create(balls, fv, merge_range);
    }
  }
  
 // Return the assigned cluster...  
  return out; 
}



void cluster(Spatial spatial, const Kernel * kernel, KernelConfig config, Balls balls, int * out, float quality, float epsilon, int iter_cap, float ident_dist, float merge_range, int check_step)
{
 // Extract some things that we need... 
  DataMatrix * dm = Spatial_dm(spatial);
  
  int exemplars = DataMatrix_exemplars(dm);
  int feats = DataMatrix_features(dm);
  float range = kernel->range(feats, config, quality);
  int states = kernel->states(feats, config);
 
 // Create some temporary storage...
  float * fv   = (float*)malloc(feats * sizeof(float));
  float * temp = (float*)malloc(feats * sizeof(float));
  int * same_dest = (int*)malloc(exemplars * sizeof(int));
  
 // Set all memeber of the output to -1, to indicate that they are not yet assigned...
  int ei;
  for (ei=0; ei<exemplars; ei++) out[ei] = -1;
 
 // Loop and process each exemplar in turn...
  for (ei=0; ei<exemplars; ei++)
  {
   if (out[ei]>=0) continue; // Already been processed - skip.
   
   // Get the exemplar, noting that we do not process exempars if their weight is 0...
    float w;
    float * loc = DataMatrix_fv(dm, ei, &w);
    if (w<1e-3) continue; 

    int i;
    for (i=0; i<feats; i++) fv[i] = loc[i];
    
   // Converge it, with breaks every check_step-s to find out if its hit a mode or not...
    float delta = 2.0 * epsilon;
    int iters = 0;
    int same_dest_count = 0;
    
    while ((delta>epsilon)&&(iters<iter_cap))
    {
     // Check if there is anyone going to the same destination as us - if so record them to be assigned to the same destination...
      if (ident_dist>1e-6)
      {
       Spatial_start(spatial, fv, ident_dist);
       while (1)
       {
        int targ = Spatial_next(spatial);
        if (targ<0) break;
        
        loc = DataMatrix_fv(dm, targ, NULL);
        float distSqr = 0.0;
        for (i=0; i<feats; i++)
        {
         float delta = loc[i] - fv[i];
         distSqr += delta*delta;
        }
        
        if (distSqr<=ident_dist*ident_dist)
        {
         // We have an exemplar that is close enough - we need to record that it is going to the same destination, noting we are not allowed to record duplicates...
          int store = 1;
          for (i=0; i<same_dest_count; i++)
          {
           if (same_dest[i]==targ)
           {
            store = 0;
            break;
           }
          }
          
          if (store!=0)
          {
           same_dest[same_dest_count] = targ;
           same_dest_count += 1;
          }
        }
       }
      }
     
     // Check if we collided with a mode that already exists...
      if ((iters%check_step)==0)
      {
       if (states>1)
       {
        int s;
        for (s=0; s<states; s++)
        {
         out[ei] = Balls_within(balls, fv);
         if (out[ei]>=0) break;
         kernel->next(feats, config, s, fv);
        }
       }
       else
       {
        out[ei] = Balls_within(balls, fv);
       }
       
       if (out[ei]>=0) break;
      }
     
     // Prepare the temporary for incrimental mean calculation...
      float weight = 0.0;
      for (i=0; i<feats; i++) temp[i] = 0.0;
   
     // Iterate all relevant samples, to calculate the mean...
      Spatial_start(spatial, fv, range);
      while (1)
      {
       int targ = Spatial_next(spatial);
       if (targ<0) break;
       
       loc = DataMatrix_fv(dm, targ, &w);
       
       kernel->to_offset(feats, config, loc, fv);
       w *= kernel->weight(feats, config, loc);
     
       if (w>1e-6)
       {
        weight += w;
        for (i=0; i<feats; i++) temp[i] += w * (loc[i] - temp[i]) / weight;
       }
      }
   
     // Copy into the fv, calculating delta as well...
      delta = kernel->offset(feats, config, fv, temp);
      
     // We just iterated...
      iters += 1; 
    }
   
   // If it has not merged with an existing mode create a new one to assign it to...
    if (out[ei]<0)
    {
     // Check if it got to the point it should merge in the last few iterations...
      if (states>1)
      {
       int s;
       for (s=0; s<states; s++)
       {
        out[ei] = Balls_within(balls, fv);
        if (out[ei]>=0) break;
        kernel->next(feats, config, s, fv);
       }
      }
      else
      {
       out[ei] = Balls_within(balls, fv);
      }
      
     // If not we have no choice but to create a mode...
      if (out[ei]<0)
      {
       out[ei] = Balls_create(balls, fv, merge_range);
      }
    }
    
   // Go through and record its destination for all exemplars that are assumed to go to the same location...
    for (i=0; i<same_dest_count; i++)
    {
     if (out[same_dest[i]]<0) out[same_dest[i]] = out[ei];
    }
  }
  
 // Clean up...
  free(same_dest);
  free(temp);
  free(fv);
}



int assign_cluster(Spatial spatial, const Kernel * kernel, KernelConfig config, Balls balls, float * fv, float * temp, float quality, float epsilon, int iter_cap, int check_step)
{
 // Extract some things that we need... 
  DataMatrix * dm = Spatial_dm(spatial);
  
  int feats = DataMatrix_features(dm);
  float range = kernel->range(feats, config, quality);
  int states = kernel->states(feats, config);

 // Converge the feature vector, regularly checking if its bumped into a ball...
  float delta = 2.0 * epsilon;
  int iters = 0;
    
  while ((delta>epsilon)&&(iters<iter_cap))
  {
   // Check if we collided with a mode that already exists...
    if ((iters%check_step)==0)
    {
     if (states>1)
     {
      int s;
      for (s=0; s<states; s++)
      {
       int ret = Balls_within(balls, fv);
       if (ret>=0) return ret;
       kernel->next(feats, config, s, fv);
      }
     }
     else
     {
      int ret = Balls_within(balls, fv);
      if (ret>=0) return ret;
     }
    }
     
   // Prepare the temporary for incrimental mean calculation...
    float weight = 0.0;
    int i;
    for (i=0; i<feats; i++) temp[i] = 0.0;
   
   // Iterate all relevant samples, to calculate the mean...
    Spatial_start(spatial, fv, range);
    while (1)
    {
     int targ = Spatial_next(spatial);
     if (targ<0) break;
     
     float w;
     float * loc = DataMatrix_fv(dm, targ, &w);

     kernel->to_offset(feats, config, loc, fv);
     w *= kernel->weight(feats, config, loc);
     
     if (w>1e-6)
     {
      weight += w;
      for (i=0; i<feats; i++) temp[i] += w * (loc[i] - temp[i]) / weight;
     }
    }
   
   // Copy into the fv, calculating delta as well...
    delta = kernel->offset(feats, config, fv, temp);
      
   // We just iterated...
    iters += 1; 
  }
  
 if (states>1)
 {
  int s;
  for (s=0; s<states; s++)
  {
   int ret = Balls_within(balls, fv);
   if (ret>=0) return ret;
   kernel->next(feats, config, s, fv);
  }
  return -1;
 }
 else
 {
  return Balls_within(balls, fv);
 }
}



void manifold(Spatial spatial, int degrees, float * fv, float * grad, float * hess, float * eigen_val, float * eigen_vec, float quality, float epsilon, int iter_cap, int always_hessian)
{
 int i, j;
 
 // Extract some things that we need... 
  DataMatrix * dm = Spatial_dm(spatial);
  
  int feats = DataMatrix_features(dm);
  float range = Gaussian.range(feats, NULL, quality);
  float norm = Gaussian.norm(feats, NULL);
  
 // Converge the feature vector, one step at a time...
  int iters = 0;
  float delta = epsilon*2.0;
  
  while ((delta>epsilon)&&(iters<iter_cap))
  {
   int update_hessian = (always_hessian!=0) || (iters==0);
   
   // First calculate the gradiant and hessian at the current location...
    // Initialise parameters...
     float weight = 0.0;
     for (i=0; i<feats; i++) grad[i] = 0.0;
     if (update_hessian)
     {
      for (i=0; i<feats*feats; i++) hess[i] = 0.0;
     }
     
    // Loop all relevant exemplars in the dataset...
     Spatial_start(spatial, fv, range);
     while (1)
     {
      int targ = Spatial_next(spatial);
      if (targ<0) break;
      
      float w;
      float * loc = DataMatrix_fv(dm, targ, &w);
      
      for (i=0; i<feats; i++) loc[i] -= fv[i];
      w *= norm * Gaussian.weight(feats, NULL, loc);
     
      if (w>1e-6)
      {
       weight += w;
       
       // Update the gradient...
        for (i=0; i<feats; i++) grad[i] += w * (loc[i] - grad[i]) / weight;
       
       // Update the hessian...
        if (update_hessian)
        {
         for (j=0; j<feats; j++)
         {
          for (i=0; i<feats; i++)
          {
           int ii = j*feats + i;
          
           float val = loc[i] * loc[j];
           if (i==j) val -= 1.0;
           hess[ii] += w * (val - hess[ii]) / weight;
          }
         }
        }
      }
     }

   // Calculate the eigenvectors of the hessian...
    if (update_hessian)
    {
     symmetric_eigen_abs(feats, hess, eigen_vec, eigen_val);
    }
   
   // Calculate the change, including projecting to the relevant subspace...
   // (Use the eigen_val vector as an intermediate.)
    // Step 1 - includes updating the delta...
     delta = 0.0;
     for (j=0; j<feats-degrees; j++)
     {
      float val = 0.0;
      for (i=0; i<feats; i++)
      {
       val += eigen_vec[i*feats + j] * grad[i]; 
      }
      
      eigen_val[j] = val;
      delta += fabs(val);
     }
    
    // Quick intermission to divide the delta by the length of the gradient, to normalise the convergance metric...
     float div = epsilon*epsilon; // epsilon is to avoid a divide by zero/problems with underflow.
     for (i=0; i<feats; i++) div += grad[i] * grad[i];
     delta /= sqrt(div);
    
    // Step 2 - update the feature vector as we calculate...
     for (i=0; i<feats; i++)
     {
      float val = 0.0;
      for (j=0; j<feats-degrees; j++)
      {
       val += eigen_vec[i*feats + j] * eigen_val[j];
      }
      fv[i] += val;
     }
   
   // To the next iteration...
    iters += 1;
  }
}



// Dummy function, to make a warning go away because it was annoying me...
void MeanShift_IgnoreMe(void)
{
 import_array();  
}
