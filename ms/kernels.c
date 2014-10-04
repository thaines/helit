// Copyright 2013 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



#include "kernels.h"

#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "philox.h"
#include "bessel.h"



// Most kernels don't need a configuration, and hecne use this dummy set of configuration handlers...
KernelConfig Kernel_config_new(int dims, const char * config)
{
 return NULL; 
}

const char * Kernel_config_verify(int dims, const char * config, int * length)
{
 if (length!=NULL) *length = 0;
 return NULL; 
}

void Kernel_config_acquire(KernelConfig config)
{
 // Noop 
}

void Kernel_config_release(KernelConfig config)
{
 // Noop 
}



// Most kernels have the same to offset method, as provided by this implimentation...
void Kernel_to_offset(int dims, KernelConfig config, float * fv, const float * base_fv)
{
 int i;
 for (i=0; i<dims; i++) fv[i] -= base_fv[i];
}

// Most kernels use the same offset method, as provided by this implimentaiton...
float Kernel_offset(int dims, KernelConfig config, float * fv, const float * offset)
{
 int i;
 float delta = 0.0;
 
 for (i=0; i<dims; i++)
 {
  delta += fabs(offset[i]);
  fv[i] += offset[i]; 
 }
 
 return delta;
}

// Most kernels have no states, hence the following two methods...
int Kernel_states(int dims, KernelConfig config)
{
 return 1; 
}

void Kernel_next(int dims, KernelConfig config, int state, float * fv)
{
 // No-op. 
}



// The uniform kernel type...
float Uniform_weight(int dims, KernelConfig config, float * offset)
{
 float dist_sqr = 0.0;
 
 int i;
 for (i=0; i<dims; i++)
 {
  dist_sqr += offset[i] * offset[i];
  if (dist_sqr>1.0) return 0.0;
 }
 
 return 1.0;
}

float Uniform_norm(int dims, KernelConfig config)
{
 if ((dims&1)==0)
 {
  // Even...
   float ret = 1.0;
   
   int i;
   for (i=0;i<(dims/2); i++)
   {
    ret *= (i+1);
    ret /= M_PI;
   }
   
   return ret;
 }
 else
 {
  // Odd...
   float ret = 0.5;
   
   int i;
   for (i=0;i<(dims/2); i++)
   {
    ret *= (2*i+3);
    ret /= 2.0 * M_PI;
   }
   
   return ret;
 }
}

float Uniform_range(int dims, KernelConfig config, float quality)
{
 return 1.0;
}

void Uniform_draw(int dims, KernelConfig config, PhiloxRNG * rng, const float * center, float * out)
{
 // Put a Gaussian into each out, keeping track of the squared length...
  int i;
  float radius = 0.0;
  for (i=0; i<dims; i+=2)
  {
   // Output...
    float * second = (i+1<dims) ? (out+i+1) : NULL;
    out[i] = PhiloxRNG_Gaussian(rng, second);;
    
    radius += out[i] * out[i];
    if (second!=NULL) radius += out[i+1] * out[i+1];
  }
  
 // Convert from squared radius to not-squared radius...
  radius = sqrt(radius);
  
 // Draw the radius we are going to emit; prepare the multiplier...
  radius = pow(PhiloxRNG_uniform(rng), 1.0/dims) / radius;
 
 // Normalise so its at the required distance...
  for (i=0; i<dims; i++)
  {
   out[i] = center[i] + out[i] * radius;
  }
}



float Uniform_mult_mass(int dims, KernelConfig config, int terms, const float ** fv, const float ** scale, MultCache * cache)
{
 return mult_area_mci(&Uniform, config, dims, terms, fv, scale, cache);
}

void Uniform_mult_draw(int dims, KernelConfig config, int terms, const float ** fv, const float ** scale, float * out, MultCache * cache, int fake)
{
 if (fake==2)
 {
  // We can do the average of feature vector positions option... 
   int i, j;
   for (i=0; i<dims; i++) out[i] = 0.0;
   
   for (j=0; j<terms; j++)
   {
    for (i=0; i<dims; i++)
    {
     out[i] += ((fv[j][i] / scale[j][i]) - out[i]) / (j+1);
    }
   }
 }
 else
 {
  // For options 0 and 1 do a proper draw...   
   mult_draw_mh(&Uniform, config, dims, terms, fv, scale, out, cache);
 }
}



const Kernel Uniform =
{
 "uniform",
 "Provides a uniform kernel - all points within the unit hypersphere get a positive constant weight, all of those outside it get zero.",
 NULL,
 Kernel_config_new,
 Kernel_config_verify,
 Kernel_config_acquire,
 Kernel_config_release,
 Uniform_weight,
 Uniform_norm,
 Uniform_range,
 Kernel_to_offset,
 Kernel_offset,
 Uniform_draw,
 Uniform_mult_mass,
 Uniform_mult_draw,
 Kernel_states,
 Kernel_next,
};



// The triangular kernel type...
float Triangular_weight(int dims, KernelConfig config, float * offset)
{
 float dist_sqr = 0.0;
 
 int i;
 for (i=0; i<dims; i++)
 {
  dist_sqr += offset[i] * offset[i];
  if (dist_sqr>1.0) return 0.0;
 }
 
 return 1.0 - sqrt(dist_sqr);
}

float Triangular_norm(int dims, KernelConfig config)
{
 return (dims + 1.0) * Uniform_norm(dims, NULL);
}

float Triangular_range(int dims, KernelConfig config, float quality)
{
 return 1.0;
}

void Triangular_draw(int dims, KernelConfig config, PhiloxRNG * rng, const float * center, float * out)
{
 // Put a Gaussian into each out, keeping track of the squared length...
  int i;
  float radius = 0.0;
  for (i=0; i<dims; i+=2)
  {
   // Output...
    float * second = (i+1<dims) ? (out+i+1) : NULL;
    out[i] = PhiloxRNG_Gaussian(rng, second);
    
    radius += out[i] * out[i];
    if (second!=NULL) radius += out[i+1] * out[i+1];
  }
  
 // Convert from squared radius to not-squared radius...
  radius = sqrt(radius);
  
 // Draw the radius we are going to emit; prepare the multiplier...
  radius = (1.0 - sqrt(1.0 - PhiloxRNG_uniform(rng))) / radius;
 
 // Normalise so its at the required distance...
  for (i=0; i<dims; i++)
  {
   out[i] = center[i] + out[i] * radius;
  }
}



float Triangular_mult_mass(int dims, KernelConfig config, int terms, const float ** fv, const float ** scale, MultCache * cache)
{
 return mult_area_mci(&Triangular, config, dims, terms, fv, scale, cache);
}

void Triangular_mult_draw(int dims, KernelConfig config, int terms, const float ** fv, const float ** scale, float * out, MultCache * cache, int fake)
{
 if (fake==2)
 {
  // We can do the average of feature vector positions option... 
   int i, j;
   for (i=0; i<dims; i++) out[i] = 0.0;
   
   for (j=0; j<terms; j++)
   {
    for (i=0; i<dims; i++)
    {
     out[i] += ((fv[j][i] / scale[j][i]) - out[i]) / (j+1);
    }
   }
 }
 else
 {
  // For options 0 and 1 do a proper draw...   
   mult_draw_mh(&Triangular, config, dims, terms, fv, scale, out, cache);
 }
}



const Kernel Triangular =
{
 "triangular",
 "Provides a linear kernel - linear falloff from the centre of the unit hypersphere, to reach 0 at the edge.",
 NULL,
 Kernel_config_new,
 Kernel_config_verify,
 Kernel_config_acquire,
 Kernel_config_release,
 Triangular_weight,
 Triangular_norm,
 Triangular_range,
 Kernel_to_offset,
 Kernel_offset,
 Triangular_draw,
 Triangular_mult_mass,
 Triangular_mult_draw,
 Kernel_states,
 Kernel_next,
};



// The Epanechnikov kernel type...
float Epanechnikov_weight(int dims, KernelConfig config, float * offset)
{
 float dist_sqr = 0.0;
 
 int i;
 for (i=0; i<dims; i++)
 {
  dist_sqr += offset[i] * offset[i];
  if (dist_sqr>1.0) return 0.0;
 }
 
 return 1.0 - dist_sqr;
}

float Epanechnikov_norm(int dims, KernelConfig config)
{
 return 0.5 * (dims + 2.0) * Uniform_norm(dims, NULL);
}

float Epanechnikov_range(int dims, KernelConfig config, float quality)
{
 return 1.0;
}

void Epanechnikov_draw(int dims, KernelConfig config, PhiloxRNG * rng, const float * center, float * out)
{
 // Put a Gaussian into each out, keeping track of the squared length...
  int i;
  float radius = 0.0;
  for (i=0; i<dims; i+=2)
  {
   // Output...
    float * second = (i+1<dims) ? (out+i+1) : NULL;
    out[i] = PhiloxRNG_Gaussian(rng, second);
    
    radius += out[i] * out[i];
    if (second!=NULL) radius += out[i+1] * out[i+1];
  }
  
 // Convert from squared radius to not-squared radius...
  radius = sqrt(radius);
  
 // Draw the radius we are going to emit; prepare the multiplier...
  float u = PhiloxRNG_uniform(rng);
  radius = -2.0 * cos((atan2(sqrt(1.0-u*u), u) + 4*M_PI) / 3.0) / radius;
 
 // Normalise so its at the required distance...
  for (i=0; i<dims; i++)
  {
   out[i] = center[i] + out[i] * radius;
  }
}



float Epanechnikov_mult_mass(int dims, KernelConfig config, int terms, const float ** fv, const float ** scale, MultCache * cache)
{
 return mult_area_mci(&Epanechnikov, config, dims, terms, fv, scale, cache);
}

void Epanechnikov_mult_draw(int dims, KernelConfig config, int terms, const float ** fv, const float ** scale, float * out, MultCache * cache, int fake)
{
 if (fake==2)
 {
  // We can do the average of feature vector positions option... 
   int i, j;
   for (i=0; i<dims; i++) out[i] = 0.0;
   
   for (j=0; j<terms; j++)
   {
    for (i=0; i<dims; i++)
    {
     out[i] += ((fv[j][i] / scale[j][i]) - out[i]) / (j+1);
    }
   }
 }
 else
 {
  // For options 0 and 1 do a proper draw...   
   mult_draw_mh(&Epanechnikov, config, dims, terms, fv, scale, out, cache);
 }
}



const Kernel Epanechnikov =
{
 "epanechnikov",
 "Provides a kernel with a squared falloff, such that it hits 0 at the edge of the hyper-sphere. Probably the fastest to calculate other than the uniform kernel, and probably the best choice for a finite kernel.",
 NULL,
 Kernel_config_new,
 Kernel_config_verify,
 Kernel_config_acquire,
 Kernel_config_release,
 Epanechnikov_weight,
 Epanechnikov_norm,
 Epanechnikov_range,
 Kernel_to_offset,
 Kernel_offset,
 Epanechnikov_draw,
 Epanechnikov_mult_mass,
 Epanechnikov_mult_draw,
 Kernel_states,
 Kernel_next,
};



// The cosine kernel type...
float Cosine_weight(int dims, KernelConfig config, float * offset)
{
 float dist_sqr = 0.0;
 
 int i;
 for (i=0; i<dims; i++)
 {
  dist_sqr += offset[i] * offset[i];
  if (dist_sqr>1.0) return 0.0;
 }
 
 return cos(0.5 * M_PI * sqrt(dist_sqr));
}

float Cosine_norm(int dims, KernelConfig config)
{
 float mult = Uniform_norm(dims, NULL) / dims;
 
 int i;
 for (i=2; i<dims; i++) mult /= i;
 
 int k;
 float sum = 0.0;
 int dir = 1;
 for (k=0; (2*k)<dims; k++)
 {
  float fact = 1.0;
  for (i=2; i<(dims-2*k); i++) fact *= i;
  
  sum += dir * pow(0.5*M_PI, -(1+2*k)) / fact; 
  
  dir *= -1;
 }
 
 if ((dims>=2)&&((dims&1)==0)) // The extra term that appears because 0^0==1, but only when dims is even and 2 or greater.
 {
  if (((dims-2)/2)&1) dir = -1;
                 else dir =  1;
  
  sum -= dir * pow(0.5*M_PI, -dims);
 }

 return mult / sum;
}

float Cosine_range(int dims, KernelConfig config, float quality)
{
 return 1.0;
}

void Cosine_draw(int dims, KernelConfig config, PhiloxRNG * rng, const float * center, float * out)
{
 // Put a Gaussian into each out, keeping track of the squared length...
  int i;
  float radius = 0.0;
  for (i=0; i<dims; i+=2)
  {
   // Output...
    float * second = (i+1<dims) ? (out+i+1) : NULL;
    out[i] = PhiloxRNG_Gaussian(rng, second);
    
    radius += out[i] * out[i];
    if (second!=NULL) radius += out[i+1] * out[i+1];
  }
  
 // Convert from squared radius to not-squared radius...
  radius = sqrt(radius);
  
 // Draw the radius we are going to emit; prepare the multiplier...
  radius = 2.0*asin(PhiloxRNG_uniform(rng)) / (M_PI * radius);
 
 // Normalise so its at the required distance...
  for (i=0; i<dims; i++)
  {
   out[i] = center[i] + out[i] * radius;
  }
}



float Cosine_mult_mass(int dims, KernelConfig config, int terms, const float ** fv, const float ** scale, MultCache * cache)
{
 return mult_area_mci(&Cosine, config, dims, terms, fv, scale, cache);
}

void Cosine_mult_draw(int dims, KernelConfig config, int terms, const float ** fv, const float ** scale, float * out, MultCache * cache, int fake)
{
 if (fake==2)
 {
  // We can do the average of feature vector positions option... 
   int i, j;
   for (i=0; i<dims; i++) out[i] = 0.0;
   
   for (j=0; j<terms; j++)
   {
    for (i=0; i<dims; i++)
    {
     out[i] += ((fv[j][i] / scale[j][i]) - out[i]) / (j+1);
    }
   }
 }
 else
 {
  // For options 0 and 1 do a proper draw...   
   mult_draw_mh(&Cosine, config, dims, terms, fv, scale, out, cache);
 }
}



const Kernel Cosine =
{
 "cosine",
 "Kernel based on the cosine function, such that it hits zero at the edge of the unit hyper-sphere. Probably the smoothest of the kernels that have a hard edge beyond which they are zero; expensive to compute however.",
 NULL,
 Kernel_config_new,
 Kernel_config_verify,
 Kernel_config_acquire,
 Kernel_config_release,
 Cosine_weight,
 Cosine_norm,
 Cosine_range,
 Kernel_to_offset,
 Kernel_offset,
 Cosine_draw,
 Cosine_mult_mass,
 Cosine_mult_draw,
 Kernel_states,
 Kernel_next,
};



// The Gaussian kernel type...
float Gaussian_weight(int dims, KernelConfig config, float * offset)
{
 float dist_sqr = 0.0;
 
 int i;
 for (i=0; i<dims; i++)
 {
  dist_sqr += offset[i] * offset[i];
 }
 
 return exp(-0.5 * dist_sqr);
}

float Gaussian_norm(int dims, KernelConfig config)
{
 return pow(2.0 * M_PI, -0.5*dims);
}

float Gaussian_range(int dims, KernelConfig config, float quality)
{
 return (1.0-quality)*1.0 + quality*3.0;
}

void Gaussian_draw(int dims, KernelConfig config, PhiloxRNG * rng, const float * center, float * out)
{
 // Put a unit Gaussian into each out - that is all...
  int i;
  for (i=0; i<dims; i+=2)
  {
   // Output...
    float * second = (i+1<dims) ? (out+i+1) : NULL;
    out[i] = center[i] + PhiloxRNG_Gaussian(rng, second);
    if (second!=NULL) *second += center[i+1];
  }
}



float Gaussian_mult_mass(int dims, KernelConfig config, int terms, const float ** fv, const float ** scale, MultCache * cache)
{
 return mult_area_gaussian(dims, terms, fv, scale, cache);
}

void Gaussian_mult_draw(int dims, KernelConfig config, int terms, const float ** fv, const float ** scale, float * out, MultCache * cache, int fake)
{
 // Ignore fake==2 option as its computationaly near as the same cost as outputing the mean - call through...
  mult_draw_gaussian(dims, terms, fv, scale, out, cache, fake);
}



const Kernel Gaussian =
{
 "gaussian",
 "Standard Gaussian kernel; for range considers 1.5 standard deviations to be low quality, 3.5 to be high quality. More often than not the best choice, but very expensive and involves approximation.",
 NULL,
 Kernel_config_new,
 Kernel_config_verify,
 Kernel_config_acquire,
 Kernel_config_release,
 Gaussian_weight,
 Gaussian_norm,
 Gaussian_range,
 Kernel_to_offset,
 Kernel_offset,
 Gaussian_draw,
 Gaussian_mult_mass,
 Gaussian_mult_draw,
 Kernel_states,
 Kernel_next,
};



// The Cauchy kernel type...
float Cauchy_weight(int dims, KernelConfig config, float * offset)
{
 float dist_sqr = 0.0;
 
 int i;
 for (i=0; i<dims; i++)
 {
  dist_sqr += offset[i] * offset[i];
 }
 
 return 1.0 / (1.0 + dist_sqr);
}

float Cauchy_norm(int dims, KernelConfig config)
{
 float ret = 0.0;
 
 // Can't integrate out analytically, so numerical integration it is (match the range of high quality)...
  int i;
  const int samples = 1024;
  const float step = 6.0 / samples;
  for (i=0; i<samples; i++)
  {
   float r = (i+0.5) * step;
   ret += pow(r, dims-1) * step / (1.0+r*r);
  }
 
 return Uniform_norm(dims, NULL) / (dims * ret);
}

float Cauchy_range(int dims, KernelConfig config, float quality)
{
 return (1.0-quality)*2.0 + quality*6.0;
}

void Cauchy_draw(int dims, KernelConfig config, PhiloxRNG * rng, const float * center, float * out)
{
 // Put a Gaussian into each out, keeping track of the squared length...
  int i;
  float radius = 0.0;
  for (i=0; i<dims; i+=2)
  {
   // Output...
    float * second = (i+1<dims) ? (out+i+1) : NULL;
    out[i] = PhiloxRNG_Gaussian(rng, second);
    
    radius += out[i] * out[i];
    if (second!=NULL) radius += out[i+1] * out[i+1];
  }
  
 // Convert from squared radius to not-squared radius...
  radius = sqrt(radius);
  
 // Draw the radius we are going to emit; prepare the multiplier...  
  radius = tan(0.5*M_PI*PhiloxRNG_uniform(rng)) / radius;
 
 // Normalise so its at the required distance, add the center offset...
  for (i=0; i<dims; i++)
  {
   out[i] = center[i] + out[i] * radius;
  }
}



float Cauchy_mult_mass(int dims, KernelConfig config, int terms, const float ** fv, const float ** scale, MultCache * cache)
{
 return mult_area_mci(&Cauchy, config, dims, terms, fv, scale, cache);
}

void Cauchy_mult_draw(int dims, KernelConfig config, int terms, const float ** fv, const float ** scale, float * out, MultCache * cache, int fake)
{
 if (fake==2)
 {
  // We can do the average of feature vector positions option... 
   int i, j;
   for (i=0; i<dims; i++) out[i] = 0.0;
   
   for (j=0; j<terms; j++)
   {
    for (i=0; i<dims; i++)
    {
     out[i] += ((fv[j][i] / scale[j][i]) - out[i]) / (j+1);
    }
   }
 }
 else
 {
  // For options 0 and 1 do a proper draw...   
   mult_draw_mh(&Cauchy, config, dims, terms, fv, scale, out, cache);
 }
}



const Kernel Cauchy =
{
 "cauchy",
 "Uses the Cauchy distribution pdf on distance from the origin in the hypersphere. A fatter distribution than the Gaussian due to its long tails. Requires very large ranges, making is quite expensive in practise, but its good at avoiding being overconfident.",
 NULL,
 Kernel_config_new,
 Kernel_config_verify,
 Kernel_config_acquire,
 Kernel_config_release,
 Cauchy_weight,
 Cauchy_norm,
 Cauchy_range,
 Kernel_to_offset,
 Kernel_offset,
 Cauchy_draw,
 Cauchy_mult_mass,
 Cauchy_mult_draw,
 Kernel_states,
 Kernel_next,
};



// The von-Mises Fisher kernel...
typedef struct FisherConfig FisherConfig;

struct FisherConfig
{
 int ref_count; // Reference count.
 float alpha; // Concentration parameter.
 float log_norm; // log of the normalising constant.
 
 int inv_culm_size; // Length of below.
 float * inv_culm; // Array containing the inverse culmative of the distribution over the dot product result.
 
 int * order; // Array of length dims that contains the integers 0..dims-1; used when drawing.
};



KernelConfig Fisher_config_new(int dims, const char * config)
{
 FisherConfig * ret = (FisherConfig*)malloc(sizeof(FisherConfig));
 static const float epsilon = 1e-6;
 
 // Basic value storage...
  ret->ref_count = 1;
  char * end;
  ret->alpha = strtof(config+1, &end); // +1 to skip the '('.

  int approximate = ret->alpha > CONC_SWITCH;
  if (end!=NULL)
  {
   if (*end=='c') approximate = 0;
   if (*end=='a') approximate = 1;
  }
  
 // Record the log of the normalising constant - we return normalised values for this distribution for reasons of numerical stability...
  const float log_2PI = log(2 * M_PI);
  
  float bessel = 0.0;
  if (approximate)
  {
   ret->log_norm = (-0.5*(dims-1)) * (log_2PI - log(ret->alpha));
  }
  else
  {
   bessel = LogModBesselFirst(dims-2, ret->alpha, epsilon, 1024);
   
   ret->log_norm  = (0.5 * dims - 1) * log(ret->alpha);
   ret->log_norm -= (0.5 * dims) * log_2PI;
   ret->log_norm -= bessel;
  }
 
 // Only create the inverse culm array if we are being accurate - its not required in approximate mode...
  int i;
  if (approximate)
  {
   ret->inv_culm_size = 0;
   ret->inv_culm = NULL;
  }
  else
  {
   // For the below we need the marginal over the dot product between directions...
    float log_base = (0.5 * dims - 1) * (log(ret->alpha) - log(2));
    log_base -= LogGamma(dims - 2);
    log_base -= 0.5 * log(M_PI);  
    log_base -= bessel;
  
    const int size = 4*1024;
    const int size_big = 32 * size;
    float * culm = (float*)malloc(size_big * sizeof(float));
    
    for (i=0;i<size_big; i++)
    {
     float dot = ((float)(2*i)) / (size_big-1) - 1.0;
     if (dot<(-1+epsilon)) dot = -1 + epsilon; // To avoid infinities
     if (dot>(1-epsilon))  dot =  1 - epsilon; // "
     
     culm[i] = ret->alpha * dot + log_base;
     if (dims!=3) culm[i] += (0.5 * (dims - 3)) * log(1.0 - dot*dot);
     culm[i] = exp(culm[i]);
    }
  
   // Make the marginal culumative...
    float spacing = 2.0 / (size_big-1);
    float prev = culm[0];
    culm[0] = 0.0;
  
    for (i=1;i<size_big; i++)
    {
     float temp = culm[i];
     culm[i] = culm[i-1] + (prev + culm[i]) * 0.5 * spacing;
     prev = temp;
    }
  
   // It should sum to 1, but numerical error is best corrected for - normalise...
    for (i=0;i<size_big-1; i++)
    {
     culm[i] /= culm[size_big-1];
    }
    culm[size_big-1] = 1.0;
  
   // Calculate and store the inverse culumative distribution over the dot product of the directions - this allows us to efficiently draw from the distribution...
    ret->inv_culm_size = size;
    ret->inv_culm = (float*)malloc(size * sizeof(float));
  
    int j = 1;
  
    ret->inv_culm[0] = -1.0;
    for (i=1; i<size-1; i++)
    {
     // Set j to be one above the value...
      float pos = ((float)i) / (size-1);
      while (culm[j]<pos) j += 1;
    
     // Interpolate to fill in the relevant inverse value...
      float div = culm[j] - culm[j-1];
      if (div<epsilon) div = epsilon;
      float t = (pos - culm[j-1]) / div;
      float low = ((float)(2*(j-1))) / (size_big-1) - 1.0;
      float high = ((float)(2*j)) / (size_big-1) - 1.0;
    
      ret->inv_culm[i] = (1.0-t) * low + t * high;
    }
    ret->inv_culm[size-1] = 1.0;
   
   // Clean up...
    free(culm);
  }
  
 // Create the order array, for use when drawing...
  ret->order = (int*)malloc(dims*sizeof(int));
  for (i=0; i<dims; i++)
  {
   ret->order[i] = i;
  }
  
 // Return...
  return (KernelConfig)ret;
}

const char * Fisher_config_verify(int dims, const char * config, int * length)
{
 if (dims==0) return "The data must be set before initialising a von-Mises Fisher kernel, so it knows the dimension count.";

 if (config[0]!='(') return "von-Mises Fisher configuration did not start with a (.";
   
 char * end;
 float conc = strtof(config+1, &end);
 
 if (end==config) return "No concentration parameter given to von-Mises Fisher distribution.";
 if (conc<0.0) return "Negative concentration parameter given to von-Mises Fisher distribution.";
 
 if ((end!=NULL)&&((*end=='a')||(*end=='c')))
 {
  ++end; // Skip a mode forcer.
 }
 
 if ((end==NULL)||(*end!=')')) return "von-Mises Fisher configuration did not end with a ).";
 
 if (length!=NULL)
 {
  if (end==NULL) *length = strlen(config);
  else
  {
   *length = 1 + (end-config);
  }
 }
 
 return NULL; 
}

void Fisher_config_acquire(KernelConfig config)
{
 FisherConfig * self = (FisherConfig*)config;
 self->ref_count += 1;
}

void Fisher_config_release(KernelConfig config)
{
 FisherConfig * self = (FisherConfig*)config;
 self->ref_count -= 1;
 
 if (self->ref_count==0)
 {
  free(self->order);
  free(self->inv_culm);
  free(self);
 }
}



float Fisher_weight(int dims, KernelConfig config, float * offset)
{
 FisherConfig * self = (FisherConfig*)config; 
 
 // Convert the offset to a dot product...
  int i;
  float d_sqr = 0.0;
  for (i=0; i<dims; i++) d_sqr += offset[i] * offset[i];
 
  float cos_ang = 1.0 - 0.5*d_sqr; // Uses the law of cosines - how to calculate the dot product of unit vectors given their difference.
    
 // Behaviour depends on if we are being approximate or not...
  if (self->inv_culm==NULL)
  {
   // Use the Gaussian approximation, by halucinating that it is a vector of the form [a, b, 0...] and the direction of the distribution is [1, 0...] - this allows all the terms of the Gaussian to be calculated and a probability generated...
    if (cos_ang>0.0)
    {
     return exp(-0.5 * self->alpha * (1.0 - cos_ang*cos_ang) + self->log_norm);
    }
    else
    {
     return 0.0; // Actually the correct value for this approximation!
    }
  }
  else
  {
   // Correct version...
    return exp(self->alpha * cos_ang + self->log_norm);
  }
}

float Fisher_norm(int dims, KernelConfig config)
{
 return 1.0; // We return normalised values directly, for reasons of numerical stability.
}

float Fisher_range(int dims, KernelConfig config, float quality)
{
 FisherConfig * self = (FisherConfig*)config;
 
 if (self->inv_culm==NULL)
 {
  float sd = 1.0 + 2.0 * quality;
  return sd / sqrt(self->alpha);
 }
 else
 {
  float prob = 0.7 + 0.3 * quality; // Lowest quality is 70% of probability mass, highest 100% - roughly matches with Gaussian 1 to 3 standard deviations.
  float dot = self->inv_culm[(int)(self->inv_culm_size * (1.0-prob))];
  return sqrt(2.0 - 2.0 * dot);
 }
}

float Fisher_offset(int dims, KernelConfig config, float * fv, const float * offset)
{
 int i;
 
 // Calculate the normalising constant...
  float norm = 0.0;
  for (i=0; i<dims; i++)
  {
   float val = fv[i] + offset[i];
   norm += val * val;
  }
  norm = 1.0 / sqrt(norm);
  
 // Apply the change, keeping track of the delta, so we can return it...
  float delta = 0.0;
  for (i=0; i<dims; i++)
  {
   float val =  norm * (fv[i] + offset[i]);
   delta += fabs(val - fv[i]);
   fv[i] = val;
  }
 
 return delta;
}

void Fisher_draw(int dims, KernelConfig config, PhiloxRNG * rng, const float * center, float * out)
{
 FisherConfig * self = (FisherConfig*)config;

 // Generate a uniform draw into all but the first dimension of out, which is currently the mean direction...
  int i;
  float radius = 0.0;
  
  for (i=1; i<dims; i+=2)
  {
   // Output...
    float * second = (i+1<dims) ? (out+i+1) : NULL;
    out[i] = PhiloxRNG_Gaussian(rng, second);
    
    radius += out[i] * out[i];
    if (second!=NULL) radius += out[i+1] * out[i+1];
  }
  
 // Two approaches, depending on if we are being apprxoimate or not...
  if (self->inv_culm==NULL)
  {
   // Put 1 into the first entry, and scale the rest to obey the Gaussian approximation...
    out[0] = 1.0;
    
    float div = sqrt(self->alpha);
    for (i=1; i<dims; i++)
    {
     out[i] /= div;
    }
    
   // Normalise to length 1...
    radius /= self->alpha;
    radius += 1.0;
   
    for (i=0; i<dims; i++) out[i] /= radius;
  }
  else
  {
   // Draw the value of the dot product between the output vector and the kernel direction (1, 0, 0, ...), putting it into out[0]...
    float t = PhiloxRNG_uniform(rng) * (self->inv_culm_size-1);
    int low = (int)t;
    if ((low+1)==self->inv_culm_size) low -= 1;
    t -= low;

    out[0] = (1.0-t) * self->inv_culm[low] + t * self->inv_culm[low+1];

   // Blend the first row of the basis with the random draw to obtain the drawn dot product - i.e. scale the uniform draw so that with the first element set to the drawn dot product the entire vector is of length 1...
    radius = sqrt(1.0 - out[0]*out[0]) / sqrt(radius);
    for (i=1; i<dims; i++) out[i] *= radius;
  }
  
 // Find the order of the indices, in self->order, such that center goes from highest absolute value to lowest - needed for numerical stability in the next bit. Use insertion sort as the indices count is typically very low, making quick sort a bad choice (Note that we start from the previous order, as draw is often called repeatedly for the same centre value - fast escape)...
  for (i=0;i<dims-1; i++)
  {
   float abs_val_i = fabs(center[self->order[i]]);
   int j;
   for (j=i+1; j<dims; j++)
   {
    float abs_val = fabs(center[self->order[j]]); 
    if (abs_val>abs_val_i)
    {
     int temp = self->order[i];
     self->order[i] = self->order[j];
     self->order[j] = temp;
     abs_val_i = abs_val; 
    }
   }
  }
  
 // Might need to swap the value of the first one...
  if (self->order[0]!=0)
  {
   float temp = out[self->order[0]];
   out[self->order[0]] = out[0];
   out[0] = temp;
  }
  
 // Rotate to put the orthonormal basis in the correct position - apply 2x2 rotation matrices in sequence to rotate the (1, 0, 0, ...) vector to center...
  float tail = 1.0;
  for (i=0; i<dims-1; i++)
  {
   // The positions we are working with - for numerical stability...
   int pos = self->order[i];
   int npos = self->order[i+1];
   
   // Calculate the rotation matrix that leaves tail at the value in the center vector...
    float cos_theta = center[pos] / tail;
    float sin_theta = sqrt(1.0  - cos_theta*cos_theta);
    if (sin_theta!=sin_theta) sin_theta = 0.0; // NaN safety
   
   // Apply the rotation matrix to the tail, but offset to the row below, ready for the next time around the loop...
    tail *= sin_theta;
   
   // In the sqrt above we might want the negative answer - check and make the change if so...
    if ((tail * center[npos]) < 0.0)
    {
     sin_theta *= -1.0;
     tail      *= -1.0;
    }
    
    if (fabs(tail)<1e-6) tail = copysignf(1e-6, tail); // Avoids divide by zeros.
    
   // Apply the 2x2 rotation we have calculated...
    float oi  = out[pos];
    float oi1 = out[npos];
    
    out[pos]  = cos_theta * oi - sin_theta * oi1;
    out[npos] = sin_theta * oi + cos_theta * oi1;
  }
  
  if ((tail * center[self->order[dims-1]]) < 0.0)
  {
   out[self->order[dims-1]] *= -1.0;
  }
}


float Fisher_mult_mass(int dims, KernelConfig config, int terms, const float ** fv, const float ** scale, MultCache * cache)
{
 FisherConfig * self = (FisherConfig*)config;
 return mult_area_fisher(self->alpha, self->log_norm, dims, terms, fv, scale, cache, self->inv_culm==NULL);
}

void Fisher_mult_draw(int dims, KernelConfig config, int terms, const float ** fv, const float ** scale, float * out, MultCache * cache, int fake)
{
 mult_draw_mh(&Fisher, config, dims, terms, fv, scale, out, cache);
}



const Kernel Fisher =
{
 "fisher",
 "A kernel for dealing with directional data, using the von-Mises Fisher distribution as a kernel (Fisher is technically 3 dimensions only, but I like short names! - this will do anything from 2 dimensions upwards. Dimensions is the embeding space, as in the length of the unit vector, to match the rest of the system - the degrees of freedom is one less.). Requires that all feature vectors be on the unit-hypersphere (does not check this - gigo), plus it uses the alpha parameter provided to the kernel as the concentration parameter of the distribution. Be careful with the merge_range parameter when using this kernel - unlike the other kernels it won't default to anything sensible, and will have to be manually set. Suffers from the problem that you must use the same kernel for multiplication, so you can only multiply distributions with the same concentration value.",
 "Specified as fisher(alpha), e.g. fisher(10), where alpha is the concentration parameter. Can optionally include a letter immediatly after the alpha value - either 'a' to force approximate mode or 'c' to force correct mode, e.g. 'fisher(64.0a)'. Note that this is generally not a good idea - the approximation breaks down for low concentration and the correct approach numerically explodes for high concentration - the default behaviour automatically takes this into account and selects the right one.",
 Fisher_config_new,
 Fisher_config_verify,
 Fisher_config_acquire,
 Fisher_config_release,
 Fisher_weight,
 Fisher_norm,
 Fisher_range,
 Kernel_to_offset,
 Fisher_offset,
 Fisher_draw,
 Fisher_mult_mass,
 Fisher_mult_draw,
 Kernel_states,
 Kernel_next,
};

// The mirrored von-Mises Fisher kernel - reuse the data structure from the non-mirrored version...
float MirrorFisher_weight(int dims, KernelConfig config, float * offset)
{
 FisherConfig * self = (FisherConfig*)config;
 
 int i;
 float d_sqr = 0.0;
 for (i=0; i<dims; i++) d_sqr += offset[i] * offset[i];
 
 float cos_ang = 1.0 - 0.5*d_sqr; // Uses the law of cosines - how to calculate the dot product of unit vectors given their difference.
 
 if (self->inv_culm==NULL)
 {
  // Ammusingly the approximation is actually marginally simpler in the mirrored case!..
  return 0.5 * exp(-0.5 * self->alpha * (1.0 - cos_ang*cos_ang) + self->log_norm); 
 }
 else
 {
  return 0.5 * exp(self->alpha * cos_ang + self->log_norm) + 0.5 * exp(-self->alpha * cos_ang + self->log_norm);
 }
}

float MirrorFisher_range(int dims, KernelConfig config, float quality)
{
 return 2.1; // Due to the nature of the distribution this optimisation is not possible - 2.1 effectivly switches it off.
}

void MirrorFisher_to_offset(int dims, KernelConfig config, float * fv, const float * base_fv)
{
 // We have a mixture of two Fisher distributions - makes sense to choose the reflection closest to the base_fv - check the dot product and act on its sign...
  int i;
  float dot = 0.0;
  for (i=0; i<dims; i++) dot += fv[i] * base_fv[i];
  
  if (dot>0.0)
  {
   for (i=0; i<dims; i++) fv[i] -= base_fv[i];   
  }
  else
  {
   for (i=0; i<dims; i++) fv[i] = -fv[i] - base_fv[i];
  }
}

void MirrorFisher_draw(int dims, KernelConfig config, PhiloxRNG * rng, const float * center, float * out)
{
 Fisher_draw(dims, config, rng, center, out);
 
 // 50% chance of negating the output...
  if (PhiloxRNG_uniform(rng)<0.5)
  {
   int i;
   for (i=0; i<dims; i++)
   {
    out[i] = -out[i]; 
   }
  }
}



float MirrorFisher_mult_mass(int dims, KernelConfig config, int terms, const float ** fv, const float ** scale, MultCache * cache)
{
 FisherConfig * self = (FisherConfig*)config;
 return mult_area_mirror_fisher(self->alpha, self->log_norm, dims, terms, fv, scale, cache, self->inv_culm==NULL);
}

void MirrorFisher_mult_draw(int dims, KernelConfig config, int terms, const float ** fv, const float ** scale, float * out, MultCache * cache, int fake)
{
 mult_draw_mh(&MirrorFisher, config, dims, terms, fv, scale, out, cache);
}



int MirrorFisher_states(int dims, KernelConfig config)
{
 return 2;  
}

void MirrorFisher_next(int dims, KernelConfig config, int state, float * fv)
{
 int i;
 for (i=0; i<dims; i++)
 {
  fv[i] *= -1; 
 }
}



const Kernel MirrorFisher =
{
 "mirror_fisher",
 "A kernel for dealing with directional data where a 180 degree rotation is meaningless; one specific use case is for rotations expressed with unit quaternions. It wraps the Fisher distribution such that it is an even mixture of two of them - one with the correct unit vector, one with its negation. All other behaviours and requirements are basically the same as a Fisher distribution however. Suffers from the problem that you must use the same kernel for multiplication, so you can only multiply distributions with the same concentration value.",
 "Specified as mirror_fisher(alpha), e.g. mirror_fisher(10), where alpha is the concentration parameter used for both of the von-Mises Fisher distributions. Same as fisher you can immediatly postcede the concentration with 'a' to force it to be approximate or 'c' to force it to be correct.",
 Fisher_config_new,
 Fisher_config_verify,
 Fisher_config_acquire,
 Fisher_config_release,
 MirrorFisher_weight,
 Fisher_norm,
 MirrorFisher_range,
 MirrorFisher_to_offset,
 Fisher_offset,
 MirrorFisher_draw,
 MirrorFisher_mult_mass,
 MirrorFisher_mult_draw,
 MirrorFisher_states,
 MirrorFisher_next,
};



// The composite kernel - allows you to have different kernels on different dimensions...
typedef struct CompositeChild CompositeChild;
typedef struct CompositeConfig CompositeConfig;

struct CompositeChild
{
 int dims; // How many dimensions this kernel is responsible for.
 const Kernel * kernel;
 KernelConfig config;
};

struct CompositeConfig
{
 int ref_count;
 int children; // Number of children.
 CompositeChild child[0];
};



KernelConfig Composite_config_new(int dims, const char * config)
{
 int child;
 
 // First pass to count how many child kernels there are...
  int children = 0;
  char * targ = (char*)(config + 1); // Skip starting (
  while (targ[0]!=')')
  {
   children += 1;
   
   // Skip dims...
    child = strtol(targ, &targ, 0); // Compiler warns if I don't assign it to something:-/
    ++targ; // Skip :
    
   // Skip kernel spec...
    int i = 0;
    while (ListKernel[i]!=NULL)
    {
     int nlength = strlen(ListKernel[i]->name);
     if (strncmp(ListKernel[i]->name, targ, nlength)==0)
     {
      targ += nlength;
      ListKernel[i]->config_verify(dims, targ, &nlength);
      targ += nlength;
      
      break;
     }
   
     ++i; 
    }
   
   // Skip comma if needed...
    if (targ[0]==',') ++targ;
  }
 
 // Create the data structure...
  CompositeConfig * ret = (CompositeConfig*)malloc(sizeof(CompositeConfig) + children * sizeof(CompositeChild));
  
  ret->ref_count = 1;
  ret->children = children;
 
 // Second pass to fill in the data structure...
  targ = (char*)(config + 1); // Skip starting (
  
  for (child=0; child<children; child++)
  {
   ret->child[child].dims = strtol(targ, &targ, 0);
   ++targ; // Skip :
   
   int i = 0;
   while (ListKernel[i]!=NULL)
   {
    int nlength = strlen(ListKernel[i]->name);
    if (strncmp(ListKernel[i]->name, targ, nlength)==0)
    {
     targ += nlength;
     ListKernel[i]->config_verify(dims, targ, &nlength);
     
     ret->child[child].kernel = ListKernel[i];
     ret->child[child].config = ListKernel[i]->config_new(ret->child[child].dims, targ);
     
     targ += nlength;
     break;
    }
   
    ++i; 
   }
    
   ++targ; // Skip ,
  }
 
 // Return...
  return (KernelConfig)ret;
}

const char * Composite_config_verify(int dims, const char * config, int * length)
{
 char * targ = (char*)config;
 
 if (targ[0]!='(') return "composite configuration string did not start with (.";
 ++targ;
 
 int count = 0;
 while ((targ[0]!=')')&&(targ[0]!=0))
 {
  // Count how many dimensions this kernel covers...
   int dims = strtol(targ, &targ, 0);
   if (dims<1) return "dimensions of child kernel must be at least 1.";
   if (targ==NULL) return "unexpected end of string.";
   if (targ[0]!=':') return "did not get an expected :.";
   ++targ;
   
  // Find out which kernel it is, then handle it...
   int i = 0;
   while (ListKernel[i]!=NULL)
   {
    int nlength = strlen(ListKernel[i]->name);
    if (strncmp(ListKernel[i]->name, targ, nlength)==0)
    {
     // We have found a matching kernel - run its verification method, and use that to find the end, or throw a wobbly if verification fails...
      ++count;
      targ += nlength;
     
      const char * error = ListKernel[i]->config_verify(dims, targ, &nlength);
      if (error!=NULL) return error;
      
      targ += nlength;
      
     break;
    }
   
    ++i; 
   }
   if (ListKernel[i]==NULL) return "unrecognised child kernel.";
   
  // Move to next bit...
   if ((targ[0]!=',')&&(targ[0]!=')')) return "error processing list of child kernels";
   if (targ[0]==',') ++targ;
 }
 
 if (targ[0]!=')') return "composite configuration string did not end with (.";
 ++targ;
 
 if (count==0) return "composite configuration needs at least 1 child kernel.";
 
 if (length!=NULL) *length = targ - config;
 
 return NULL;
}

void Composite_config_acquire(KernelConfig config)
{
 CompositeConfig * self = (CompositeConfig*)config;
 self->ref_count += 1;
}

void Composite_config_release(KernelConfig config)
{
 CompositeConfig * self = (CompositeConfig*)config;
 self->ref_count -= 1;
 
 if (self->ref_count==0)
 {
  int i;
  for (i=0; i<self->children; i++)
  {
   self->child[i].kernel->config_release(self->child[i].config);
  }
   
  free(self); 
 }
}


float Composite_weight(int dims, KernelConfig config, float * offset)
{
 CompositeConfig * self = (CompositeConfig*)config;
 
 float ret = 1.0;
 
 int child;
 for (child=0; child<self->children; child++)
 {
  ret *= self->child[child].kernel->weight(self->child[child].dims, self->child[child].config, offset);
  
  offset += self->child[child].dims;
 }
 
 return ret;
}

float Composite_norm(int dims, KernelConfig config)
{
 CompositeConfig * self = (CompositeConfig*)config;
 
 float ret = 1.0;
 
 int child;
 for (child=0; child<self->children; child++)
 {
  ret *= self->child[child].kernel->norm(self->child[child].dims, self->child[child].config);
 }
 
 return ret;
}

float Composite_range(int dims, KernelConfig config, float quality)
{
 CompositeConfig * self = (CompositeConfig*)config;
 
 float ret = 0.0;
 
 // The system does not support ranges that vary by dimension, so the output has to be the maximum of the children - not computationally optimal, but unavoidable, and as long as all the dimensions are useful for filtering should not be too painful...
  int child;
  for (child=0; child<self->children; child++)
  {
  float range = self->child[child].kernel->range(self->child[child].dims, self->child[child].config, quality);
  if (range>ret) ret = range;
 }
 
 return ret;
}

void Composite_to_offset(int dims, KernelConfig config, float * fv, const float * base_fv)
{
 CompositeConfig * self = (CompositeConfig*)config;

 int child;
 for (child=0; child<self->children; child++)
 {
  self->child[child].kernel->to_offset(self->child[child].dims, self->child[child].config, fv, base_fv);
   
  fv += self->child[child].dims;
  base_fv += self->child[child].dims;
 }
}

float Composite_offset(int dims, KernelConfig config, float * fv, const float * offset)
{
 CompositeConfig * self = (CompositeConfig*)config;
 
 float ret = 0.0;
 
 int child;
 for (child=0; child<self->children; child++)
 {
  ret += self->child[child].kernel->offset(self->child[child].dims, self->child[child].config, fv, offset);
   
  fv += self->child[child].dims;
  offset += self->child[child].dims;
 }
 
 return ret;
}

void Composite_draw(int dims, KernelConfig config, PhiloxRNG * rng, const float * center, float * out)
{
 CompositeConfig * self = (CompositeConfig*)config;
 
 int child;
 for (child=0; child<self->children; child++)
 {
  self->child[child].kernel->draw(self->child[child].dims, self->child[child].config, rng, center, out);
   
  center += self->child[child].dims;
  out    += self->child[child].dims;
 }
}



float Composite_mult_mass(int dims, KernelConfig config, int terms, const float ** fv, const float ** scale, MultCache * cache)
{
 float ret = 1.0;
 CompositeConfig * self = (CompositeConfig*)config;
 
 int i;
 int child;
 
 int total = 0;
 for (child=0; child<self->children; child++)
 {
  // Factor in the current child...
   int c_dims = self->child[child].dims;
   ret *= self->child[child].kernel->mult_mass(c_dims, self->child[child].config, terms, fv, scale, cache);
  
  // Move array pointers to the next item...
   for (i=0; i<terms; i++)
   {
    fv[i] += c_dims;
    scale[i] += c_dims;
   }
   total += c_dims;
 }
 
 // Put fv and scale back how they were...
  for (i=0; i<terms; i++)
  {
   fv[i] -= total;
   scale[i] -= total;
  }
  
 return ret;
}

void Composite_mult_draw(int dims, KernelConfig config, int terms, const float ** fv, const float ** scale, float * out, MultCache * cache, int fake)
{
 CompositeConfig * self = (CompositeConfig*)config;
 
 int i;
 int child;
 
 int total = 0;
 for (child=0; child<self->children; child++)
 {
  // Draw for the current child...
   int c_dims = self->child[child].dims;
   self->child[child].kernel->mult_draw(c_dims, self->child[child].config, terms, fv, scale, out, cache, fake);
  
  // Move array pointers to the next item...
   for (i=0; i<terms; i++)
   {
    fv[i] += c_dims;
    scale[i] += c_dims;
   }
   out += c_dims;
   total += c_dims;
 }
 
 // Put fv and scale back how they were...
  for (i=0; i<terms; i++)
  {
   fv[i] -= total;
   scale[i] -= total;
  }
}



int Composite_states(int dims, KernelConfig config)
{
 CompositeConfig * self = (CompositeConfig*)config;
 
 int child;
 int ret = 1;
 
 for (child=0; child<self->children; child++)
 {
  int c_dims = self->child[child].dims;
  ret *= self->child[child].kernel->states(c_dims, self->child[child].config);
 }
 
 return ret;
}

void Composite_next(int dims, KernelConfig config, int state, float * fv)
{
 CompositeConfig * self = (CompositeConfig*)config;
 
 int child;
 int step_size = 1;
 
 for (child=0; child<self->children; child++)
 {
  int c_dims = self->child[child].dims;
  KernelConfig c_config = self->child[child].config;
  
  int steps = self->child[child].kernel->states(c_dims, c_config);
  
  if (steps>1)
  {
   if ((state%step_size)==0)
   {
    self->child[child].kernel->next(c_dims, c_config, state / step_size, fv);
   }
    
   step_size *= steps;
  }
  
  fv += c_dims;
 }
}



const Kernel Composite =
{
 "composite",
 "Allows you to use different kernels on different features, by specifying a list of kernels and how many dimensions each child kernel applies to. For instance, you could have a Gaussian on the first three features then a Fisher on the last three features. Note that this assumes that the number of dims in the data matches the number specificed in the kernel - if this is not the case brown stuff will interact with the spining blades of cooling.",
 "Configured with a comma seperated list of kernel specifications, where any kernel can be used; each kernel is proceded by the  number of features/dimensions it covers then a colon before giving the actual kernel spec. For example: composite(3:gaussian,3:fisher(48.0)) to have a Gaussian kernel on the first three dimensions then a Fisher kernel on the last three.",
 Composite_config_new,
 Composite_config_verify,
 Composite_config_acquire,
 Composite_config_release,
 Composite_weight,
 Composite_norm,
 Composite_range,
 Composite_to_offset,
 Composite_offset,
 Composite_draw,
 Composite_mult_mass,
 Composite_mult_draw,
 Composite_states,
 Composite_next,
};



// The list of known kernels...
const Kernel * ListKernel[] =
{
 &Uniform,
 &Triangular,
 &Epanechnikov,
 &Cosine,
 &Gaussian,
 &Cauchy,
 &Fisher,
 &MirrorFisher,
 &Composite,
 NULL
};



// Dummy function, to make a warning go away because it was annoying me...
void KernelsModule_IgnoreMe(void)
{
 import_array();  
}
