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

void Uniform_draw(int dims, KernelConfig config, const unsigned int index[3], const float * center, float * out)
{
 unsigned int random[4];
 
 // Put a Gaussian into each out, keeping track of the squared length...
  int i;
  float radius = 0.0;
  for (i=0; i<dims; i+=2)
  {
   // We need some random data...
    random[0] = index[0];
    random[1] = index[1];
    random[2] = index[2];
    random[3] = i;
    philox(random);
    
   // Output...
    float * second = (i+1<dims) ? (out+i+1) : NULL;
    out[i] = box_muller(random[0], random[1], second);
    
    radius += out[i] * out[i];
    if (second!=NULL) radius += out[i+1] * out[i+1];
  }
  
 // Convert from squared radius to not-squared radius...
  radius = sqrt(radius);
  
 // Draw the radius we are going to emit; prepare the multiplier...
  if ((dims&1)==0)
  {
   random[0] = index[0];
   random[1] = index[1];
   random[2] = index[2];
   random[3] = dims;
   philox(random);
  }
    
  radius = pow(uniform(random[3]), 1.0/dims) / radius;
 
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
 Kernel_offset,
 Uniform_draw,
 Uniform_mult_mass,
 Uniform_mult_draw,
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

void Triangular_draw(int dims, KernelConfig config, const unsigned int index[3], const float * center, float * out)
{
 unsigned int random[4];
 
 // Put a Gaussian into each out, keeping track of the squared length...
  int i;
  float radius = 0.0;
  for (i=0; i<dims; i+=2)
  {
   // We need some random data...
    random[0] = index[0];
    random[1] = index[1];
    random[2] = index[2];
    random[3] = i;
    philox(random);
    
   // Output...
    float * second = (i+1<dims) ? (out+i+1) : NULL;
    out[i] = box_muller(random[0], random[1], second);
    
    radius += out[i] * out[i];
    if (second!=NULL) radius += out[i+1] * out[i+1];
  }
  
 // Convert from squared radius to not-squared radius...
  radius = sqrt(radius);
  
 // Draw the radius we are going to emit; prepare the multiplier...
  if ((dims&1)==0)
  {
   random[0] = index[0];
   random[1] = index[1];
   random[2] = index[2];
   random[3] = dims;
   philox(random);
  }
    
  radius = (1.0 - sqrt(1.0 - uniform(random[3]))) / radius;
 
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
 Kernel_offset,
 Triangular_draw,
 Triangular_mult_mass,
 Triangular_mult_draw,
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

void Epanechnikov_draw(int dims, KernelConfig config, const unsigned int index[3], const float * center, float * out)
{
 unsigned int random[4];
 
 // Put a Gaussian into each out, keeping track of the squared length...
  int i;
  float radius = 0.0;
  for (i=0; i<dims; i+=2)
  {
   // We need some random data...
    random[0] = index[0];
    random[1] = index[1];
    random[2] = index[2];
    random[3] = i;
    philox(random);
    
   // Output...
    float * second = (i+1<dims) ? (out+i+1) : NULL;
    out[i] = box_muller(random[0], random[1], second);
    
    radius += out[i] * out[i];
    if (second!=NULL) radius += out[i+1] * out[i+1];
  }
  
 // Convert from squared radius to not-squared radius...
  radius = sqrt(radius);
  
 // Draw the radius we are going to emit; prepare the multiplier...
  if ((dims&1)==0)
  {
   random[0] = index[0];
   random[1] = index[1];
   random[2] = index[2];
   random[3] = dims;
   philox(random);
  }
  
  float u = uniform(random[3]);
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
 Kernel_offset,
 Epanechnikov_draw,
 Epanechnikov_mult_mass,
 Epanechnikov_mult_draw,
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
  
  sum += dir * pow(0.5*M_PI, 1+2*k) / fact; 
  
  dir *= -1;
 }

 return mult / sum;
}

float Cosine_range(int dims, KernelConfig config, float quality)
{
 return 1.0;
}

void Cosine_draw(int dims, KernelConfig config, const unsigned int index[3], const float * center, float * out)
{
 unsigned int random[4];
 
 // Put a Gaussian into each out, keeping track of the squared length...
  int i;
  float radius = 0.0;
  for (i=0; i<dims; i+=2)
  {
   // We need some random data...
    random[0] = index[0];
    random[1] = index[1];
    random[2] = index[2];
    random[3] = i;
    philox(random);
    
   // Output...
    float * second = (i+1<dims) ? (out+i+1) : NULL;
    out[i] = box_muller(random[0], random[1], second);
    
    radius += out[i] * out[i];
    if (second!=NULL) radius += out[i+1] * out[i+1];
  }
  
 // Convert from squared radius to not-squared radius...
  radius = sqrt(radius);
  
 // Draw the radius we are going to emit; prepare the multiplier...
  if ((dims&1)==0)
  {
   random[0] = index[0];
   random[1] = index[1];
   random[2] = index[2];
   random[3] = dims;
   philox(random);
  }
  
  radius = 2.0*asin(uniform(random[3])) / (M_PI * radius);
 
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
 Kernel_offset,
 Cosine_draw,
 Cosine_mult_mass,
 Cosine_mult_draw,
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

void Gaussian_draw(int dims, KernelConfig config, const unsigned int index[3], const float * center, float * out)
{
 unsigned int random[4];
 
 // Put a unit Gaussian into each out - that is all...
  int i;
  for (i=0; i<dims; i+=2)
  {
   // We need some random data...
    random[0] = index[0];
    random[1] = index[1];
    random[2] = index[2];
    random[3] = i;
    philox(random);
    
   // Output...
    float * second = (i+1<dims) ? (out+i+1) : NULL;
    out[i] = center[i] + box_muller(random[0], random[1], second);
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
 Kernel_offset,
 Gaussian_draw,
 Gaussian_mult_mass,
 Gaussian_mult_draw,
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
 
 // Can't integrate out analytically, so numerical integration it is...
  int i;
  const int samples = 1024;
  for (i=0; i<samples; i++)
  {
   float r = (i+0.5) / samples;
   ret += pow(r, dims-1) / ((1.0+r*r) * samples);
  }
 
 return ret * Uniform_norm(dims, NULL) / dims;
}

float Cauchy_range(int dims, KernelConfig config, float quality)
{
 return (1.0-quality)*2.0 + quality*6.0;
}

void Cauchy_draw(int dims, KernelConfig config, const unsigned int index[3], const float * center, float * out)
{
 unsigned int random[4];
 
 // Put a Gaussian into each out, keeping track of the squared length...
  int i;
  float radius = 0.0;
  for (i=0; i<dims; i+=2)
  {
   // We need some random data...
    random[0] = index[0];
    random[1] = index[1];
    random[2] = index[2];
    random[3] = i;
    philox(random);
    
   // Output...
    float * second = (i+1<dims) ? (out+i+1) : NULL;
    out[i] = box_muller(random[0], random[1], second);
    
    radius += out[i] * out[i];
    if (second!=NULL) radius += out[i+1] * out[i+1];
  }
  
 // Convert from squared radius to not-squared radius...
  radius = sqrt(radius);
  
 // Draw the radius we are going to emit; prepare the multiplier...
  if ((dims&1)==0)
  {
   random[0] = index[0];
   random[1] = index[1];
   random[2] = index[2];
   random[3] = dims;
   philox(random);
  }
  
  radius = tan(0.5*M_PI*uniform(random[3])) / radius;
 
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
 Kernel_offset,
 Cauchy_draw,
 Cauchy_mult_mass,
 Cauchy_mult_draw,
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
};



KernelConfig Fisher_config_new(int dims, const char * config)
{
 FisherConfig * ret = (FisherConfig*)malloc(sizeof(FisherConfig));
 static const float epsilon = 1e-6;
 
 // Basic value storage...
  ret->ref_count = 1;
  ret->alpha = atof(config+1); // +1 to skip the '('.
  
 // Record the log of the normalising constant - we return normalised values for this distribution for reasons of numerical stability...
  float bessel = LogModBesselFirst(dims-2, ret->alpha, epsilon, 1024);
  
  ret->log_norm  = (0.5 * dims - 1) * log(ret->alpha);
  ret->log_norm -= (0.5 * dims) * log(2 * M_PI);
  ret->log_norm -= bessel;
 
 // For the below we need the marginal over the dot product between directions...
  float log_base = (0.5 * dims - 1) * (log(ret->alpha) - log(2));
  log_base -= LogGamma(dims - 2);
  log_base -= 0.5 * log(M_PI);  
  log_base -= bessel;
  
  const int size = 4*1024;
  const int size_big = 32 * size;
  float * culm = (float*)malloc(size_big * sizeof(float));
  
  int i;
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
  
 // Clean up and return...
  free(culm);
 
 return (KernelConfig)ret;
}

const char * Fisher_config_verify(int dims, const char * config, int * length)
{
 if (config[0]!='(') return "von-Mises Fisher configuration did not start with a (.";
   
 char * end;
 float conc = strtof(config+1, &end);
 
 if (end==config) return "No concentration parameter given to von-Mises Fisher distribution.";
 if (conc<0.0) return "Negative concentration parameter given to von-Mises Fisher distribution.";
 
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
  free(self->inv_culm);
  free(self); 
 }
}



float Fisher_weight(int dims, KernelConfig config, float * offset)
{
 FisherConfig * self = (FisherConfig*)config;
 
 int i;
 float d_sqr = 0.0;
 for (i=0; i<dims; i++) d_sqr += offset[i] * offset[i];
 
 float cos_ang = 1.0 - 0.5*d_sqr; // Uses the law of cosines - how to calculate the dot product of unit vectors given their difference.
 
 return exp(self->alpha * cos_ang + self->log_norm);
}

float Fisher_norm(int dims, KernelConfig config)
{
 return 1.0; // We return normalised values directly, for reasons of numerical stability.
}

float Fisher_range(int dims, KernelConfig config, float quality)
{
 FisherConfig * self = (FisherConfig*)config;
 
 float prob = 0.7 + 0.3 * quality; // Lowest quality is 70% of probability mass, highest 100% - roughly matches with Gaussian 1 to 3 standard deviations.
 float dot = self->inv_culm[(int)(self->inv_culm_size * (1.0-prob))];
 return sqrt(2.0 - 2.0 * dot);
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

void Fisher_draw(int dims, KernelConfig config, const unsigned int index[3], const float * center, float * out)
{
 FisherConfig * self = (FisherConfig*)config;
 
 unsigned int random[4];

 // Generate a uniform draw into all but the first dimension of out, which is currently the mean direction...
  int i;
  float radius = 0.0;
  
  for (i=1; i<dims; i+=2)
  {
   // We need some random data...
    random[0] = index[0];
    random[1] = index[1];
    random[2] = index[2];
    random[3] = i;
    philox(random);
    
   // Output...
    float * second = (i+1<dims) ? (out+i+1) : NULL;
    out[i] = box_muller(random[0], random[1], second);
    
    radius += out[i] * out[i];
    if (second!=NULL) radius += out[i+1] * out[i+1];
  }
  
 // Make sure random[2] and random[3] contain unused random bits...
  if ((dims&1)!=0)
  {
   random[0] = index[0];
   random[1] = index[1];
   random[2] = index[2];
   random[3] = dims;
   philox(random);
  }
  
 // Draw the value of the dot product between the output vector and the kernel direction (1, 0, 0, ...), putting it into out[0]...
  float t = uniform(random[2]) * (self->inv_culm_size-1);
  int low = (int)t;
  if ((low+1)==self->inv_culm_size) low -= 1;
  t -= low;

  out[0] = (1.0-t) * self->inv_culm[low] + t * self->inv_culm[low+1];

 // Blend the first row of the basis with the random draw to obtain the drawn dot product - i.e. scale the uniform draw so that with the first element set to the drawn dot product the entire vector is of length 1...
  radius = sqrt(1.0 - out[0]*out[0]) / sqrt(radius);
  for (i=1; i<dims; i++) out[i] *= radius;
  
 // Rotate to put the orthonormal basis in the correct position - apply 2x2 rotation matrices in sequence to rotate the (1, 0, 0, ...) vector to center...
  float tail = 1.0;
  for (i=0; i<dims-1; i++)
  {
   // Calculate the rotation matrix that leaves tail at the value in the center vector...
    float cos_theta = center[i] / tail;
    float sin_theta = sqrt(1.0  - cos_theta*cos_theta);
    if (sin_theta!=sin_theta) sin_theta = 0.0; // NaN safety
   
   // Apply the rotation matrix to the tail, but offset to the row below, ready for the next time around the loop...
    tail *= sin_theta;
    if (tail<1e-6) tail = 1e-6; // Avoids divide by zeros.
   
   // In the sqrt above we might want the negative answer - check and make the change if so...
    if ((tail * center[i+1]) < 0.0)
    {
     sin_theta *= -1.0;
     tail      *= -1.0;
    }
    
   // Apply the 2x2 rotation we have calculated...
    float oi  = out[i];
    float oi1 = out[i+1];
    
    out[i]   = cos_theta * oi - sin_theta * oi1;
    out[i+1] = sin_theta * oi + cos_theta * oi1;
  }
}



float Fisher_mult_mass(int dims, KernelConfig config, int terms, const float ** fv, const float ** scale, MultCache * cache)
{
 FisherConfig * self = (FisherConfig*)config;
 return mult_area_fisher(self->alpha, self->log_norm, dims, terms, fv, scale, cache);

 //return mult_area_mci(&Fisher, config, dims, terms, fv, scale, cache);
}

void Fisher_mult_draw(int dims, KernelConfig config, int terms, const float ** fv, const float ** scale, float * out, MultCache * cache, int fake)
{
 mult_draw_mh(&Fisher, config, dims, terms, fv, scale, out, cache);
}



const Kernel Fisher =
{
 "fisher",
 "A kernel for dealing with directional data, using the von-Mises Fisher distribution as a kernel (Fisher is technically 3 dimensions only, but I like short names! - this will do anything from 2 dimensions upwards). Requires that all feature vectors be on the unit-hypersphere, plus it uses the alpha parameter provided to the kernel as the concentration parameter of the distribution. Be careful with the merge_range parameter when using this kernel - unlike the other kernels it won't default to anything sensible, and will have to be manually set.",
 "Specified as fisher(alpha), e.g. fisher(10), where alpha is the concentration parameter",
 Fisher_config_new,
 Fisher_config_verify,
 Fisher_config_acquire,
 Fisher_config_release,
 Fisher_weight,
 Fisher_norm,
 Fisher_range,
 Fisher_offset,
 Fisher_draw,
 Fisher_mult_mass,
 Fisher_mult_draw,
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

void Composite_draw(int dims, KernelConfig config, const unsigned int index[3], const float * center, float * out)
{
 CompositeConfig * self = (CompositeConfig*)config;
 
 unsigned int pos[3];
 pos[0] = index[0];
 pos[1] = index[1];
 pos[2] = index[2];
 
 int child;
 for (child=0; child<self->children; child++)
 {
  self->child[child].kernel->draw(self->child[child].dims, self->child[child].config, pos, center, out);
   
  center += self->child[child].dims;
  out    += self->child[child].dims;
  pos[2] += 1;
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
 Composite_offset,
 Composite_draw,
 Composite_mult_mass,
 Composite_mult_draw,
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
 &Composite,
 NULL
};



// Dummy function, to make a warning go away because it was annoying me...
void KernelsModule_IgnoreMe(void)
{
 import_array();  
}
