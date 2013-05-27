// Copyright 2013 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



#include "kernels.h"

#include <math.h>

#include "bessel.h"



// Most kernels use the same offset method, as provided by this implimentaiton...
float Kernel_offset(int dims, float alpha, float * fv, const float * offset)
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
float Uniform_weight(int dims, float alpha, float * offset)
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

float Uniform_norm(int dims, float alpha)
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

float Uniform_range(int dims, float alpha, float quality)
{
 return 1.0;
}



const Kernel Uniform =
{
 "uniform",
 "Provides a uniform kernel - all points within the unit hypersphere get a positive constant weight, all of those outside it get zero.",
 Uniform_weight,
 Uniform_norm,
 Uniform_range,
 Kernel_offset,
};



// The triangular kernel type...
float Triangular_weight(int dims, float alpha, float * offset)
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

float Triangular_norm(int dims, float alpha)
{
 return (dims + 1.0) * Uniform_norm(dims, 0.0);
}

float Triangular_range(int dims, float alpha, float quality)
{
 return 1.0;
}



const Kernel Triangular =
{
 "triangular",
 "Provides a linear kernel - linear falloff from the centre of the unit hypersphere, to reach 0 at the edge.",
 Triangular_weight,
 Triangular_norm,
 Triangular_range,
 Kernel_offset,
};



// The Epanechnikov kernel type...
float Epanechnikov_weight(int dims, float alpha, float * offset)
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

float Epanechnikov_norm(int dims, float alpha)
{
 return 0.5 * (dims + 2.0) * Uniform_norm(dims, 0.0);
}

float Epanechnikov_range(int dims, float alpha, float quality)
{
 return 1.0;
}



const Kernel Epanechnikov =
{
 "epanechnikov",
 "Provides a kernel with a squared falloff, such that it hits 0 at the edge of the hyper-sphere. Probably the fastest to calculate other than the uniform kernel, and probably the best choice for a finite kernel.",
 Epanechnikov_weight,
 Epanechnikov_norm,
 Epanechnikov_range,
 Kernel_offset,
};



// The cosine kernel type...
float Cosine_weight(int dims, float alpha, float * offset)
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

float Cosine_norm(int dims, float alpha)
{
 float mult = Uniform_norm(dims, 0.0) / dims;
 
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

float Cosine_range(int dims, float alpha, float quality)
{
 return 1.0;
}



const Kernel Cosine =
{
 "cosine",
 "Kernel based on the cosine function, such that it hits zero at the edge of the unit hyper-sphere. Probably the smoothest of the kernels that have a hard edge beyond which they are zero; expensive to compute however.",
 Cosine_weight,
 Cosine_norm,
 Cosine_range,
 Kernel_offset,
};



// The Gaussian kernel type...
float Gaussian_weight(int dims, float alpha, float * offset)
{
 float dist_sqr = 0.0;
 
 int i;
 for (i=0; i<dims; i++)
 {
  dist_sqr += offset[i] * offset[i];
 }
 
 return exp(-0.5 * dist_sqr);
}

float Gaussian_norm(int dims, float alpha)
{
 return pow(2.0 * M_PI, -0.5*dims);
}

float Gaussian_range(int dims, float alpha, float quality)
{
 return (1.0-quality)*1.5 + quality*3.5;
}



const Kernel Gaussian =
{
 "gaussian",
 "Standard Gaussian kernel; for range considers 1.5 standard deviations to be low quality, 3.5 to be high quality. More often than not the best choice, but very expensive and involves approximation.",
 Gaussian_weight,
 Gaussian_norm,
 Gaussian_range,
 Kernel_offset,
};



// The Cauchy kernel type...
float Cauchy_weight(int dims, float alpha, float * offset)
{
 float dist_sqr = 0.0;
 
 int i;
 for (i=0; i<dims; i++)
 {
  dist_sqr += offset[i] * offset[i];
 }
 
 return 1.0 / (1.0 + dist_sqr);
}

float Cauchy_norm(int dims, float alpha)
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
 
 return ret * Uniform_norm(dims, 0.0) / dims;
}

float Cauchy_range(int dims, float alpha, float quality)
{
 return (1.0-quality)*2.0 + quality*6.0;
}



const Kernel Cauchy =
{
 "cauchy",
 "Uses the Cauchy distribution pdf on distance from the origin in the hypersphere. A fatter distribution than the Gaussian due to its long tails. Requires very large ranges, making is quite expensive in practise, but its good at avoiding being overconfident.",
 Cauchy_weight,
 Cauchy_norm,
 Cauchy_range,
 Kernel_offset,
};



// The von-Mises Fisher kernel...
float Fisher_weight(int dims, float alpha, float * offset)
{
 int i;
 float d_sqr = 0.0;
 for (i=0; i<dims; i++) d_sqr += offset[i] * offset[i];
 
 float cos_ang = 1.0 - 0.5*d_sqr; // Uses the law of cosines - how to calculate the dot product of unit vectors given their difference.
 
 return exp(alpha * cos_ang);
}

float Fisher_norm(int dims, float alpha)
{
 float ret = pow(alpha, 0.5 * dims -1);
 ret /= pow(2.0 * M_PI, 0.5 * dims);
 ret /= ModBesselFirst(dims-2, alpha, 1e-6, 1024);
 return ret;
}

float Fisher_range(int dims, float alpha, float quality)
{
 float inv_accuracy = pow(10.0, 1.0+quality*3.0);
 return 2.0 - 2.0 * log(inv_accuracy) / alpha;
}

float Fisher_offset(int dims, float alpha, float * fv, const float * offset)
{
 int i;
 float dist = 0.0;
 float delta = 0.0;
 for (i=0; i<dims; i++)
 {
  delta += fabs(offset[i]);
  fv[i] += offset[i];
  dist += fv[i] * fv[i];
 }
 dist = sqrt(dist);
 
 for (i=0; i<dims; i++) fv[i] /= dist;
 
 return delta;
}



const Kernel Fisher =
{
 "fisher",
 "A kernel for dealing with directional data, using the von-Mises Fisher distribution as a kernel. Requires that all feature vectors be on the unit-hypersphere, plus it uses the alpha parameter provided to the kernel as the concentration parameter of the distribution. To avoid numerical problems its best to avoid taking the concentration parameter much above 64.",
 Fisher_weight,
 Fisher_norm,
 Fisher_range,
 Fisher_offset,
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
 NULL
};
