# Copyright (c) 2011, Tom SF Haines
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#  * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



from utils.start_cpp import start_cpp



# Code for sampling from various distributions, including some very specific situations involving Dirichlet processes...
sampling_code = start_cpp() + """
#ifndef SAMPLING_CODE
#define SAMPLING_CODE

#include <stdlib.h>
#include <math.h>

const double gamma_approx = 32.0; // Threshold between the two methods of doing a gamma draw.

// Returns a sample from the natural numbers [0,n)...
int sample_nat(int n)
{
 return lrand48()%n;
}

// Returns a sample from [0.0,1.0)...
double sample_uniform()
{
 return drand48();
 //return double(random())/(double(RAND_MAX)+1.0);
}

// Samples from a normal distribution with a mean of 0 and a standard deviation of 1...
double sample_standard_normal()
{
 double u = 1.0-sample_uniform();
 double v = 1.0-sample_uniform();
 return sqrt(-2.0*log(u)) * cos(2.0*M_PI*v);
}

// Samples from a normal distribution with the given mean and standard deviation...
double sample_normal(double mean, double sd)
{
 return mean + sd*sample_standard_normal();
}

// Samples from the Gamma distribution, base version that has no scaling parameter...
/*double sample_gamma(double alpha)
{
 // Check if the alpha value is high enough to approximate via a normal distribution...
  if (alpha>gamma_approx)
  {
   while (true)
   {
    double ret = sample_normal(alpha, sqrt(alpha));
    if (ret<0.0) continue;
    return ret;
   }
  }
 
 // First do the integer part of gamma(alpha)...
  double ret = 0.0; // 1.0
  while (alpha>=1.0)
  {
   alpha -= 1.0;
   //ret /= 1.0 - sample_uniform();
   ret -= log(1.0-sample_uniform());
  }
  //ret = log(ret);

 // Now do the remaining fractional part and sum it in - uses rejection sampling...
  if (alpha>1e-4)
  {
   while (true)
   {
    double u1 = 1.0 - sample_uniform();
    double u2 = 1.0 - sample_uniform();
    double u3 = 1.0 - sample_uniform();

    double frac, point;
    if (u1<=(M_E/(M_E+alpha)))
    {
     frac = pow(u2,1.0/alpha);
     point = u3*pow(frac,alpha-1.0);
    }
    else
    {
     frac = 1.0 - log(u2);
     point = u3*exp(-frac);
    }

    if (point<=(pow(frac,alpha-1.0)*exp(-frac)))
    {
     ret += frac;
     break;
    }
   }
  }

 // Finally return...
  return ret;
}*/

// As above, but faster...
double sample_gamma(double alpha)
{
 // Check if the alpha value is high enough to approximate via a normal distribution...
  if (alpha>gamma_approx)
  {
   while (true)
   {
    double ret = sample_normal(alpha, sqrt(alpha));
    if (ret<0.0) continue;
    return ret;
   }
  }

 // If alpha is one, within tolerance, just use an exponential distribution...
  if (fabs(alpha-1.0)<1e-4)
  {
   return -log(1.0-sample_uniform());
  }

 if (alpha>1.0)
 {
  // If alpha is 1 or greater use the Cheng/Feast method...
   while (true)
   {
    double u1 = sample_uniform();
    double u2 = sample_uniform();
    double v = ((alpha - 1.0/(6.0*alpha))*u1) / ((alpha-1.0)*u2);

    double lt2 = 2.0*(u2-1.0)/(alpha-1) + v + 1.0/v;
    if (lt2<=2.0)
    {
     return (alpha-1.0)*v;
    }

    double lt1 = 2.0*log(u2)/(alpha-1.0) - log(v) + v;
    if (lt1<=1.0)
    {
     return (alpha-1.0)*v;
    }
   }
 }
 else
 {
  // If alpha is less than 1 use a rejection sampling method...
   while (true)
   {
    double u1 = 1.0 - sample_uniform();
    double u2 = 1.0 - sample_uniform();
    double u3 = 1.0 - sample_uniform();

    double frac, point;
    if (u1<=(M_E/(M_E+alpha)))
    {
     frac = pow(u2,1.0/alpha);
     point = u3*pow(frac,alpha-1.0);
    }
    else
    {
     frac = 1.0 - log(u2);
     point = u3*exp(-frac);
    }

    if (point<=(pow(frac,alpha-1.0)*exp(-frac)))
    {
     return frac;
     break;
    }
   }
 }
}

// Samples from the Gamma distribution, version that has a scaling parameter...
double sample_gamma(double alpha, double beta)
{
 return sample_gamma(alpha)/beta;
}

// Samples from the Beta distribution...
double sample_beta(double alpha, double beta)
{
 double g1 = sample_gamma(alpha);
 double g2 = sample_gamma(beta);

 return g1 / (g1 + g2);
}

#endif
"""
