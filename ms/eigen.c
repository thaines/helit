// Copyright 2013 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



#include "eigen.h"

#include <math.h>



int symmetric_eigen_raw(int dims, float * a, float * q, float * d)
{
 const static float epsilon = 1e-6;
 
 // First perform a householder tri-diagonalisation, makes the following QR
 // iterations better...
  int k, r, c, i;
  for (k=0; k<dims-2; k++)
  {
   // Calculate householder vector, store it in the subdiagonal...
   // (Replacing first value of v (Which is always 1) with beta.)
   // (For off diagonal symmetric entrys only fill in super-diagonal.)
    float sigma = 0.0;
    float x0 = a[(k+1)*dims + k];
    for (r=k+2; r<dims; r++) sigma += a[r*dims+k] * a[r*dims+k];
    
    int kp1_k = (k+1) * dims + k;
    
    if (fabs(sigma)<epsilon) a[(k+1)*dims+k] = 0.0;
    else
    {
     float mu = sqrt(a[kp1_k]*a[kp1_k] + sigma);
     if (a[kp1_k]<=0.0) a[kp1_k] = a[kp1_k] - mu;
                   else a[kp1_k] = -sigma / (a[kp1_k] + mu);
     
     for (r=k+2; r<dims; r++) a[r*dims+k] /= a[kp1_k];
     a[kp1_k] = 2.0 * a[kp1_k] * a[kp1_k] / (sigma + a[kp1_k] * a[kp1_k]);
    }
   
   // Set the symmetric entry, needs info from above...
    a[k*dims + k+1] = sqrt(sigma + x0*x0);
   
   // Update the matrix with the householder transform (Make use of symmetry)...
    // Calculate p/beta, store in d...
     for (c=k+1; c<dims; c++)
     {
      d[c] = a[c*dims + k+1]; // First entry of v is 1.
      for (r=k+2; r<dims; r++) d[c] += a[r*dims+k] * a[c*dims+r];
     }
     
    // Calculate w, replace p with it in d...
     float mult = d[k+1];
     for (r=k+2; r<dims; r++) mult += a[r*dims+k] * d[r];
     mult *= 0.5 * a[kp1_k] * a[kp1_k];
     
     d[k+1] = a[kp1_k] * d[k+1] - mult;
     for (c=k+2; c<dims; c++) d[c] = a[kp1_k] * d[c] - mult * a[c*dims + k];

    // Apply the update - make use of symmetry by only calculating the lower
    // triangular set...
     // First column where first entry of v being 1 matters...
      a[(k+1)*dims + k+1] -= 2.0 * d[k+1];
      for (r=k+2; r<dims; r++) a[r*dims + k+1] -= a[r*dims + k] * d[k+1] + d[r];

     // Remainning columns...
      for (c=k+2; c<dims; c++)
      {
       for (r=c; r<dims; r++)
       {
        a[r*dims + c] -= a[r*dims + k] * d[c] + a[c*dims + k] * d[r];
       }
      }
     
     // Do the mirroring...
      for (r=k+1; r<dims; r++)
      {
       for (c=r+1; c<dims; c++) a[r*dims + c] = a[c*dims + r];
      }
  }


 // Use the stored sub-diagonal house-holder vectors to initialise q...
  for (r=0; r<dims; r++)
  {
   for (c=0; c<dims; c++) q[r*dims + c] = (c==r) ? 1.0 : 0.0; 
  }
  
  for (k=dims-3; k>=0; k--)
  {
   // Arrange for v to start with 1 - avoids special cases...
    float beta = a[(k+1)*dims + k];
    a[(k+1)*dims + k] = 1.0;
   
   // Update q, column by column...
    for (c=k+1; c<dims; c++)
    {
     // Copy column to tempory storage...
      for (r=k+1; r<dims; r++) d[r] = q[r*dims + c];

     // Update each row in column...
      for (r=k+1; r<dims; r++)
      {
       float mult = beta * a[r*dims + k];
       for (i=k+1; i<dims; i++) q[r*dims + c] -= mult * a[i*dims + k] * d[i];
      }
    }
  }


 // Now perform QR iterations till we have a diagonalised - at which point it
 // will be the eigenvalues... (Update q as we go.)
  // These parameters decide how many iterations are required...
   static const int max_iters = 64; // Maximum iters per value pair.
   int iters = 0; // Number of iterations done on current value pair.

   int all_good = 0; // Return value -set to nonzero if iters ever reaches max_iters.
 
  // Range of sub-matrix being processed - start is inclusive, end exclusive.
   int start = dims; // Set to force recalculate.
   int end = dims;

  // (Remember that below code ignores the sub-diagonal, as its a mirror of the super diagonal.)
  while (1)
  {
   // Move end up as far as possible, finish if done...
    int pend = end;
    while (1)
    {
     int em1 = end-1;
     int em2 = end-2;
     float tol = epsilon * (fabs(a[em2*dims + em2]) + fabs(a[em1*dims + em1]));
     if (fabs(a[em2*dims + em1])<tol)
     {
      end -= 1;
      if (end<2) break;
     }
     else break;
    }
       
    if (pend==end)
    {
     iters += 1;
     if (iters==max_iters)
     {
      all_good = 1;
      if (end==2) break;
      iters = 0;
      end -= 1;
      continue;
     }
    }
    else
    {
     if (end<2) break;
     iters = 0;
    }


   // If end has caught up with start recalculate it...
    if ((start+2)>end)
    {
     start = end-2;
     while (start>0)
     {
      int sm1 = start-1;
      float tol = epsilon * (fabs(a[sm1*dims + sm1]) + fabs(a[start*dims + start]));
      if (fabs(a[sm1*dims + start])>=tol) start -= 1;
                                     else break;
     }
    }


   // Do the QR step, with lots of juicy optimisation...
    // Calculate eigenvalue of trailing 2x2 matrix...
     int em1 = end-1;
     int em2 = end-2;
     float temp = 0.5 * (a[em2*dims + em2] - a[em1*dims + em1]);
     float sign_temp = (temp<0.0) ? (-1.0) : 1.0;
     float div = temp + sign_temp * sqrt(temp*temp + a[em2*dims + em1] * a[em2*dims + em1]);
     float tev = a[em1*dims + em1] - a[em2*dims + em1] * a[em2*dims + em1] / div;

    // Calculate and apply relevant sequence of givens transforms to
    // flow the numbers down the super/sub-diagonals...
     float x = a[start*dims + start] - tev;
     float z = a[start*dims + start+1];
     for (k=start; ; k++)
     {
      // Calculate givens transform...
       float gc = 1.0;
       float gs = 0.0;
       if (fabs(z)>epsilon)
       {
        if (fabs(z)>fabs(x))
        {
         float r = -x / z;
         gs = 1.0 / sqrt(1.0 + r*r);
         gc = gs * r;
        }
        else
        {
         float r = -z / x;
         gc = 1.0 / sqrt(1.0 + r*r);
         gs = gc * r;
        }
       }
       
       float gcc = gc * gc;
       float gss = gs * gs;


      // Update matrix q (Post multiply)...
       for (r=0; r<dims; r++)
       {
        float ck  = q[r*dims + k];
        float ck1 = q[r*dims + k+1];
        q[r*dims + k]   = gc*ck - gs*ck1;
        q[r*dims + k+1] = gs*ck + gc*ck1;
       }


      // Update matrix a...
       // Conditional on not being at start of range...
        if (k!=start) a[(k-1)*dims + k] = gc*x - gs*z;
       
       // Non-conditional...
       {
        float e = a[k*dims + k];
        float f = a[(k+1)*dims + k+1];
        float i = a[k*dims + k+1];
       
        a[k*dims + k] = gcc*e + gss*f - 2.0*gc*gs*i;
        a[(k+1)*dims + k+1] = gss*e + gcc*f + 2.0*gc*gs*i;
        a[k*dims + k+1] = gc*gs*(e-f) + (gcc - gss)*i;
        x = a[k*dims + k+1];
       }

       // Conditional on not being at end of range...
        if (k!=end-2)
        {
         z = -gs*a[(k+1)*dims + k+2]; // a[k][k+2]
         a[(k+1)*dims + k+2] *= gc;
        }
        else break;
     }
  }


  // Fill in the diagonal...
   for (i=0; i<dims; i++) d[i] = a[i*dims + i];
 
 // Return the status of the run...
  return all_good;  
}



int symmetric_eigen(int dims, float * a, float * q, float * d)
{
 int ret = symmetric_eigen_raw(dims, a, q, d);
 
 // Sort the eigen-values to be decreasing (rather inefficient)...
  int r, c, cc;
  for (cc=1; cc<dims; cc++)
  {
   for (c=cc; c>0; c--)
   {
    if (d[c]<d[c-1]) break;
    
    float t = d[c];
    d[c] = d[c-1];
    d[c-1] = t;

    for (r=0; r<dims; r++)
    {
     int b = r*dims;
     t = q[b+c];
     q[b+c] = q[b+c-1];
     q[b+c-1] = t;
    }
   }
 }

 return ret; 
}



int symmetric_eigen_abs(int dims, float * a, float * q, float * d)
{
 int ret = symmetric_eigen_raw(dims, a, q, d);
 
 // Sort the eigen-values to be decreasing (rather inefficient)...
  int r, c, cc;
  for (cc=1; cc<dims; cc++)
  {
   for (c=cc; c>0; c--)
   {
    if (fabs(d[c])<fabs(d[c-1])) break;
    
    float t = d[c];
    d[c] = d[c-1];
    d[c-1] = t;

    for (r=0; r<dims; r++)
    {
     int b = r*dims;
     t = q[b+c];
     q[b+c] = q[b+c-1];
     q[b+c-1] = t;
    }
   }
 }

 return ret; 
}
