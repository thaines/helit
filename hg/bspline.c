// Copyright 2016 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

#include "bspline.h"

#include <math.h>



inline float B(int degree, float x)
{
 switch(degree)
 {
  case 0:
  {
   // Base case, of basic step function (nearest neighbour)...
    x = fabs(x);
    if (x<0.5) return 1.0;
    if (x>0.5) return 0.0;
    return 0.5;
  }
  
  case 1:
  {
   // Degree 1, which is a triangle for linear interpolation...
    x = fabs(x);
    if (x<1.0) return 1.0 - x;
    else return 0.0;
  }
  
  case 2:
  {
   // Degree 2, for the quadratic case...
    x = fabs(x);
    if (x<=0.5)
    {
     x += 1.5;
     return -1 * x * x  + 3*x - 1.5;
    }
    else
    {
     if (x<1.5)
     {
      x = 1.5 - x;
      return 0.5 * x * x;
     }
     else
     {
      return 0.0; 
     }
    }
  }
  
  case 3:
  {
   // Degree 3, cubic case...
    x = fabs(x);
    if (x<=1.0)
    {
     x += 2.0;
     float xx = x * x;
     return 0.5 * xx*x - 4.0*xx + 10.0*x - (22.0/3.0);
    }
    else
    {
     if (x<2.0)
     {
      x += 2.0;
      float xx = x * x;
      return (-1.0/6.0)*xx*x + 2.0*xx - 8.0*x + (32.0/3.0);
     }
     else
     {
      return 0.0; 
     }
    }
  }
  
  default:
  {
   // Recursive case - could get expensive for high degree, but degree should remain low...
    float range = (degree+1) * 0.5;
    if ((x<-range)||(x>range)) return 0.0;
  
    float a = ((range + x) / degree) * B(degree-1, x+0.5);
    float b = ((range - x) / degree) * B(degree-1, x-0.5);
  
    return a + b;
  }
 }
}



float SampleB(int degree, float y, float x, PyArrayObject * data)
{
 // Evaluate B-spline weights for both axes...
  static const int max_degree = 5; // Can be changed if needed, but degree 5 is already beyond sensible.
  
  int dy;
  int iy = (int)floorf(y+0.5);
  float by[max_degree*2+1];
  
  for (dy=-degree; dy<=degree; dy++)
  {
   by[degree+dy] = B(degree, iy+dy - y);
  }
  
  int dx;
  int ix = (int)floorf(x+0.5);
  float bx[max_degree*2+1];
  
  for (dx=-degree; dx<=degree; dx++)
  {
   bx[degree+dx] = B(degree, ix+dx - x);
  }
  
 // Now loop the relevant range, summing the output weighted by the relevant weights and handling boundary conditions...
  float ret = 0.0;
  int shape[2] = {PyArray_SHAPE(data)[0], PyArray_SHAPE(data)[1]};
  
  for (dy=-degree; dy<=degree; dy++)
  {
   for (dx=-degree; dx<=degree; dx++)
   {
    int sy = iy+dy;
    if (sy<0) sy = 0;
    if (sy>=shape[0]) sy = shape[0] - 1;
    
    int sx = ix+dx;
    if (sx<0) sx = 0;
    if (sx>=shape[1]) sx = shape[1] - 1;

    ret += by[degree+dy] * bx[degree+dx] * *(float*)PyArray_GETPTR2(data, sy, sx);
   }
  }
  
 // Return the evaluated value...
  return ret;
}



float LayerSampleB(int degree, int layer, float y, float x, PyArrayObject * data)
{
 // Evaluate B-spline weights for both axes...
  static const int max_degree = 5; // Can be changed if needed, but degree 5 is already beyond sensible.
  
  int dy;
  int iy = (int)floorf(y+0.5);
  float by[max_degree*2+1];
  
  for (dy=-degree; dy<=degree; dy++)
  {
   by[degree+dy] = B(degree, iy+dy - y);
  }
  
  int dx;
  int ix = (int)floorf(x+0.5);
  float bx[max_degree*2+1];
  
  for (dx=-degree; dx<=degree; dx++)
  {
   bx[degree+dx] = B(degree, ix+dx - x);
  }
  
 // Now loop the relevant range, summing the output weighted by the relevant weights and handling boundary conditions...
  float ret = 0.0;
  int shape[2] = {PyArray_SHAPE(data)[1], PyArray_SHAPE(data)[2]};
  
  for (dy=-degree; dy<=degree; dy++)
  {
   for (dx=-degree; dx<=degree; dx++)
   {
    int sy = iy+dy;
    if (sy<0) sy = 0;
    if (sy>=shape[0]) sy = shape[0] - 1;
    
    int sx = ix+dx;
    if (sx<0) sx = 0;
    if (sx>=shape[1]) sx = shape[1] - 1;

    ret += by[degree+dy] * bx[degree+dx] * *(float*)PyArray_GETPTR3(data, layer, sy, sx);
   }
  }
  
 // Return the evaluated value...
  return ret;
}



void MultivariateSampleB(int degree, float y, float x, int shape[2], int channels, PyArrayObject ** image, float * out)
{
 // Evaluate B-spline weights for both axes...
  static const int max_degree = 5; // Can be changed if needed, but degree 5 is already beyond sensible.
  
  int dy;
  int iy = (int)floorf(y+0.5);
  float by[max_degree*2+1];
  
  for (dy=-degree; dy<=degree; dy++)
  {
   by[degree+dy] = B(degree, iy+dy - y);
  }
  
  int dx;
  int ix = (int)floorf(x+0.5);
  float bx[max_degree*2+1];
  
  for (dx=-degree; dx<=degree; dx++)
  {
   bx[degree+dx] = B(degree, ix+dx - x);
  }
  
 // Zero the output...
  int i;
  for (i=0; i<channels; i++)
  {
   out[i] = 0.0;
  }
  
 // Now loop the relevant range, summing the output weighted by the relevant weights and handling boundary conditions...
  for (dy=-degree; dy<=degree; dy++)
  {
   for (dx=-degree; dx<=degree; dx++)
   {
    int sy = iy+dy;
    if (sy<0) sy = 0;
    if (sy>=shape[0]) sy = shape[0] - 1;
    
    int sx = ix+dx;
    if (sx<0) sx = 0;
    if (sx>=shape[1]) sx = shape[1] - 1;
    
    float w = by[degree+dy] * bx[degree+dx];
    
    for (i=0; i<channels; i++)
    {
     out[i] += w * *(float*)PyArray_GETPTR2(image[i], sy, sx);
    }
   }
  }
}



void PrepareBSpline(void)
{
 _import_array();
}
