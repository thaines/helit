# -*- coding: utf-8 -*-

# Copyright 2010 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



# Some code used by the smo module...

cacheCode = """
// Right now only provides the most basic of caching, by pre-caclulating the diagonal, which is admitedly used very heavilly...
npy_intp * dmSize;
double * dm;
double * diagBuff;

void cacheBegin(npy_intp * dmSizeIn, double * dmIn)
{
 dmSize = dmSizeIn;
 dm = dmIn;

 diagBuff = (double*)malloc(dmSize[0] * sizeof(double));
 for (int i=0;i<dmSize[0];i++)
 {
  double * vec = dm + i*dmSize[1];
  diagBuff[i] = kernel(dmSize[1], vec, vec);
 }
}

double cacheK(int a,int b) // Indices of two vectors from the data matrix.
{
 if (a!=b)
 {
  return kernel(dmSize[1], dm + a*dmSize[1], dm + b*dmSize[1]);
 }
 else
 {
  return diagBuff[a];
 }
}

void cacheEnd()
{
 free(diagBuff);
}
"""



smoCoreCode = """
// Constant...
 const double eps = 1e-3;

// Initialise the cache...
 cacheBegin(Ndm,dm);

// Iterate until convergance...
 int pv1 = -1;
 int pv2 = -1;
 //long long int maxIter = ((long long int)Ny[0])*((long long int)Ny[0])*2; // Cap iters, to avoid any chance of it getting stuck in an infinite loop if a cyclic set of edits were to appear. Note that this number is really high - typically it going to need a few more than Ny[0] iterations.
 //for (long long int iter=0;iter<maxIter;iter++)
 while (true)
 {
  // Determine which pair we are going to optimise, break if all are optimised...
  double maxG = -1e100;
  double minG = 1e100;
  double minObj = 1e100;
   // Select the first member of the pair, v1...
    int v1 = -1;
    for (int i=0;i<Ny[0];i++)
    {
     double c = (y[i]<0)?cn:cp;
     if (((y[i]>0)&&(alpha[i]<c))||((y[i]<0)&&(alpha[i]>0.0)))
     {
      double g = -y[i] * gradient[i];
      if (g >= maxG)
      {
       v1 = i;
       maxG = g;
      }
     }
    }

   // Select the second member of the pair, v2...
    int v2 = -1;
    double a;
    for (int i=0;i<Ny[0];i++)
    {
     double c = (y[i]<0)?cn:cp;
     if (((y[i]>0)&&(alpha[i]>0.0))||((y[i]<0)&&(alpha[i]<c)))
     {
      double g = -y[i] * gradient[i];
      if (g <= minG) minG = g;

      double b = maxG - g;
      if (b>0)
      {
       double na = cacheK(v1,v1) + cacheK(i,i) - 2.0*cacheK(v1,i);
       if (na<=0.0) na = 1e12;

       double obj = -(b*b)/a;
       if (obj <= minObj)
       {
        if ((i!=pv2)&&(v1!=pv1)) // Prevents it selecting the same pair twice in a row - this can cause an infinite loop.
        {
         v2 = i;
         a = na;
         minObj = obj;
        }
       }
      }
     }
    }

   // Check for convergance/algorithm has done its best...
    if (v2==-1) break;
    if ((maxG-minG)<eps) break;

    pv1 = v1;
    pv2 = v2;


  // Calculate new alpha values, to reduce the objective function...
   double b = -y[v1]*gradient[v1] + y[v2]*gradient[v2];

   double oldA1 = alpha[v1];
   double oldA2 = alpha[v2];

   alpha[v1] += y[v1]*b/a;
   alpha[v2] -= y[v2]*b/a;

  // Correct for alpha being out of range...
   double sum = y[v1]*oldA1 + y[v2]*oldA2;

   double c = (y[v1]<0)?cn:cp;
   if (alpha[v1]<0.0) alpha[v1] = 0.0;
   else { if (alpha[v1]>c) alpha[v1] = c; }
   alpha[v2] = y[v2] * (sum - y[v1]*alpha[v1]);

   c = (y[v2]<0)?cn:cp;
   if (alpha[v2]<0.0) alpha[v2] = 0.0;
   else { if (alpha[v2]>c) alpha[v2] = c; }
   alpha[v1] = y[v1] * (sum - y[v2]*alpha[v2]);

  // Update the gradient...
   double dA1 = alpha[v1] - oldA1;
   double dA2 = alpha[v2] - oldA2;

   for (int i=0;i<Ny[0];i++)
   {
    gradient[i] += y[i] * y[v1] * cacheK(i,v1) * dA1;
    gradient[i] += y[i] * y[v2] * cacheK(i,v2) * dA2;
   }
 }

// Deinitialise the cache...
 cacheEnd();
"""