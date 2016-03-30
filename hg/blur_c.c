#include "blur_c.h"

// Copyright 2016 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



static PyObject * Gaussian(PyObject * self, PyObject * args, PyObject * kw)
{
 int i;
 
 // Parse the arguments...
  PyArrayObject * data;
  PyArrayObject * out;
  PyArrayObject * sd = NULL;
  PyArrayObject * derivative = NULL;
  float quality = 4.0;
  PyArrayObject * out_weight = NULL;
  
  static char * kw_list[] = {"data", "out", "sd", "derivative", "quality", "weight", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kw, "O!O!|O!O!fO!", kw_list, &PyArray_Type, &data, &PyArray_Type, &out, &PyArray_Type, &sd, &PyArray_Type, &derivative, &quality, &PyArray_Type, &out_weight)) return NULL;
 
  
 // Verify the input...
  const int dims = PyArray_NDIM(data);
  if (dims!=PyArray_NDIM(out))
  {
   PyErr_SetString(PyExc_RuntimeError, "Data matrices 'data' and 'out' must have the same number of dimensions.");
   return NULL;
  }
  
  for (i=0; i<dims; i++)
  {
   if (PyArray_SHAPE(data)[i]!=PyArray_SHAPE(out)[i])
   {
    PyErr_SetString(PyExc_RuntimeError, "Data matrices 'data' and 'out' must have the same shape.");
    return NULL; 
   }
  }
  
  if ((PyArray_DESCR(data)->kind!='f')||(PyArray_DESCR(data)->elsize!=sizeof(float)))
  {
   PyErr_SetString(PyExc_RuntimeError, "Data matrix 'data' must be of type float32");
   return NULL;
  }
  
  if ((PyArray_DESCR(out)->kind!='f')||(PyArray_DESCR(out)->elsize!=sizeof(float)))
  {
   PyErr_SetString(PyExc_RuntimeError, "Data matrix 'out' must be of type float32");
   return NULL;
  }
  
  if (sd!=NULL)
  {
   if (PyArray_NDIM(sd)!=1)
   {
    PyErr_SetString(PyExc_RuntimeError, "Vector of standard deviations should be a vector, i.e. 1D.");
    return NULL;
   }
   
   if (PyArray_SHAPE(sd)[0]!=dims)
   {
    PyErr_SetString(PyExc_RuntimeError, "Length of standard deviations vector must match number of dimensions of data/out.");
    return NULL;
   }
    
   if ((PyArray_DESCR(sd)->kind!='f')||(PyArray_DESCR(sd)->elsize!=sizeof(float)))
   {
    PyErr_SetString(PyExc_RuntimeError, "Standard deviations vector must be of type float32");
    return NULL;
   }
  }
  
  if (derivative!=NULL)
  {
   if (PyArray_NDIM(derivative)!=1)
   {
    PyErr_SetString(PyExc_RuntimeError, "Vector of derivative indicator variables should be a vector, i.e. 1D.");
    return NULL;
   }
   
   if (PyArray_SHAPE(derivative)[0]!=dims)
   {
    PyErr_SetString(PyExc_RuntimeError, "Length of derivative indicator variables must match number of dimensions of data/out.");
    return NULL;
   }
    
   if (PyArray_DESCR(derivative)->kind!='i')
   {
    PyErr_SetString(PyExc_RuntimeError, "Derivative indicator variables must be of integer type.");
    return NULL;
   }
  }
  
  if (quality<=0.0)
  {
   PyErr_SetString(PyExc_RuntimeError, "Quality parameter should be positive.");
   return NULL;
  }
  
  if (out_weight!=NULL)
  {
   if ((PyArray_DESCR(out_weight)->kind!='f')||(PyArray_DESCR(out_weight)->elsize!=sizeof(float)))
   {
    PyErr_SetString(PyExc_RuntimeError, "Data matrix 'weight' must be of type float32");
    return NULL;
   }
   
   if (dims!=PyArray_NDIM(out_weight))
   {
    PyErr_SetString(PyExc_RuntimeError, "Output matrix 'weight' must have the same number of dimensions as the 'data'/'out' matrices.");
    return NULL;
   }
   
   for (i=0; i<dims; i++)
   {
    if (PyArray_SHAPE(data)[i]!=PyArray_SHAPE(out_weight)[i])
    {
     PyErr_SetString(PyExc_RuntimeError, "Output matrix 'weight' must have the same shape as 'data'/'out'.");
     return NULL;
    }
   }   
  }
  
  
 // Create the intermediate structure...
  const int count = PyArray_SIZE(data);
  IncMean * im = (IncMean*)malloc(count * sizeof(IncMean));
  
  
 // Create an array to store the numbers for each blur in - find out the maximum range required and make it that large...
  float max = 0.0;
  if (sd!=NULL)
  {
   for (i=0; i<dims; i++)
   {
    float value = *(float*)PyArray_GETPTR1(sd, i);
    if (value>max) max = value; 
   }
  }
  else
  {
   max = M_SQRT2;
  }
  max *= quality;
  
  float * weight = (float*)malloc(((int)ceil(max) + 1) * sizeof(float));
  
  
 // Fill the structure in with the starting values, taking care to handle NaNs and inf...
  npy_intp * pos = (npy_intp*)malloc(dims * sizeof(npy_intp));
  for (i=0; i<dims; i++) pos[i] = 0;
  
  for (i=0; i<count; i++)
  {
   // Get current value...
    float value = *(float*)PyArray_GetPtr(data, pos);
   
   // Store the starting state, into position 0, with zero weight used when its a dodgy value...
    if (isfinite(value))
    {
     im[i].mean[0] = value;
     im[i].weight[0] = 1.0; 
    }
    else
    {
     im[i].mean[0] = 0.0;
     im[i].weight[0] = 0.0;
    }
    
   // Move to next position...
    int j = dims-1;
    while (j>=0)
    {
     pos[j] += 1;
     if (pos[j]<PyArray_SHAPE(data)[j]) break;
    
     pos[j] = 0;
     j -= 1;
    }
  }
  
  
 // Loop and do each dimension in turn, fliping between the two buffers...
  for (i=0; i<dims; i++)
  {
   // From and to indices for incrimental mean array...
    int fi = i%2;
    int ti = (i+1)%2;
    
   // Zero out the to array, ready to receive...
    int j;
    for (j=0; j<count; j++)
    {
     im[j].mean[ti] = 0.0;
     im[j].weight[ti] = 0.0;
    }
    
   // Calculate the weights for the Gaussian blur, even when its the derivative; also calculate seperate normalisation terms for derivative...
    float tsd = (sd!=NULL) ? (*(float*)PyArray_GETPTR1(sd, i)) : M_SQRT2;
    if (tsd<1e-6) continue; // Skip if no blur in this dimension.
    int range = (int)ceil(tsd * quality) + 1;
    
    float norm = 1.0 / (tsd * sqrt(2.0 * M_PI));
    
    char der = (derivative==NULL) ? 0 : *(char*)PyArray_GETPTR1(derivative, i); // This is ignoring variable size and assuming Intel byte order - big fat bug if you are on another architecture.
    float sign = (der<0) ? -1.0 : 1.0;
    //if (abs(der)==1) norm_extra = 1.0 / (tsd * tsd);
    //if (abs(der)==2) norm_extra = 1.0 / (tsd * tsd * tsd * tsd);
    
    for (j=0; j<=range; j++)
    {
     weight[j] = norm * exp(-0.5 * (j*j) / (tsd*tsd));
    }
    
   // Loop and process each pixel in turn, updating via incrimental means and handling boundary conditions for the relevant neighbours in the current dimension...
    int upper = PyArray_SHAPE(data)[i] - 1;
    for (j=0; j<dims; j++) pos[j] = 0;
    
    int step = PyArray_STRIDES(data)[i] / PyArray_STRIDES(data)[dims-1];
    
    for (j=0; j<count; j++)
    {
     // Loop the range of positions for this value...
      int low = pos[i] - range;
      int high = pos[i] + range;
      
      if (low<0) low = 0;
      if (high>upper) high = upper;
      
      int k;
      for (k=low; k<=high; k++)
      {
       int os = k - pos[i];
       int oi = j + step * os;
       float w = weight[abs(os)] * im[j].weight[fi];
       if (w<1e-6) continue;
       
       float f = sign;
       switch(abs(der))
       {
        case 0: break;
        case 1: f *= os; break;
        case 2: f *= os * os - 1.0; break;
        case 3: f *= os * os * os - 3.0 * os; break;
        case 4: f *= os * os * os * os - 6.0 * os * os + 3.0; break;
        case 5: f *= os * os * os * os * os - 10.0 * os * os * os + 15.0 * os; break;
        case 6: f *= os * os * os * os * os * os - 15.0 * os * os * os * os + 45.0 * os * os - 15.0; break;
       }
       
       im[oi].weight[ti] += w;
       im[oi].mean[ti] += (f * im[j].mean[fi] - im[oi].mean[ti]) * w / im[oi].weight[ti];
      }
      
     // Move to next position...
      k = dims-1;
      while (k>=0)
      {
       pos[k] += 1;
       if (pos[k]<PyArray_SHAPE(out)[k]) break;
    
       pos[k] = 0;
       k -= 1;
      }
    }
  }
  
  
 // Copy the result into out...
  for (i=0; i<dims; i++) pos[i] = 0;
  
  for (i=0; i<count; i++)
  {
   // Copy over...
    float value = im[i].mean[dims%2];
    *(float*)PyArray_GetPtr(out, pos) = value;
    
    if (out_weight!=NULL)
    {
     float w = im[i].weight[dims%2];
     *(float*)PyArray_GetPtr(out_weight, pos) = w;
    }
  
   // Move to next position...
    int j = dims-1;
    while (j>=0)
    {
     pos[j] += 1;
     if (pos[j]<PyArray_SHAPE(out)[j]) break;
    
     pos[j] = 0;
     j -= 1;
    }
  }
  
  
 // Free the memory of the temporary structure...
  free(weight);
  free(pos);
  free(im);

 
 // Return None...
  Py_INCREF(Py_None);
  return Py_None;  
}



static PyMethodDef blur_c_methods[] =
{
 {"Gaussian", (PyCFunction)Gaussian, METH_VARARGS | METH_KEYWORDS, "Does a Gaussian blur on an n dimensional numpy array of type float32. Takes the following arguments, in the following order or with keywords: {data : An nd numpy array of values - can contain inf and NaN, which will be ignored; out : array identical to data which will be overwriten with the output. Can in fact be the same array as data; sd : 1D array giving standard deviation for each dimension, so length must match number of dimensions of data. Type must be float32, and it will handle values of zero correctly with a noop. If not provided it defaults to sqrt(2) for all values; derivative - an optional integer array whose length matches the number of dimensions. A value of 0 means to use the normal Gaussian for that dimension, a value of 1 its derivative, a value of -1 its mirrored derivative. Also supports 2/-2 for the second derivative etc. upto the 6th derivative. It rarely make sense to have more than one non-zero value; quality : Number of standard deviations out to go - defaults to 4, weight : An array, same shape as input/output of float32 type, into which the weights will be written - should be 1 in all cases.}. A little different from most implimentations because it drops values outside the array/numbers that are not finite, and renormalises the output values accordingly - does the correct thing for data that contains gaps in other words."},
 {NULL}
};



#ifndef PyMODINIT_FUNC
#define PyMODINIT_FUNC void
#endif

PyMODINIT_FUNC initblur_c(void)
{
 Py_InitModule3("blur_c", blur_c_methods, "Provides a simple method that does a Gaussian blur on a numpy array, arbitrary number of dimensions. A little different from typical versions as it handles the edge by renormalising the sum to ignore edge pixels, rather than any extending. Also handles numbers that are not finite by ignoring them and not counting their weight when normalising the outputs -- essentially this is safe when there are holes in the data.");
 
 import_array();
}
