// Copyright 2016 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

#include "transform_c.h"

#include "bspline.h"



// Support method, that is actually exposed to the system as a whole - image must be a dictionary of equally sized 2D float arrays, indexed by strings, where the entry 'mask' must exist and must be uint8...
static void FillMasked(PyObject * image)
{
 // Extract the image...
  int channels = PyDict_Size(image) - 1;
  PyArrayObject ** field = (PyArrayObject **)malloc(channels * sizeof(PyArrayObject*)); // All borrowed
  
  Py_ssize_t pos = 0;
  PyObject * key;
  PyObject * value;
  
  int i = 0;
  PyArrayObject * mask = NULL;
  
  while (PyDict_Next(image, &pos, &key, &value))
  {
   if (PyArray_TYPE((PyArrayObject*)value)==NPY_FLOAT)
   {
    field[i] = (PyArrayObject*)value;
    i += 1;
   }
   else
   {
    mask = (PyArrayObject*)value;
   }
  }


 // Create a data structure to make this approach efficient - effectively distance to the current value, so it can be updated...
  const npy_intp * shape = PyArray_SHAPE(field[0]);
  int * distance = malloc(shape[0] * shape[1] * sizeof(int));
  
  int y, x;
  for (y=0; y<shape[0]; y++)
  {
   for (x=0; x<shape[1]; x++)
   {
    if (0==*(unsigned char*)PyArray_GETPTR2(mask, y, x))
    {
     distance[y*shape[1] + x] = -1; // Not yet assigned.
    }
    else
    {
     distance[y*shape[1] + x] = 0; // Known value.
    }
   }
  }
  
 // Do passes in the four compass directions...
  // +ve x...
   for (y=0; y<shape[0]; y++)
   {
    for (x=1; x<shape[1]; x++)
    {
     int target = y*shape[1] + x;
     int source = target - 1;
     if ((distance[source]>=0) && ((distance[target]<0) || (distance[target]>(distance[source]+1))))
     {
      distance[target] = distance[source] + 1;
      
      for (i=0; i<channels; i++)
      {
       *(float*)PyArray_GETPTR2(field[i], y, x) = *(float*)PyArray_GETPTR2(field[i], y, x-1);
      }
     }
    }
   }
   
  // -ve x...
   for (y=0; y<shape[0]; y++)
   {
    for (x=shape[1]-2; x>=0; x--)
    {
     int target = y*shape[1] + x;
     int source = target + 1;
     if ((distance[source]>=0) && ((distance[target]<0) || (distance[target]>(distance[source]+1))))
     {
      distance[target] = distance[source] + 1;
      
      for (i=0; i<channels; i++)
      {
       *(float*)PyArray_GETPTR2(field[i], y, x) = *(float*)PyArray_GETPTR2(field[i], y, x+1);
      }
     }
    }
   }
   
  // +ve y...
   for (y=1; y<shape[0]; y++)
   {
    for (x=0; x<shape[1]; x++)
    {
     int target = y*shape[1] + x;
     int source = target - shape[1];
     if ((distance[source]>=0) && ((distance[target]<0) || (distance[target]>(distance[source]+1))))
     {
      distance[target] = distance[source] + 1;
      
      for (i=0; i<channels; i++)
      {
       *(float*)PyArray_GETPTR2(field[i], y, x) = *(float*)PyArray_GETPTR2(field[i], y-1, x);
      }
     }
    }
   }
   
  // -ve y...
   for (y=shape[0]-2; y>=0; y--)
   {
    for (x=0; x<shape[1]; x++)
    {
     int target = y*shape[1] + x;
     int source = target + shape[1];
     if ((distance[source]>=0) && ((distance[target]<0) || (distance[target]>(distance[source]+1))))
     {
      distance[target] = distance[source] + 1;
      
      for (i=0; i<channels; i++)
      {
       *(float*)PyArray_GETPTR2(field[i], y, x) = *(float*)PyArray_GETPTR2(field[i], y+1, x);
      }
     }
    }
   }
  
 // Clean up...
  free(distance);
  free(field);
}


// Python interface to above, with error checking...
static PyObject * FillMasked_py(PyObject * module, PyObject * args)
{
 // Handle the parameter...
  PyObject * image;
  if (!PyArg_ParseTuple(args, "O!", &PyDict_Type, &image)) return NULL;
 
 // Validate...
  Py_ssize_t pos = 0;
  PyObject * key;
  PyObject * value;
   
  int shape[2] = {-1,-1};
  PyArrayObject * in_mask = NULL;
   
  while (PyDict_Next(image, &pos, &key, &value))
  {
   if (PyString_Check(key)==0)
   {
    PyErr_SetString(PyExc_RuntimeError, "All dictionary keys when specifying an image must be strings.");
    return NULL; 
   }
   
   if (PyArray_Check(value)==0)
   {
    PyErr_SetString(PyExc_RuntimeError, "All dictionary values when specifying an image must be ndarray-s.");
    return NULL;  
   }
   
   PyArrayObject * data = (PyArrayObject*)value;
   
   if (PyArray_NDIM(data)!=2)
   {
    PyErr_SetString(PyExc_RuntimeError, "Image arrays must be 2D.");
     return NULL;   
   }
   
   if (shape[0]==-1)
   {
    shape[0] = PyArray_SHAPE(data)[0];
    shape[1] = PyArray_SHAPE(data)[1];
   }
   else
   {
    if ((shape[0]!=PyArray_SHAPE(data)[0]) || (shape[1]!=PyArray_SHAPE(data)[1]))
    {
     PyErr_SetString(PyExc_RuntimeError, "All image arrays must have the same shape.");
     return NULL; 
    }
   }
   
   int type = PyArray_TYPE(data);
   
   if (type!=NPY_FLOAT)
   {
    if ((type==NPY_UBYTE)&&(strcmp(PyString_AsString(key),"mask")==0))
    {
     // Its a mask - thats ok...
      in_mask = data;
    }
    else
    {
     PyErr_SetString(PyExc_RuntimeError, "All image channels must be float32, with the exception of a mask channel, which must be uint8.");
     return NULL; 
    }
   }
  }
 
 // Call through to actual workhorse, but only if there is a mask, as there is nothing to do otherwise...
  if (in_mask!=NULL)
  {
   FillMasked(image);
  }
  
 // Return None...
  Py_INCREF(Py_None);
  return Py_None; 
}



// Actual workhorse...
static PyObject * Transform(PyObject * module, PyObject * args)
{
 // Handle the parameters...
  PyArrayObject * hg;
  PyObject * image;
  int height = -1;
  int width = -1;
  int degree = 3;
  
  if (!PyArg_ParseTuple(args, "O!O!|iii", &PyArray_Type, &hg, &PyDict_Type, &image, &height, &width, &degree)) return NULL;
 
 
 // Validate...
  // hg...
   if ((PyArray_NDIM(hg)!=2)||(PyArray_SHAPE(hg)[0]!=3)||(PyArray_SHAPE(hg)[1]!=3))
   {
    PyErr_SetString(PyExc_RuntimeError, "Homography must be 3x3.");
    return NULL; 
   }
   
   if (PyArray_TYPE(hg)!=NPY_FLOAT32)
   {
    PyErr_SetString(PyExc_RuntimeError, "Homography must be of type float32");
    return NULL;
   }
  
  // image...
   Py_ssize_t pos = 0;
   PyObject * key;
   PyObject * value;
   
   int shape[2] = {-1,-1};
   PyArrayObject * in_mask = NULL;
   
   while (PyDict_Next(image, &pos, &key, &value))
   {
    if (PyString_Check(key)==0)
    {
     PyErr_SetString(PyExc_RuntimeError, "All dictionary keys when specifying an image must be strings.");
     return NULL; 
    }
   
    if (PyArray_Check(value)==0)
    {
     PyErr_SetString(PyExc_RuntimeError, "All dictionary values when specifying an image must be ndarray-s.");
     return NULL;  
    }
   
    PyArrayObject * data = (PyArrayObject*)value;
   
    if (PyArray_NDIM(data)!=2)
    {
     PyErr_SetString(PyExc_RuntimeError, "Image arrays must be 2D.");
     return NULL;   
    }
   
    if (shape[0]==-1)
    {
     shape[0] = PyArray_SHAPE(data)[0];
     shape[1] = PyArray_SHAPE(data)[1];
     
     if (height==-1) height = shape[0];
     if (width==-1) width = shape[1];
    }
    else
    {
     if ((shape[0]!=PyArray_SHAPE(data)[0]) || (shape[1]!=PyArray_SHAPE(data)[1]))
     {
      PyErr_SetString(PyExc_RuntimeError, "All image arrays must have the same shape.");
      return NULL; 
     }
    }
   
   int type = PyArray_TYPE(data);
   
   if (type!=NPY_FLOAT)
   {
    if ((type==NPY_UBYTE)&&(strcmp(PyString_AsString(key),"mask")==0))
    {
     // Its a mask - thats ok - no-op.
      in_mask = data;
    }
    else
    {
     PyErr_SetString(PyExc_RuntimeError, "All image channels must be float32, with the exception of a mask channel which must be uint8.");
     return NULL; 
    }
   }
  }
  
 // dimensions...
  if ((height<1)||(width<1))
  {
   PyErr_SetString(PyExc_RuntimeError, "Output dimensions must be positive.");
   return NULL;  
  }
 
 // degree...
  if ((degree<0)||(degree>5))
  {
   PyErr_SetString(PyExc_RuntimeError, "Polynomial degree must be between 0 and 5, inclusive.");
   return NULL;
  }

 
 // Run FillMasked, to simplify all of the below code...
  if (in_mask!=NULL)
  {
   FillMasked(image);
  }

  
 // Extract the image as PyArrayObject's...
  int channels = PyDict_Size(image);
  if (in_mask!=NULL) channels -= 1;
  const char ** keys = (const char**)malloc(channels * sizeof(char*)); // All borrowed
  PyArrayObject ** in = (PyArrayObject **)malloc(channels * sizeof(PyArrayObject*)); // All borrowed
  
  pos = 0;
  int i = 0;
  
  while (PyDict_Next(image, &pos, &key, &value))
  {
   if (PyArray_TYPE((PyArrayObject*)value)==NPY_FLOAT)
   {
    keys[i] = PyString_AsString(key);
    in[i] = (PyArrayObject*)value;
    i += 1;
   }
  }


 // Construct the return...
  PyObject * ret = PyDict_New();
  
  npy_intp os[2] = {height, width};
  PyArrayObject * out_mask = (PyArrayObject*)PyArray_SimpleNew(2, os, NPY_UINT8);
  PyDict_SetItemString(ret, "mask", (PyObject*)out_mask);
  Py_DECREF(out_mask);
  
  PyArrayObject ** out = (PyArrayObject **)malloc(channels * sizeof(PyArrayObject*)); // All borrowed
  for (i=0; i<channels; i++)
  {
   out[i] = (PyArrayObject*)PyArray_SimpleNew(2, os, NPY_FLOAT32);
   PyDict_SetItemString(ret, keys[i], (PyObject*)out[i]);
   Py_DECREF(out[i]);
  }

 
 // Loop and calculate each pixel in turn...
  float * temp = (float*)malloc(channels * sizeof(float));
  int y, x;
  for (y=0; y<height; y++)
  {
   for (x=0; x<width; x++)
   {
    // Apply homography and find the source coordinate, avoiding divide by zero...
     float sx = *(float*)PyArray_GETPTR2(hg, 0, 0) * x + *(float*)PyArray_GETPTR2(hg, 0, 1) * y + *(float*)PyArray_GETPTR2(hg, 0, 2);
     float sy = *(float*)PyArray_GETPTR2(hg, 1, 0) * x + *(float*)PyArray_GETPTR2(hg, 1, 1) * y + *(float*)PyArray_GETPTR2(hg, 1, 2);
     
     float sw = *(float*)PyArray_GETPTR2(hg, 2, 0) * x + *(float*)PyArray_GETPTR2(hg, 2, 1) * y + *(float*)PyArray_GETPTR2(hg, 2, 2);
     if (fabs(sw)<1e-12) sw = copysign(1e-12, sw);
     
     sx /= sw;
     sy /= sw;
     
    // Check its in bounds and hasn't hit a masked pixel...
     int cx = (int)floorf(sx+0.5);
     int cy = (int)floorf(sy+0.5);
     
     if ((cx<0)||(cx>=shape[1])||(cy<0)||(cy>=shape[0]))
     {
      *(unsigned char*)PyArray_GETPTR2(out_mask, y, x) = 0;
      continue;
     }
     
     if ((in_mask!=NULL) && (0==*(unsigned char*)PyArray_GETPTR2(in_mask, cy, cx)))
     {
      *(unsigned char*)PyArray_GETPTR2(out_mask, y, x) = 0;
      continue; 
     }
     
    // Indicate that this pixel is going to be valid...
     *(unsigned char*)PyArray_GETPTR2(out_mask, y, x) = 1;
     

    // B-spline interpolation...
     MultivariateSampleB(degree, sy, sx, shape, channels, in, temp);
     
    // Copy into output...
     for (i=0; i<channels; i++)
     {
      *(float*)PyArray_GETPTR2(out[i], y, x) = temp[i]; 
     }
   }
  }
  
 
 // Clean up...
  free(out);
  free(temp);
  free(in);
  free(keys);

  
 // Do the return...
  return ret;
}



// For just sampling random poitns...
static PyObject * Sample(PyObject * module, PyObject * args)
{
 // Handle the parameters...
  PyObject * image;
  PyArrayObject * locations;
  int degree = 3;
  
  if (!PyArg_ParseTuple(args, "O!O!|i", &PyDict_Type, &image, &PyArray_Type, &locations, &degree)) return NULL;
  
 
 // Validate...
  // image...
   Py_ssize_t pos = 0;
   PyObject * key;
   PyObject * value;
   
   int shape[2] = {-1,-1};
   PyArrayObject * in_mask = NULL;
   
   while (PyDict_Next(image, &pos, &key, &value))
   {
    if (PyString_Check(key)==0)
    {
     PyErr_SetString(PyExc_RuntimeError, "All dictionary keys when specifying an image must be strings.");
     return NULL; 
    }
   
    if (PyArray_Check(value)==0)
    {
     PyErr_SetString(PyExc_RuntimeError, "All dictionary values when specifying an image must be ndarray-s.");
     return NULL;  
    }
   
    PyArrayObject * data = (PyArrayObject*)value;
   
    if (PyArray_NDIM(data)!=2)
    {
     PyErr_SetString(PyExc_RuntimeError, "Image arrays must be 2D.");
     return NULL;   
    }
   
    if (shape[0]==-1)
    {
     shape[0] = PyArray_SHAPE(data)[0];
     shape[1] = PyArray_SHAPE(data)[1];
    }
    else
    {
     if ((shape[0]!=PyArray_SHAPE(data)[0]) || (shape[1]!=PyArray_SHAPE(data)[1]))
     {
      PyErr_SetString(PyExc_RuntimeError, "All image arrays must have the same shape.");
      return NULL; 
     }
    }
   
   int type = PyArray_TYPE(data);
   
   if (type!=NPY_FLOAT)
   {
    if ((type==NPY_UBYTE)&&(strcmp(PyString_AsString(key),"mask")==0))
    {
     // Its a mask - thats ok - no-op.
      in_mask = data;
    }
    else
    {
     PyErr_SetString(PyExc_RuntimeError, "All image channels must be float32, with the exception of a mask channel which must be uint8.");
     return NULL; 
    }
   }
  }
  
 // locations...
  if (PyArray_NDIM(locations)!=2)
  {
   PyErr_SetString(PyExc_RuntimeError, "Points array must be 2D.");
   return NULL;
  }
  
  if (PyArray_SHAPE(locations)[1]!=2)
  {
   PyErr_SetString(PyExc_RuntimeError, "Points array must have two columns.");
   return NULL;
  }
  
  if (PyArray_TYPE(locations)!=NPY_FLOAT)
  {
   PyErr_SetString(PyExc_RuntimeError, "Points array must be of type float32.");
   return NULL;
  }
 
 // degree...
  if ((degree<0) || (degree>5))
  {
   PyErr_SetString(PyExc_RuntimeError, "degree must be in 0 to 5, inclusive.");
   return NULL; 
  }
  
  
 // Run FillMasked, to simplify all of the below code...
  if (in_mask!=NULL)
  {
   FillMasked(image);
  }

  
 // Extract the image as PyArrayObject's...
  int channels = PyDict_Size(image);
  if (in_mask!=NULL) channels -= 1;
  const char ** keys = (const char**)malloc(channels * sizeof(char*)); // All borrowed
  PyArrayObject ** in = (PyArrayObject **)malloc(channels * sizeof(PyArrayObject*)); // All borrowed
  
  pos = 0;
  int i = 0;
  
  while (PyDict_Next(image, &pos, &key, &value))
  {
   if (PyArray_TYPE((PyArrayObject*)value)==NPY_FLOAT)
   {
    keys[i] = PyString_AsString(key);
    in[i] = (PyArrayObject*)value;
    i += 1;
   }
  }

  
 // Construct the return...
  PyObject * ret = PyDict_New();  
  npy_intp os = PyArray_SHAPE(locations)[0];
  
  PyArrayObject ** out = (PyArrayObject **)malloc(channels * sizeof(PyArrayObject*)); // All borrowed
  for (i=0; i<channels; i++)
  {
   out[i] = (PyArrayObject*)PyArray_SimpleNew(1, &os, NPY_FLOAT32);
   PyDict_SetItemString(ret, keys[i], (PyObject*)out[i]);
   Py_DECREF(out[i]);
  }


 // Loop and calculate each offset in turn...
  float * temp = (float*)malloc(channels * sizeof(float));
  int l;
  for (l=0; l<os; l++)
  {
   // Get location to query...
     float y = *(float*)PyArray_GETPTR2(locations, l, 0);
     float x = *(float*)PyArray_GETPTR2(locations, l, 1);
     
   // B-spline interpolation...
    MultivariateSampleB(degree, y, x, shape, channels, in, temp);
     
   // Copy into output...
    for (i=0; i<channels; i++)
    {
     *(float*)PyArray_GETPTR1(out[i], l) = temp[i];
    }
  }


 // Clean up...
  free(out);
  free(temp);
  free(in);
  free(keys);

  
 // Do the return...
  return ret;
}



// For extracting features...
static PyObject * Offsets(PyObject * module, PyObject * args)
{
 // Handle the parameters...
  PyObject * image;
  PyArrayObject * points;
  PyArrayObject * offsets;
  int degree = 3;
  
  if (!PyArg_ParseTuple(args, "O!O!O!|i", &PyDict_Type, &image, &PyArray_Type, &points, &PyArray_Type, &offsets, &degree)) return NULL;
 
 
 // Validate...
  // image...
   Py_ssize_t pos = 0;
   PyObject * key;
   PyObject * value;
   
   int shape[2] = {-1,-1};
   PyArrayObject * in_mask = NULL;
   
   while (PyDict_Next(image, &pos, &key, &value))
   {
    if (PyString_Check(key)==0)
    {
     PyErr_SetString(PyExc_RuntimeError, "All dictionary keys when specifying an image must be strings.");
     return NULL; 
    }
   
    if (PyArray_Check(value)==0)
    {
     PyErr_SetString(PyExc_RuntimeError, "All dictionary values when specifying an image must be ndarray-s.");
     return NULL;  
    }
   
    PyArrayObject * data = (PyArrayObject*)value;
   
    if (PyArray_NDIM(data)!=2)
    {
     PyErr_SetString(PyExc_RuntimeError, "Image arrays must be 2D.");
     return NULL;   
    }
   
    if (shape[0]==-1)
    {
     shape[0] = PyArray_SHAPE(data)[0];
     shape[1] = PyArray_SHAPE(data)[1];
    }
    else
    {
     if ((shape[0]!=PyArray_SHAPE(data)[0]) || (shape[1]!=PyArray_SHAPE(data)[1]))
     {
      PyErr_SetString(PyExc_RuntimeError, "All image arrays must have the same shape.");
      return NULL; 
     }
    }
   
   int type = PyArray_TYPE(data);
   
   if (type!=NPY_FLOAT)
   {
    if ((type==NPY_UBYTE)&&(strcmp(PyString_AsString(key),"mask")==0))
    {
     // Its a mask - thats ok - no-op.
      in_mask = data;
    }
    else
    {
     PyErr_SetString(PyExc_RuntimeError, "All image channels must be float32, with the exception of a mask channel which must be uint8.");
     return NULL; 
    }
   }
  }
  
 // points...
  if (PyArray_NDIM(points)!=2)
  {
   PyErr_SetString(PyExc_RuntimeError, "Points array must be 2D.");
   return NULL;
  }
  
  if (PyArray_SHAPE(points)[1]!=2)
  {
   PyErr_SetString(PyExc_RuntimeError, "Points array must have two columns.");
   return NULL;
  }
  
  if (PyArray_TYPE(points)!=NPY_FLOAT)
  {
   PyErr_SetString(PyExc_RuntimeError, "Points array must be of type float32.");
   return NULL;
  }
 
 // offsets...
  if (PyArray_NDIM(offsets)!=2)
  {
   PyErr_SetString(PyExc_RuntimeError, "Offsets array must be 2D.");
   return NULL;
  }
  
  if (PyArray_SHAPE(offsets)[1]!=2)
  {
   PyErr_SetString(PyExc_RuntimeError, "Offsets array must have two columns.");
   return NULL;
  }
  
  if (PyArray_TYPE(offsets)!=NPY_FLOAT)
  {
   PyErr_SetString(PyExc_RuntimeError, "Offsets array must be of type float32.");
   return NULL;
  }
 
 // degree...
  if ((degree<0) || (degree>5))
  {
   PyErr_SetString(PyExc_RuntimeError, "degree must be in 0 to 5, inclusive.");
   return NULL; 
  }
  
  
 // Run FillMasked, to simplify all of the below code...
  if (in_mask!=NULL)
  {
   FillMasked(image);
  }

  
 // Extract the image as PyArrayObject's...
  int channels = PyDict_Size(image);
  if (in_mask!=NULL) channels -= 1;
  const char ** keys = (const char**)malloc(channels * sizeof(char*)); // All borrowed
  PyArrayObject ** in = (PyArrayObject **)malloc(channels * sizeof(PyArrayObject*)); // All borrowed
  
  pos = 0;
  int i = 0;
  
  while (PyDict_Next(image, &pos, &key, &value))
  {
   if (PyArray_TYPE((PyArrayObject*)value)==NPY_FLOAT)
   {
    keys[i] = PyString_AsString(key);
    in[i] = (PyArrayObject*)value;
    i += 1;
   }
  }

  
 // Construct the return...
  PyObject * ret = PyDict_New();  
  npy_intp os[2] = {PyArray_SHAPE(points)[0], PyArray_SHAPE(offsets)[0]};
  
  PyArrayObject ** out = (PyArrayObject **)malloc(channels * sizeof(PyArrayObject*)); // All borrowed
  for (i=0; i<channels; i++)
  {
   out[i] = (PyArrayObject*)PyArray_SimpleNew(2, os, NPY_FLOAT32);
   PyDict_SetItemString(ret, keys[i], (PyObject*)out[i]);
   Py_DECREF(out[i]);
  }


 // Loop and calculate each offset in turn...
  float * temp = (float*)malloc(channels * sizeof(float));
  int point, offset;
  for (point=0; point<os[0]; point++)
  {
   for (offset=0; offset<os[1]; offset++)
   {
    // Calculate location to query...
     float sy = *(float*)PyArray_GETPTR2(points, point, 0) + *(float*)PyArray_GETPTR2(offsets, offset, 0);
     float sx = *(float*)PyArray_GETPTR2(points, point, 1) + *(float*)PyArray_GETPTR2(offsets, offset, 1);
     
    // B-spline interpolation...
     MultivariateSampleB(degree, sy, sx, shape, channels, in, temp);
     
    // Copy into output...
     for (i=0; i<channels; i++)
     {
      *(float*)PyArray_GETPTR2(out[i], point, offset) = temp[i]; 
     }
   }
  }


 // Clean up...
  free(out);
  free(temp);
  free(in);
  free(keys);

  
 // Do the return...
  return ret;
}



// For extracting features with rotation/scale...
static PyObject * Rotsets(PyObject * module, PyObject * args)
{
 // Handle the parameters...
  PyObject * image;
  PyArrayObject * points;
  PyArrayObject * rotations;
  PyArrayObject * offsets;
  int degree = 3;
  
  if (!PyArg_ParseTuple(args, "O!O!O!O!|i", &PyDict_Type, &image, &PyArray_Type, &points, &PyArray_Type, &rotations, &PyArray_Type, &offsets, &degree)) return NULL;
 
 
 // Validate...
  // image...
   Py_ssize_t pos = 0;
   PyObject * key;
   PyObject * value;
   
   int shape[2] = {-1,-1};
   PyArrayObject * in_mask = NULL;
   
   while (PyDict_Next(image, &pos, &key, &value))
   {
    if (PyString_Check(key)==0)
    {
     PyErr_SetString(PyExc_RuntimeError, "All dictionary keys when specifying an image must be strings.");
     return NULL; 
    }
   
    if (PyArray_Check(value)==0)
    {
     PyErr_SetString(PyExc_RuntimeError, "All dictionary values when specifying an image must be ndarray-s.");
     return NULL;  
    }
   
    PyArrayObject * data = (PyArrayObject*)value;
   
    if (PyArray_NDIM(data)!=2)
    {
     PyErr_SetString(PyExc_RuntimeError, "Image arrays must be 2D.");
     return NULL;   
    }
   
    if (shape[0]==-1)
    {
     shape[0] = PyArray_SHAPE(data)[0];
     shape[1] = PyArray_SHAPE(data)[1];
    }
    else
    {
     if ((shape[0]!=PyArray_SHAPE(data)[0]) || (shape[1]!=PyArray_SHAPE(data)[1]))
     {
      PyErr_SetString(PyExc_RuntimeError, "All image arrays must have the same shape.");
      return NULL; 
     }
    }
   
   int type = PyArray_TYPE(data);
   
   if (type!=NPY_FLOAT)
   {
    if ((type==NPY_UBYTE)&&(strcmp(PyString_AsString(key),"mask")==0))
    {
     // Its a mask - thats ok - no-op.
      in_mask = data;
    }
    else
    {
     PyErr_SetString(PyExc_RuntimeError, "All image channels must be float32, with the exception of a mask channel which must be uint8.");
     return NULL; 
    }
   }
  }
  
 // points...
  if (PyArray_NDIM(points)!=2)
  {
   PyErr_SetString(PyExc_RuntimeError, "Points array must be 2D.");
   return NULL;
  }
  
  if (PyArray_SHAPE(points)[1]!=2)
  {
   PyErr_SetString(PyExc_RuntimeError, "Points array must have two columns.");
   return NULL;
  }
  
  if (PyArray_TYPE(points)!=NPY_FLOAT)
  {
   PyErr_SetString(PyExc_RuntimeError, "Points array must be of type float32.");
   return NULL;
  }
  
 // rotations...
  if (PyArray_NDIM(rotations)!=2)
  {
   PyErr_SetString(PyExc_RuntimeError, "Rotations array must be 2D.");
   return NULL;
  }
  
  if (PyArray_SHAPE(rotations)[1]!=2)
  {
   PyErr_SetString(PyExc_RuntimeError, "Rotations array must have two columns.");
   return NULL;
  }
  
  if (PyArray_SHAPE(rotations)[0]!=PyArray_SHAPE(points)[0])
  {
   PyErr_SetString(PyExc_RuntimeError, "Rotations array must have same number of rows as points array.");
   return NULL;
  }
  
  if (PyArray_TYPE(rotations)!=NPY_FLOAT)
  {
   PyErr_SetString(PyExc_RuntimeError, "Rotations array must be of type float32.");
   return NULL;
  }
 
 // offsets...
  if (PyArray_NDIM(offsets)!=2)
  {
   PyErr_SetString(PyExc_RuntimeError, "Offsets array must be 2D.");
   return NULL;
  }
  
  if (PyArray_SHAPE(offsets)[1]!=2)
  {
   PyErr_SetString(PyExc_RuntimeError, "Offsets array must have two columns.");
   return NULL;
  }
  
  if (PyArray_TYPE(offsets)!=NPY_FLOAT)
  {
   PyErr_SetString(PyExc_RuntimeError, "Offsets array must be of type float32.");
   return NULL;
  }
 
 // degree...
  if ((degree<0) || (degree>5))
  {
   PyErr_SetString(PyExc_RuntimeError, "degree must be in 0 to 5, inclusive.");
   return NULL; 
  }
  
  
 // Run FillMasked, to simplify all of the below code...
  if (in_mask!=NULL)
  {
   FillMasked(image);
  }

  
 // Extract the image as PyArrayObject's...
  int channels = PyDict_Size(image);
  if (in_mask!=NULL) channels -= 1;
  const char ** keys = (const char**)malloc(channels * sizeof(char*)); // All borrowed
  PyArrayObject ** in = (PyArrayObject **)malloc(channels * sizeof(PyArrayObject*)); // All borrowed
  
  pos = 0;
  int i = 0;
  
  while (PyDict_Next(image, &pos, &key, &value))
  {
   if (PyArray_TYPE((PyArrayObject*)value)==NPY_FLOAT)
   {
    keys[i] = PyString_AsString(key);
    in[i] = (PyArrayObject*)value;
    i += 1;
   }
  }

  
 // Construct the return...
  PyObject * ret = PyDict_New();  
  npy_intp os[2] = {PyArray_SHAPE(points)[0], PyArray_SHAPE(offsets)[0]};
  
  PyArrayObject ** out = (PyArrayObject **)malloc(channels * sizeof(PyArrayObject*)); // All borrowed
  for (i=0; i<channels; i++)
  {
   out[i] = (PyArrayObject*)PyArray_SimpleNew(2, os, NPY_FLOAT32);
   PyDict_SetItemString(ret, keys[i], (PyObject*)out[i]);
   Py_DECREF(out[i]);
  }


 // Loop and calculate each offset in turn...
  float * temp = (float*)malloc(channels * sizeof(float));
  int point, offset;
  for (point=0; point<os[0]; point++)
  {
   for (offset=0; offset<os[1]; offset++)
   {
    // Calculate location to query...
     float sy = *(float*)PyArray_GETPTR2(points, point, 0);
     float sx = *(float*)PyArray_GETPTR2(points, point, 1);
     
     float ny = *(float*)PyArray_GETPTR2(rotations, point, 0);
     float nx = *(float*)PyArray_GETPTR2(rotations, point, 1);
     
     float oy = *(float*)PyArray_GETPTR2(offsets, offset, 0);
     float ox = *(float*)PyArray_GETPTR2(offsets, offset, 1);
     
     sy +=  nx * oy;
     sx += -ny * oy;
     sy +=  ny * ox;
     sx +=  nx * ox;
     
    // B-spline interpolation...
     MultivariateSampleB(degree, sy, sx, shape, channels, in, temp);
     
    // Copy into output...
     for (i=0; i<channels; i++)
     {
      *(float*)PyArray_GETPTR2(out[i], point, offset) = temp[i]; 
     }
   }
  }


 // Clean up...
  free(out);
  free(temp);
  free(in);
  free(keys);

  
 // Do the return...
  return ret;
}



// Module stuff...
static PyMethodDef transform_c_methods[] =
{
 {"fillmasked", (PyCFunction)FillMasked_py, METH_VARARGS, "Given a dictionary representing an image fills in all values outside the mask with the same colour as the closest valid pixel, measured with Manhatten distance. Primarily a method used internally by transform(...) to avoid the complexity of handling a mask, but exposed incase its useful elsewhere. A no-op if called on an image that has no mask. The image is a set of numpy arrays indexed by channel names, all 2D and with the same size, all float32 except for a mask which is uint8 where non-zero means valid."},
 {"transform", (PyCFunction)Transform, METH_VARARGS, "Given a dictionary representing an image returns a new dictionary of the image having been transformed by a provided homography. Note that you typically think of homographys as going from source to target - this expects the inverse. You should also provide the width and height of the output image, though they default to the same as the input image if not provided. Parameters are (hg - homography to apply; each pixel coordinate is multiplied by it to get the source coordinate, image - dictionary of channels, each a float32 2D numpy array of the same size, indexed [y,x]. Can also include a 'mask' channel, uint8, that is nonzero when a pixel is valid, optional height, optional width, optional degree of the polynomial, which can be 1-5, and defaults to 3 (cubic)). Return is a new image dictionary, which will always contain a 'mask' channel indicating which pixels are valid. Note that if there is a mask it will make changes to the original image, but only in the areas marked as invalid by the mask."},
 
 {"sample", (PyCFunction)Sample, METH_VARARGS, "Lets you sample a specified set of locations in an image. Takes parameters (image, locations, degree). image is a dictionary of 2D float32 arrays indexed [y,x], all the same size, to be sampled. Can also include a 'mask' of uint8 where nonzero means valid. locations is a list of coordinates in the image to evaluate, as a 2D float32 numpy array with y in column 0 and x in column 1. degree is the optional degree of the B-spline to use - defaults to 3 (cubic; must be 0-5).  It returns a dictionary of 1D float32 numpy arrays indexed [location] of all the evaluations, one per input image channel. Note that any coordinates that land outside the image will be evaluated using repetition of border pixels - no mask is generated. Also note that if there is a mask it will make changes to the original image, but only in the areas marked as invalid by the mask."},
 
 {"offsets", (PyCFunction)Offsets, METH_VARARGS, "Slightly strange - lets you sample a specified set of offsets around each point in an array of coordinates. For extracting the values required by features that need this kind of thing. Takes parameters (image, points, offsets, degree). image is a dictionary of 2D float32 arrays indexed [y,x], all the same size, to be sampled. Can also include a 'mask' of uint8 where nonzero means valid. points is a list of points in the image to evaluate, as another 2D float32 numpy array with y in column 0 and x in column 1. offsets is a 2D float32 numpy array, of offsets from an origin pixel, the y axis in column 0, the x axis in column 1. Note that you can get the order the wrong way around with the only consequence being the indexing order of the returned matrices. degree is the optional degree of the B-spline to use - defaults to 3 (cubic; must be 0-5). It then returns a dictionary of float32 numpy arrays indexed [point, offset] of all the relevant evaluations, one per input image channel. Note that any coordinates that land outside the image will be evaluated using repetition of border pixels - no mask is generated. Also note that if there is a mask it will make changes to the original image, but only in the areas marked as invalid by the mask."},
 {"rotsets", (PyCFunction)Rotsets, METH_VARARGS, "Same as Offsets, except it makes rather more sense, as each location also has an orientation (given as sin(angle), cos(angle) - direction of x-axis) which is applied to the offsets before evaluation. In other words, this is for extracting feature vectors from images that estimate a rotation before sampling. Takes parameters (image, points, rotations, offsets, degree). image is a dictionary of 2D float32 arrays indexed [y,x], all the same size, to be sampled. Can also include a 'mask' of uint8 where nonzero means valid. points is a list of points in the image to evaluate, as another 2D float32 numpy array with y in column 0 and x in column 1. rotations is a 2D float32 array, aligned with the points and giving ny in column 0 and nx in column 1. These should be the unit length direction of the x axis - you can think of it as ny=sin(angle), nx=cos(angle). Note that scaling these vectors will have the expected effect, so you can have per-point scales as well as per-point angles. offsets is a 2D float32 numpy array, of offsets from an origin pixel, the y axis in column 0, the x axis in column 1. degree is the optional degree of the B-spline to use - defaults to 3 (cubic; must be 0-5). It then returns a dictionary of float32 numpy arrays indexed [point, offset] of all the relevant evaluations, one per input image channel. Note that any coordinates that land outside the image will be evaluated using repetition of border pixels - no mask is generated. Also note that if there is a mask it will make changes to the original image, but only in the areas marked as invalid by the mask."},
  
 {NULL}
};



#ifndef PyMODINIT_FUNC
#define PyMODINIT_FUNC void
#endif

PyMODINIT_FUNC inittransform_c(void)
{
 Py_InitModule3("transform_c", transform_c_methods, "Provides code for transforming an image by applying an arbitrary homography. Plus some other stuff for sampling with B-Spline interpolation of the image.");
 
 import_array();
 PrepareBSpline();
}
