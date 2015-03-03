// Copyright 2013 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



#include "ms_c.h"

#include <string.h>



// Note to anyone reading this code - the atof function used throughout is not related to the normal function with that name - try not to get confused!
static PyTypeObject MeanShiftType;



void MeanShift_new(MeanShift * this)
{
 this->kernel = &Uniform;
 this->config = NULL; // We know this is valid for the Uniform kernel.
 this->name = NULL; // Only contains a value if it differs from the kernel name.
 this->spatial_type = &KDTreeType;
 this->balls_type = &BallsHashType;
 DataMatrix_init(&this->dm);
 this->weight = -1.0;
 this->norm = -1.0;
 this->spatial = NULL;
 this->balls = NULL;
 
 this->quality = 0.5;
 this->epsilon = 1e-3;
 this->iter_cap = 1024;
 this->spatial_param = 0.1;
 this->ident_dist = 0.0;
 this->merge_range = 0.5;
 this->merge_check_step = 4;
 
 this->rng_link = NULL;
 int i;
 for (i=0; i<4; i++) this->rng[i] = 0;
 
 this->fv_int = NULL;
 this->fv_ext = NULL;
}

void MeanShift_dealloc(MeanShift * this)
{
 Py_XDECREF(this->rng_link);
 DataMatrix_deinit(&this->dm);
 if (this->spatial!=NULL) Spatial_delete(this->spatial);
 if (this->balls!=NULL) Balls_delete(this->balls);
 this->kernel->config_release(this->config);
 
 free(this->fv_int);
 free(this->fv_ext);
}


static PyObject * MeanShift_new_py(PyTypeObject * type, PyObject * args, PyObject * kwds)
{
 // Allocate the object...
  MeanShift * self = (MeanShift*)type->tp_alloc(type, 0);

 // On success construct it...
  if (self!=NULL) MeanShift_new(self);

 // Return the new object...
  return (PyObject*)self;
}

static void MeanShift_dealloc_py(MeanShift * self)
{
 MeanShift_dealloc(self);
 self->ob_type->tp_free((PyObject*)self);
}



static PyObject * MeanShift_kernels_py(MeanShift * self, PyObject * args)
{
 // Create the return list...
  PyObject * ret = PyList_New(0);
  
 // Add each random variable type in turn...
  int i = 0;
  while (ListKernel[i]!=NULL)
  {
   PyObject * name = PyString_FromString(ListKernel[i]->name);
   PyList_Append(ret, name);
   Py_DECREF(name);
   
   ++i; 
  }
 
 // Return...
  return ret;
}


static PyObject * MeanShift_get_kernel_py(MeanShift * self, PyObject * args)
{
 if (self->name==NULL)
 {
  return Py_BuildValue("s", self->kernel->name);
 }
 else
 {
  Py_INCREF(self->name);
  return self->name; 
 }
}


static PyObject * MeanShift_get_range_py(MeanShift * self, PyObject * args)
{
 int dims = DataMatrix_features(&self->dm);
 float range = self->kernel->range(dims, self->config, self->quality);
 return Py_BuildValue("f", range);
}


static PyObject * MeanShift_set_kernel_py(MeanShift * self, PyObject * args)
{
 // Parse the parameters...
  char * kname;
  if (!PyArg_ParseTuple(args, "s", &kname)) return NULL;
 
 // Try and find the relevant kernel - if found assign it and return...
  int i = 0;
  while (ListKernel[i]!=NULL)
  {
   int klength = strlen(ListKernel[i]->name);
   if (strncmp(ListKernel[i]->name, kname, klength)==0)
   {
    int dims = DataMatrix_features(&self->dm);
    const char * error = ListKernel[i]->config_verify(dims, kname+klength, NULL);
    if (error!=NULL)
    {
     PyErr_SetString(PyExc_RuntimeError, error);
     return NULL;
    }
    
    self->kernel->config_release(self->config);
    
    self->kernel = ListKernel[i];
    self->config = self->kernel->config_new(dims, kname+klength);
    self->norm = -1.0;
    
    if (self->name!=NULL)
    {
     Py_DECREF(self->name);
     self->name = NULL;
    }
    
    if (self->kernel->configuration!=NULL)
    {
     self->name = Py_BuildValue("s", kname);
    }
    
    Py_INCREF(Py_None);
    return Py_None;
   }
   
   ++i; 
  }
  
 // Was not succesful - throw an error...
  PyErr_SetString(PyExc_RuntimeError, "unrecognised kernel type");
  return NULL; 
}


static PyObject * MeanShift_copy_kernel_py(MeanShift * self, PyObject * args)
{
 // Get the parameters - another mean shift object...
  MeanShift * other;
  if (!PyArg_ParseTuple(args, "O!", &MeanShiftType, &other)) return NULL;
  
 // Clean up current...
  self->kernel->config_release(self->config);
  
  if (self->name!=NULL)
  {
   Py_DECREF(self->name);
   self->name = NULL;
  }
  
 // Copy across...
  self->kernel = other->kernel;
  
  self->config = other->config;
  self->kernel->config_acquire(self->config);
  
  if (other->name!=NULL)
  {
   self->name = other->name;
   Py_INCREF(self->name);
  }
 
 // Return None...
  Py_INCREF(Py_None);
  return Py_None;
}


static PyObject * MeanShift_link_rng_py(MeanShift * self, PyObject * args)
{
 // Get the parameters - another mean shift object...
  MeanShift * other = NULL;
  if (!PyArg_ParseTuple(args, "|O!", &MeanShiftType, &other)) return NULL;

 // Unlink the current...
  Py_XDECREF(self->rng_link);
  self->rng_link = NULL;
  
 // If required link the other MeanShift object 
  if (other!=NULL)
  {
   while (other->rng_link!=NULL) other = other->rng_link;
   
   self->rng_link = other;
   Py_INCREF(self->rng_link);
  }
  
 // Return None...
  Py_INCREF(Py_None);
  return Py_None;
}



static PyObject * MeanShift_spatials_py(MeanShift * self, PyObject * args)
{
 // Create the return list...
  PyObject * ret = PyList_New(0);
  
 // Add each spatial type in turn...
  int i = 0;
  while (ListSpatial[i]!=NULL)
  {
   PyObject * name = PyString_FromString(ListSpatial[i]->name);
   PyList_Append(ret, name);
   Py_DECREF(name);
   
   ++i; 
  }
 
 // Return...
  return ret;
}


static PyObject * MeanShift_get_spatial_py(MeanShift * self, PyObject * args)
{
 return Py_BuildValue("s", self->spatial_type->name);
}


static PyObject * MeanShift_set_spatial_py(MeanShift * self, PyObject * args)
{
 // Parse the parameters...
  char * sname;
  if (!PyArg_ParseTuple(args, "s", &sname)) return NULL;
 
 // Try and find the relevant indexing method - if found assign it and return...
  int i = 0;
  while (ListSpatial[i]!=NULL)
  {
   if (strcmp(ListSpatial[i]->name, sname)==0)
   {
    self->spatial_type = ListSpatial[i];
    
    if (self->spatial!=NULL)
    {
     Spatial_delete(self->spatial);
     self->spatial = NULL; 
    }
    
    Py_INCREF(Py_None);
    return Py_None;
   }
   
   ++i; 
  }
  
 // Was not succesful - throw an error...
  PyErr_SetString(PyExc_RuntimeError, "unrecognised spatial type");
  return NULL; 
}



static PyObject * MeanShift_balls_py(MeanShift * self, PyObject * args)
{
 // Create the return list...
  PyObject * ret = PyList_New(0);
  
 // Add each balls type in turn...
  int i = 0;
  while (ListBalls[i]!=NULL)
  {
   PyObject * name = PyString_FromString(ListBalls[i]->name);
   PyList_Append(ret, name);
   Py_DECREF(name);
   
   ++i; 
  }
 
 // Return...
  return ret;
}


static PyObject * MeanShift_get_balls_py(MeanShift * self, PyObject * args)
{
 return Py_BuildValue("s", self->balls_type->name);
}


static PyObject * MeanShift_set_balls_py(MeanShift * self, PyObject * args)
{
 // Parse the parameters...
  char * bname;
  if (!PyArg_ParseTuple(args, "s", &bname)) return NULL;
 
 // Try and find the relevant techneque - if found assign it and return...
  int i = 0;
  while (ListBalls[i]!=NULL)
  {
   if (strcmp(ListBalls[i]->name, bname)==0)
   {
    self->balls_type = ListBalls[i];
   
    // Trash the cluster centers...
     if (self->balls!=NULL)
     {
      Balls_delete(self->balls);
      self->balls = NULL;
     }
  
    Py_INCREF(Py_None);
    return Py_None;
   }
   
   ++i; 
  }
  
 // Was not succesful - throw an error...
  PyErr_SetString(PyExc_RuntimeError, "unrecognised balls type");
  return NULL; 
}



static PyObject * MeanShift_info_py(MeanShift * self, PyObject * args)
{
 // Parse the parameters...
  char * name;
  if (!PyArg_ParseTuple(args, "s", &name)) return NULL;
  int i;
  
 // Try and find the relevant entity - if found assign it and return...
  i = 0;
  while (ListKernel[i]!=NULL)
  {
   if (strcmp(ListKernel[i]->name, name)==0)
   {
    return PyString_FromString(ListKernel[i]->description);
   }
   
   ++i; 
  }
  
  i = 0;
  while (ListSpatial[i]!=NULL)
  {
   if (strcmp(ListSpatial[i]->name, name)==0)
   {
    return PyString_FromString(ListSpatial[i]->description);
   }
   
   ++i; 
  }
  
  i = 0;
  while (ListBalls[i]!=NULL)
  {
   if (strcmp(ListBalls[i]->name, name)==0)
   {
    return PyString_FromString(ListBalls[i]->description);
   }
   
   ++i; 
  }
  
 // Was not succesful - throw an error...
  PyErr_SetString(PyExc_RuntimeError, "unrecognised entity name");
  return NULL; 
}


static PyObject * MeanShift_info_config_py(MeanShift * self, PyObject * args)
{
 // Parse the parameters...
  char * name;
  if (!PyArg_ParseTuple(args, "s", &name)) return NULL;
  int i;
  
 // Try and find the relevant entity - if found assign it and return...
  i = 0;
  while (ListKernel[i]!=NULL)
  {
   if (strcmp(ListKernel[i]->name, name)==0)
   {
    if (ListKernel[i]->configuration!=NULL)
    {
     return PyString_FromString(ListKernel[i]->configuration);
    }
    else
    {
     Py_INCREF(Py_None);
     return Py_None;
    }
   }
   
   ++i; 
  }

 // Was not succesful - throw an error...
  PyErr_SetString(PyExc_RuntimeError, "unrecognised kernel name");
  return NULL; 
}



static PyObject * MeanShift_copy_all_py(MeanShift * self, PyObject * args)
{
 // Get the parameters - another mean shift object...
  MeanShift * other;
  if (!PyArg_ParseTuple(args, "O!", &MeanShiftType, &other)) return NULL;
  
 // Clean up current stuff...
  self->kernel->config_release(self->config);
  
  if (self->name!=NULL)
  {
   Py_DECREF(self->name);
   self->name = NULL;
  }
  
  self->weight = -1.0;
  self->norm = -1.0;
  
  if (self->spatial!=NULL)
  {
   Spatial_delete(self->spatial);
   self->spatial = NULL;
  }
  
  if (self->balls!=NULL)
  {
   Balls_delete(self->balls); 
   self->balls = NULL;
  }
  
 // Copy across...
  self->kernel = other->kernel;
  
  self->config = other->config;
  self->kernel->config_acquire(self->config);
  
  if (other->name!=NULL)
  {
   self->name = other->name;
   Py_INCREF(self->name);
  }
  
  self->spatial_type = other->spatial_type;
  self->balls_type = other->balls_type;
  
  self->quality = other->quality;
  self->epsilon = other->epsilon;
  self->iter_cap = other->iter_cap;
  self->spatial_param = other->spatial_param;
  self->ident_dist = other->ident_dist;
  self->merge_range = other->merge_range;
  self->merge_check_step = other->merge_check_step;
  
 // Return None...
  Py_INCREF(Py_None);
  return Py_None;
}


static PyObject * MeanShift_reset_py(MeanShift * self, PyObject * args)
{
 // Trash everything dependent on the contents of the data matrix...
  self->weight = -1.0;
  self->norm = -1.0;
  
  if (self->spatial!=NULL)
  {
   Spatial_delete(self->spatial);
   self->spatial = NULL;
  }
  
  if (self->balls!=NULL)
  {
   Balls_delete(self->balls); 
   self->balls = NULL;
  }
  
 // Return None...
  Py_INCREF(Py_None);
  return Py_None; 
}



static PyObject * MeanShift_converters_py(MeanShift * self, PyObject * args)
{
 // Create the return list...
  PyObject * ret = PyList_New(0);
  
 // Add each balls type in turn...
  int i = 0;
  while (ListConvert[i]!=NULL)
  {
   PyObject * name = PyString_FromStringAndSize(&ListConvert[i]->code, 1);
   PyList_Append(ret, name);
   Py_DECREF(name);
   
   ++i; 
  }
 
 // Return...
  return ret;
}


static PyObject * MeanShift_converter_py(MeanShift * self, PyObject * args)
{
 // Parse the parameter...
  char * code;
  if (!PyArg_ParseTuple(args, "s", &code)) return NULL;

 // Search for the relevant convertor...
  int i = 0;
  while (ListConvert[i]!=NULL)
  {
   if (code[0]==ListConvert[i]->code)
   {
    // Create and return a dictionary with the relevant stuff in... 
     return Py_BuildValue("{ss#sssssisi}", "code", &ListConvert[i]->code, 1, "name", ListConvert[i]->name, "description", ListConvert[i]->description, "external", ListConvert[i]->dim_ext, "internal", ListConvert[i]->dim_int);
   }
    
   ++i; 
  }
  
 // Didn't find it - return none...
  Py_INCREF(Py_None);
  return Py_None; 
}



static PyObject * MeanShift_set_data_py(MeanShift * self, PyObject * args)
{
 // Extract the parameters...
  PyArrayObject * data;
  char * dim_types;
  PyObject * weight_index = NULL;
  char * conv_codes = NULL;
  if (!PyArg_ParseTuple(args, "O!s|Os", &PyArray_Type, &data, &dim_types, &weight_index, &conv_codes)) return NULL;
  
  if ((conv_codes!=NULL)&&(conv_codes[0]==0)) conv_codes = NULL;
  
 // Check its all ok, starting with the dimension string length...
  if (strlen(dim_types)!=PyArray_NDIM(data))
  {
   PyErr_SetString(PyExc_RuntimeError, "dimension type string must be the same length as the number of dimensions in the data matrix");
   return NULL;
  }
  
 // Verify the numpy array is a sane type...
  if ((PyArray_DESCR(data)->kind!='b')&&(PyArray_DESCR(data)->kind!='i')&&(PyArray_DESCR(data)->kind!='u')&&(PyArray_DESCR(data)->kind!='f'))
  {
   PyErr_SetString(PyExc_RuntimeError, "provided data matrix is not of a supported type");
   return NULL; 
  }
  
 // Verify the dimension types...
  int i;
  int features = 1;
  int duals = 0;
  
  for (i=0; i<PyArray_NDIM(data); i++)
  {
   if ((dim_types[i]!='d')&&(dim_types[i]!='f')&&(dim_types[i]!='b'))
   {
    PyErr_SetString(PyExc_RuntimeError, "dimension type string includes an unrecognised code");
    return NULL;
   }
   
   if (dim_types[i]=='b') duals += 1;
   else
   {
    if (dim_types[i]=='f')
    {
     features *= PyArray_DIMS(data)[i];
    }
   }
  }
  
  features += duals;
  
 // Handle the weight index...
  int weight_i = -1;
  if ((weight_index!=NULL)&&(weight_index!=Py_None))
  {
   if (PyInt_Check(weight_index)==0)
   {
    PyErr_SetString(PyExc_RuntimeError, "weight index must be an integer");
    return NULL;  
   }
    
   weight_i = PyInt_AsLong(weight_index);
   
   if ((weight_i>=0)&&(weight_i<features))
   {
    features -= 1;
   }
  }
  
 // Verify the conversion codes...
  if (conv_codes!=NULL)
  {
   int length = strlen(conv_codes);
   int external_length = 0;
   
   for (i=0; i<length; i++)
   {
    int j;
    for (j=0; ; j++)
    {
     if (ListConvert[j]==NULL)
     {
      // Unrecognised conversion code - escape.
       PyErr_SetString(PyExc_RuntimeError, "conversion type string includes an unrecognised code"); 
       return NULL;
     }
     
     if (ListConvert[j]->code==conv_codes[i])
     {
      external_length += ListConvert[j]->dim_ext;
      break; 
     }
    }
   }
   
   if (external_length!=features)
   {
    PyErr_SetString(PyExc_RuntimeError, "conversion codes total length does not match length of features as extracted from provided numpy array.");
    return NULL;
   }
  }
  
 // Make the assignment...
  DimType * dt = (DimType*)malloc(PyArray_NDIM(data) * sizeof(DimType));
  for (i=0; i<PyArray_NDIM(data); i++)
  {
   switch (dim_types[i])
   {
    case 'd': dt[i] = DIM_DATA;    break;
    case 'f': dt[i] = DIM_FEATURE; break;
    case 'b': dt[i] = DIM_DUAL;    break;
   }
  }
  
  DataMatrix_set(&self->dm, data, dt, weight_i, conv_codes);
  free(dt);
  
 // Setup the temporaries kept with the mean shift object...
  self->fv_int = (float*)realloc(self->fv_int, DataMatrix_features(&self->dm) * sizeof(float));
  self->fv_ext = (float*)realloc(self->fv_ext, DataMatrix_ext_features(&self->dm) * sizeof(float));
  
 // Trash the spatial...
  if (self->spatial!=NULL)
  {
   Spatial_delete(self->spatial);
   self->spatial = NULL; 
  }
  
 // Trash the cluster centers...
  if (self->balls!=NULL)
  {
   Balls_delete(self->balls);
   self->balls = NULL;
  }
  
 // Trash the weight record...
  self->weight = -1.0;
  self->norm = -1.0;
 
 // Return None...
  Py_INCREF(Py_None);
  return Py_None;
}


static PyObject * MeanShift_get_dm_py(MeanShift * self, PyObject * args)
{
 if (self->dm.array!=NULL) // Verify that there is a data matrix to in fact return!
 {
  Py_INCREF((PyObject*)self->dm.array);
  return (PyObject*)self->dm.array;
 }
 else
 {
  Py_INCREF(Py_None);
  return Py_None;
 }
}


static PyObject * MeanShift_get_dim_py(MeanShift * self, PyObject * args)
{
 PyObject * ret = PyString_FromStringAndSize(NULL, PyArray_NDIM(self->dm.array));
 char * out = PyString_AsString(ret);
 
 int i;
 for (i=0;i<PyArray_NDIM(self->dm.array);i++)
 {
  switch (self->dm.dt[i])
  {
   case DIM_DATA:    out[i] = 'd'; break;
   case DIM_FEATURE: out[i] = 'f'; break;
   case DIM_DUAL:    out[i] = 'b'; break;
   default: out[i] = 'e'; break; // Should never happen, obviously.
  }
 }
  
 return ret;
}


static PyObject * MeanShift_get_weight_dim_py(MeanShift * self, PyObject * args)
{
 if (self->dm.weight_index>=0)
 {
  return Py_BuildValue("f", self->dm.weight_index);
 }
 else
 {
  Py_INCREF(Py_None);
  return Py_None;
 }
}



static PyObject * MeanShift_fetch_dm_py(MeanShift * self, PyObject * args)
{
 // Handle there being no data by returning None...
  if (self->dm.array==NULL)
  {
   Py_INCREF(Py_None);
   return Py_None;
  }
 
 // Calculate the dimensions and create the matrix to return...
  npy_intp dim[2];
  dim[0] = DataMatrix_exemplars(&self->dm);
  dim[1] = DataMatrix_ext_features(&self->dm);
   
  PyArrayObject * ret = (PyArrayObject*)PyArray_SimpleNew(2, dim, NPY_FLOAT32);
  
 // Loop and fill in the feature vectors...
  int i;
  for (i=0; i<dim[0]; i++)
  {
   float * fv = DataMatrix_ext_fv(&self->dm, i, NULL);
   
   int j;
   for (j=0; j<dim[1]; j++)
   {
    *(float*)PyArray_GETPTR2(ret, i, j) = fv[j];
   }
  }
  
 // Return the new data matrix...
  return (PyObject*)ret;
}


static PyObject * MeanShift_fetch_weight_py(MeanShift * self, PyObject * args)
{
 if (self->dm.array==NULL)
 {
  Py_INCREF(Py_None);
  return Py_None;
 }
 
 // Create the matrix to return...
  npy_intp exemplars = DataMatrix_exemplars(&self->dm);
  PyArrayObject * ret = (PyArrayObject*)PyArray_SimpleNew(1, &exemplars, NPY_FLOAT32);
 
// Loop and fill in the weights...
  int i;
  for (i=0; i<exemplars; i++)
  {
   DataMatrix_ext_fv(&self->dm, i, (float*)PyArray_GETPTR1(ret, i)); 
  }
 
 // Return the new weight matrix...
  return (PyObject*)ret; 
}



static PyObject * MeanShift_set_scale_py(MeanShift * self, PyObject * args)
{
 // Extract the parameters...
  PyArrayObject * scale;
  float weight_scale = 1.0;
  if (!PyArg_ParseTuple(args, "O!|f", &PyArray_Type, &scale, &weight_scale)) return NULL;
 
 // Handle the scale...
  if ((PyArray_NDIM(scale)!=1)||(PyArray_DIMS(scale)[0]!=DataMatrix_features(&self->dm)))
  {
   PyErr_SetString(PyExc_RuntimeError, "scale vector must be a simple 1D numpy array with length matching the number of features after any conversion.");
   return NULL;
  }
  ToFloat atof = KindToFunc(PyArray_DESCR(scale));
  
  int i;
  for (i=0; i<PyArray_DIMS(scale)[0]; i++)
  {
   self->fv_int[i] = atof(PyArray_GETPTR1(scale, i));
  }
  
  DataMatrix_set_scale(&self->dm, self->fv_int, weight_scale);
  
 // Trash the spatial - changing either of the above invalidates it...
  if (self->spatial!=NULL)
  {
   Spatial_delete(self->spatial);
   self->spatial = NULL; 
  }
  
 // Trash the cluster centers...
  if (self->balls!=NULL)
  {
   Balls_delete(self->balls);
   self->balls = NULL;
  }
  
 // Trash the weight record...
  self->weight = -1.0;
  self->norm = -1.0;
 
 // Return None...
  Py_INCREF(Py_None);
  return Py_None;
}


static PyObject * MeanShift_get_scale_py(MeanShift * self, PyObject * args)
{
 npy_intp dim = DataMatrix_features(&self->dm);
 PyArrayObject * ret = (PyArrayObject*)PyArray_SimpleNew(1, &dim, NPY_FLOAT32);
 
 int i;
 for (i=0; i<dim; i++)
 {
  *(float*)PyArray_GETPTR1(ret, i) = self->dm.mult[i];
 }
 
 return (PyObject*)ret;
}


static PyObject * MeanShift_get_weight_scale_py(MeanShift * self, PyObject * args)
{
 return Py_BuildValue("f", self->dm.weight_scale);
}


static PyObject * MeanShift_copy_scale_py(MeanShift * self, PyObject * args)
{
 // Get the parameters - another mean shift object...
  MeanShift * other;
  if (!PyArg_ParseTuple(args, "O!", &MeanShiftType, &other)) return NULL;

 // Copy over the scale...  
  DataMatrix_set_scale(&self->dm, other->dm.mult, other->dm.weight_scale);
  
 // Trash the spatial - changing either of the above invalidates it...
  if (self->spatial!=NULL)
  {
   Spatial_delete(self->spatial);
   self->spatial = NULL; 
  }
  
 // Trash the cluster centers...
  if (self->balls!=NULL)
  {
   Balls_delete(self->balls);
   self->balls = NULL;
  }
  
 // Trash the weight record...
  self->weight = -1.0;
  self->norm = -1.0;

 // Return None...
  Py_INCREF(Py_None);
  return Py_None;
}



static PyObject * MeanShift_exemplars_py(MeanShift * self, PyObject * args)
{
 return Py_BuildValue("i", DataMatrix_exemplars(&self->dm));
}


static PyObject * MeanShift_features_py(MeanShift * self, PyObject * args)
{
 return Py_BuildValue("i", DataMatrix_ext_features(&self->dm));
}


static PyObject * MeanShift_features_internal_py(MeanShift * self, PyObject * args)
{
 return Py_BuildValue("i", DataMatrix_features(&self->dm));
}


float MeanShift_weight(MeanShift * this)
{
 if (this->weight<0.0)
 {
  this->weight = calc_weight(&this->dm);
 }
  
 return this->weight;
}


static PyObject * MeanShift_weight_py(MeanShift * self, PyObject * args)
{
 return Py_BuildValue("f", MeanShift_weight(self));
}


static PyObject * MeanShift_stats_py(MeanShift * self, PyObject * args)
{
 // Prep...
  int exemplars = DataMatrix_exemplars(&self->dm);
  npy_intp features = DataMatrix_features(&self->dm);
  
  PyArrayObject * mean = (PyArrayObject*)PyArray_SimpleNew(1, &features, NPY_FLOAT32);
  PyArrayObject * sd   = (PyArrayObject*)PyArray_SimpleNew(1, &features, NPY_FLOAT32);
  
  int i, j;
  for (j=0; j<features; j++)
  {
   *(float*)PyArray_GETPTR1(mean, j) = 0.0;
   *(float*)PyArray_GETPTR1(sd, j)   = 0.0;
  }
  
 // Single pass to calculate everything at once...
  float total = 0.0;  
  for (i=0; i<exemplars; i++)
  {
   float w;
   float * fv = DataMatrix_fv(&self->dm, i, &w);
   float new_total = total + w;
   
   for (j=0; j<features; j++)
   {
    float delta = fv[j] - *(float*)PyArray_GETPTR1(mean , j);
    float r = delta * w / new_total;
    *(float*)PyArray_GETPTR1(mean, j) += r;
    *(float*)PyArray_GETPTR1(sd, j) += total * delta * r;
   }
    
   total = new_total;
  }
    
 // Finish the sd and adapt the scale...
  if (total<1e-6) total = 1e-6; // Safety, for if they are all outliers.
  for (j=0; j<features; j++)
  {
   *(float*)PyArray_GETPTR1(mean, j) /= self->dm.mult[j];
   *(float*)PyArray_GETPTR1(sd, j) = sqrt(*(float*)PyArray_GETPTR1(sd, j) / total) / self->dm.mult[j];
  }
  
 // Construct and do the return...
  return Py_BuildValue("(N,N)", mean, sd);
}



static PyObject * MeanShift_scale_silverman_py(MeanShift * self, PyObject * args)
{
 int i, j;
 
 // Reset the scale to 1 before we start, so fv does something sensible...
  int exemplars = DataMatrix_exemplars(&self->dm);
  int features = DataMatrix_features(&self->dm);
  
  for (i=0; i<features; i++) self->dm.mult[i] = 1.0;
  
 // Use Silverman's rule of thumb to calculate scale values...
  float weight = 0.0;
  float * mean = (float*)malloc(features * sizeof(float));
  float * sd = (float*)malloc(features * sizeof(float));
  
  // Calculate and store the standard deviation of each dimension...
   for (i=0; i<features; i++)
   {
    mean[i] = 0.0;
    sd[i] = 0.0;
   }
   
   for (j=0; j<exemplars; j++)
   {
    float w;
    float * fv = DataMatrix_fv(&self->dm, j, &w);
    float temp = weight + w;
    
    for (i=0; i<features; i++)
    {
     float delta = fv[i] - mean[i];
     float r = delta * w / temp;
     mean[i] += r;
     sd[i] += weight * delta * r;
    }
    
    weight = temp;
   }
   
   for (i=0; i<features; i++)
   {
    sd[i] = sqrt(sd[i] / weight);
    if (sd[i]<1e-6) sd[i] = 1e-6;
   }
  
  // Convert standard deviations into a bandwidth via applying the rule...
   float mult = pow(weight * (features + 2.0) / 4.0, -1.0 / (features + 4.0));
   for (i=0; i<features; i++)
   {
    sd[i] = 1.0 / (sd[i] * mult);
   }
 
 // Set the scale...
  DataMatrix_set_scale(&self->dm, sd, self->dm.weight_scale);
  free(sd);
  free(mean);
 
 // Trash the spatial - changing the above invalidates it...
  if (self->spatial!=NULL)
  {
   Spatial_delete(self->spatial);
   self->spatial = NULL; 
  }
  
 // Trash the cluster centers...
  if (self->balls!=NULL)
  {
   Balls_delete(self->balls);
   self->balls = NULL;
  }
  
 // Trash the normalising constant...
  self->norm = -1.0;

 // Return None...
  Py_INCREF(Py_None);
  return Py_None;
}


static PyObject * MeanShift_scale_scott_py(MeanShift * self, PyObject * args)
{
 int i, j;
 
 // Reset the scale to 1 before we start, so fv does something sensible...
  int exemplars = DataMatrix_exemplars(&self->dm);
  int features = DataMatrix_features(&self->dm);
  
  for (i=0; i<features; i++) self->dm.mult[i] = 1.0;
 
 // Use Silverman's rule fo thumb to calculate scale values...   
  float weight = 0.0;
  float * mean = (float*)malloc(features * sizeof(float));
  float * sd = (float*)malloc(features * sizeof(float));
  
  // Calculate and store into s the standard deviation of each dimension...
   for (i=0; i<features; i++)
   {
    mean[i] = 0.0;
    sd[i] = 0.0;
   }
   
   for (j=0; j<exemplars; j++)
   {
    float w;
    float * fv = DataMatrix_fv(&self->dm, j, &w);
    float temp = weight + w;
    
    for (i=0; i<features; i++)
    {
     float delta = fv[i] - mean[i];
     float r = delta * w / temp;
     mean[i] += r;
     sd[i] += weight * delta * r;
    }
    
    weight = temp;
   }
   
   for (i=0; i<features; i++)
   {
    sd[i] = sqrt(sd[i] / weight);
    if (sd[i]<1e-6) sd[i] = 1e-6;
   }
  
  // Convert standard deviations into a bandwidth via applying the rule...
   float mult = pow(weight, -1.0 / (features + 4.0));
   for (i=0; i<features; i++)
   {
    sd[i] = 1.0 / (sd[i] * mult);
   }
 
 // Set the scale...
  DataMatrix_set_scale(&self->dm, sd, self->dm.weight_scale);
  free(sd);
  free(mean);
 
 // Trash the spatial - changing the above invalidates it...
  if (self->spatial!=NULL)
  {
   Spatial_delete(self->spatial);
   self->spatial = NULL; 
  }
  
 // Trash the cluster centers...
  if (self->balls!=NULL)
  {
   Balls_delete(self->balls);
   self->balls = NULL;
  }
  
 // Trash the normalising constant...
  self->norm = -1.0;
  
 // Return None...
  Py_INCREF(Py_None);
  return Py_None;
}


static PyObject * MeanShift_loo_nll_py(MeanShift * self, PyObject * args)
{
 // Extract the limits from the parameters...
  float limit = 1e-16;
  int sample_limit = -1;
  if (!PyArg_ParseTuple(args, "|fi", &limit, &sample_limit)) return NULL;
 
 // If spatial is null create it...
  if (self->spatial==NULL)
  {
   self->spatial = Spatial_new(self->spatial_type, &self->dm, self->spatial_param); 
  }
  
 // Calculate the normalising term if needed...
  if (self->norm<0.0)
  {
   self->norm = calc_norm(&self->dm, self->kernel, self->config, MeanShift_weight(self));
  }
  
 // Calculate the probability...
  float nll;
  if (sample_limit>0)
  {
   PhiloxRNG rng;
   PhiloxRNG_init(&rng, (self->rng_link!=NULL)?self->rng_link->rng:self->rng);
 
   nll = loo_nll(self->spatial, self->kernel, self->config, self->norm, self->quality, limit, sample_limit, &rng);
  }
  else
  {
   nll = loo_nll(self->spatial, self->kernel, self->config, self->norm, self->quality, limit, 0, NULL);
  }
 
 // Return it...
  return Py_BuildValue("f", nll);
}



static PyObject * MeanShift_entropy_py(MeanShift * self, PyObject * args)
{
 // Extract the limits from the parameters...
  int sample_limit = -1;
  if (!PyArg_ParseTuple(args, "|i", &sample_limit)) return NULL;
  
 // If spatial is null create it...
  if (self->spatial==NULL)
  {
   self->spatial = Spatial_new(self->spatial_type, &self->dm, self->spatial_param); 
  }
  
 // Calculate the normalising term if needed...
  if (self->norm<0.0)
  {
   self->norm = calc_norm(&self->dm, self->kernel, self->config, MeanShift_weight(self));
  }
  
 // Calculate and return...
  float ret;
  if (sample_limit>0)
  {
   PhiloxRNG rng;
   PhiloxRNG_init(&rng, (self->rng_link!=NULL)?self->rng_link->rng:self->rng);
 
   ret = entropy(self->spatial, self->kernel, self->config, self->norm, self->quality, sample_limit, &rng);
  }
  else
  {
   ret = entropy(self->spatial, self->kernel, self->config, self->norm, self->quality, 0, NULL);
  }
  
  return Py_BuildValue("f", ret);
}


static PyObject * MeanShift_kl_py(MeanShift * self, PyObject * args)
{
 // Get the parameters - another mean shift object and an optional limit...
  MeanShift * other;
  float limit = 1e-16;
  int sample_limit = -1;
  if (!PyArg_ParseTuple(args, "O!|fi", &MeanShiftType, &other, &limit, &sample_limit)) return NULL;
  
 // Make sure the required data structures for self are ready...
  if (self->spatial==NULL)
  {
   self->spatial = Spatial_new(self->spatial_type, &self->dm, self->spatial_param); 
  }
  
  if (self->norm<0.0)
  {
   self->norm = calc_norm(&self->dm, self->kernel, self->config, MeanShift_weight(self));
  }
  
 // Ditto for other...
  if (other->spatial==NULL)
  {
   other->spatial = Spatial_new(other->spatial_type, &other->dm, self->spatial_param); 
  }
  
  if (other->norm<0.0)
  {
   other->norm = calc_norm(&other->dm, other->kernel, other->config, MeanShift_weight(other));
  }
 
 // Calculate and return...
  float ret;
  if (sample_limit>0)
  {
   PhiloxRNG rng;
   PhiloxRNG_init(&rng, (self->rng_link!=NULL)?self->rng_link->rng:self->rng);
 
   ret = kl_divergence(self->spatial, self->kernel, self->config, self->norm, self->quality, other->spatial, other->kernel, other->config, other->norm, other->quality, limit, sample_limit, &rng);
  }
  else
  {
   ret = kl_divergence(self->spatial, self->kernel, self->config, self->norm, self->quality, other->spatial, other->kernel, other->config, other->norm, other->quality, limit, 0, NULL);
  }
  
  return Py_BuildValue("f", ret);
}



static PyObject * MeanShift_prob_py(MeanShift * self, PyObject * args)
{
 // Get the argument - a feature vector... 
  PyArrayObject * start;
  if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &start)) return NULL;
  
 // Check the input is acceptable...
  npy_intp feats = DataMatrix_ext_features(&self->dm);
  if ((PyArray_NDIM(start)!=1)||(PyArray_DIMS(start)[0]!=feats))
  {
   PyErr_SetString(PyExc_RuntimeError, "input vector must be 1D with the same length as the number of features.");
   return NULL;
  }
  ToFloat atof = KindToFunc(PyArray_DESCR(start));
  
 // If spatial is null create it...
  if (self->spatial==NULL)
  {
   self->spatial = Spatial_new(self->spatial_type, &self->dm, self->spatial_param); 
  }
  
 // Calculate the normalising term if needed...
  if (self->norm<0.0)
  {
   self->norm = calc_norm(&self->dm, self->kernel, self->config, MeanShift_weight(self));
  }
 
 // Create a temporary to hold the feature vector; handle conversion...
  int i;
  for (i=0; i<feats; i++)
  {
   self->fv_ext[i] = atof(PyArray_GETPTR1(start, i));
  }
  
  float * fv = DataMatrix_to_int(&self->dm, self->fv_ext, self->fv_int);
  
 // Calculate the probability...
  float p = prob(self->spatial, self->kernel, self->config, fv, self->norm, self->quality);
 
 // Return the calculated probability...
  return Py_BuildValue("f", p);
}



static PyObject * MeanShift_probs_py(MeanShift * self, PyObject * args)
{
 // Get the argument - a data matrix... 
  PyArrayObject * start;
  if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &start)) return NULL;

 // Check the input is acceptable...
  npy_intp feats = DataMatrix_ext_features(&self->dm);
  if ((PyArray_NDIM(start)!=2)||(PyArray_DIMS(start)[1]!=feats))
  {
   PyErr_SetString(PyExc_RuntimeError, "input matrix must be 2D with the same length as the number of features in the second dimension");
   return NULL;
  }
  ToFloat atof = KindToFunc(PyArray_DESCR(start));

 // If spatial is null create it...
  if (self->spatial==NULL)
  {
   self->spatial = Spatial_new(self->spatial_type, &self->dm, self->spatial_param); 
  }
  
 // Calculate the normalising term if needed...
  if (self->norm<0.0)
  {
   self->norm = calc_norm(&self->dm, self->kernel, self->config, MeanShift_weight(self));
  }
  
 // Create the output array... 
  PyArrayObject * out = (PyArrayObject*)PyArray_SimpleNew(1, PyArray_DIMS(start), NPY_FLOAT32);
  
  
 // Run the algorithm...
  int i;
  for (i=0; i<PyArray_DIMS(start)[0]; i++)
  {
   // Copy the feature vector into the temporary storage...
    int j;
    for (j=0; j<feats; j++)
    {
     self->fv_ext[j] = atof(PyArray_GETPTR2(start, i, j));
    }
    
   // Convert...
    float * fv = DataMatrix_to_int(&self->dm, self->fv_ext, self->fv_int);
   
   // Calculate the probability...
    float p = prob(self->spatial, self->kernel, self->config, fv, self->norm, self->quality);
   
   // Store it...
    *(float*)PyArray_GETPTR1(out, i) = p;
  }
 
 // Return the assigned clusters...
  return (PyObject*)out;
}



static PyObject * MeanShift_draw_py(MeanShift * self, PyObject * args)
{
 // Setup the rng...
  PhiloxRNG rng;
  PhiloxRNG_init(&rng, (self->rng_link!=NULL)?self->rng_link->rng:self->rng);
  
 // Create the return array...
  npy_intp feats = DataMatrix_ext_features(&self->dm);
  PyArrayObject * ret = (PyArrayObject*)PyArray_SimpleNew(1, &feats, NPY_FLOAT32);
  
 // Generate the return...
  draw(&self->dm, self->kernel, self->config, &rng, self->fv_int);
  
 // Convert from internal to external, stick in array...
  float * fv = DataMatrix_to_ext(&self->dm, self->fv_int, self->fv_ext);
  
  int i;
  for (i=0; i<feats; i++)
  {
   *(float*)PyArray_GETPTR1(ret, i) = fv[i];
  }
  
 // Return the draw...
  return (PyObject*)ret;
}



static PyObject * MeanShift_draws_py(MeanShift * self, PyObject * args)
{
 // Get the argument - how many to output... 
  npy_intp shape[2];
  if (!PyArg_ParseTuple(args, "n", &shape[0])) return NULL;
  
 // Setup the rng...
  PhiloxRNG rng;
  PhiloxRNG_init(&rng, (self->rng_link!=NULL)?self->rng_link->rng:self->rng);

 // Create the return array...
  shape[1] = DataMatrix_ext_features(&self->dm);
  PyArrayObject * ret = (PyArrayObject*)PyArray_SimpleNew(2, shape, NPY_FLOAT32);
  
 // Fill in the return matrix...
  int j;
  for (j=0; j<shape[0]; j++)
  {
   draw(&self->dm, self->kernel, self->config, &rng, self->fv_int);
   float * fv = DataMatrix_to_ext(&self->dm, self->fv_int, self->fv_ext);
   
   int i;
   for (i=0; i<shape[1]; i++)
   {
    *(float*)PyArray_GETPTR2(ret, j, i) = fv[i];
   }
  }
  
 // Return the draw...
  return (PyObject*)ret;
}



static PyObject * MeanShift_bootstrap_py(MeanShift * self, PyObject * args)
{
 // Get the arguments - how many to output... 
  npy_intp shape[2];
  if (!PyArg_ParseTuple(args, "n", &shape[0])) return NULL;
  
 // Create the return array...
  shape[1] = DataMatrix_features(&self->dm);
  PyArrayObject * ret = (PyArrayObject*)PyArray_SimpleNew(2, shape, NPY_FLOAT32);
 
 // Prepare the rng...
  PhiloxRNG rng;
  PhiloxRNG_init(&rng, (self->rng_link!=NULL)?self->rng_link->rng:self->rng);
  
 // Fill in the return matrix...
  int j;
  for (j=0; j<shape[0]; j++)
  {
   int ind = DataMatrix_draw(&self->dm, &rng);
   float * fv = DataMatrix_ext_fv(&self->dm, ind, NULL);
   
   int i;
   for (i=0; i<shape[1]; i++)
   {
    *(float*)PyArray_GETPTR2(ret, j, i) = fv[i];
   }
  }
  
 // Return the draw...
  return (PyObject*)ret;
}



static PyObject * MeanShift_mode_py(MeanShift * self, PyObject * args)
{
 // Get the argument - a feature vector... 
  PyArrayObject * start;
  if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &start)) return NULL;
  
 // Check the input is acceptable...
  npy_intp feats = DataMatrix_ext_features(&self->dm);
  if ((PyArray_NDIM(start)!=1)||(PyArray_DIMS(start)[0]!=feats))
  {
   PyErr_SetString(PyExc_RuntimeError, "input vector must be 1D with the same length as the number of features.");
   return NULL;
  }
  ToFloat atof = KindToFunc(PyArray_DESCR(start));
  
 // If spatial is null create it...
  if (self->spatial==NULL)
  {
   self->spatial = Spatial_new(self->spatial_type, &self->dm, self->spatial_param); 
  }

 // Create an output matrix...  
  PyArrayObject * ret = (PyArrayObject*)PyArray_SimpleNew(1, &feats, NPY_FLOAT32);
  
 // Fetch the data and covert into the internal format...
  int i;
  for (i=0; i<feats; i++)
  {
   self->fv_ext[i] = atof(PyArray_GETPTR1(start, i));
  }
  
  float * fv = DataMatrix_to_int(&self->dm, self->fv_ext, self->fv_int);
   
 // Run the algorithm; we need some temporary storage...
  int feats_int = DataMatrix_features(&self->dm);
  float * temp = (float*)malloc(feats_int * sizeof(float));
  
  mode(self->spatial, self->kernel, self->config, fv, temp, self->quality, self->epsilon, self->iter_cap);
  
  free(temp);
  
 // Convert back and write into the output...
  fv = DataMatrix_to_ext(&self->dm, fv, self->fv_ext);
  
  for (i=0; i<feats; i++)
  {
   *(float*)PyArray_GETPTR1(ret, i) = fv[i];
  }
 
 // Return the array...
  return (PyObject*)ret;
}



static PyObject * MeanShift_modes_py(MeanShift * self, PyObject * args)
{
 // Get the argument - a data matrix... 
  PyArrayObject * start;
  if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &start)) return NULL;

 // Check the input is acceptable...
  npy_intp dims[2];
  dims[0] = PyArray_DIMS(start)[0];
  dims[1] = DataMatrix_ext_features(&self->dm);
  
  if ((PyArray_NDIM(start)!=2)||(PyArray_DIMS(start)[1]!=dims[1]))
  {
   PyErr_SetString(PyExc_RuntimeError, "input matrix must be 2D with the same length as the number of features in the second dimension");
   return NULL;
  }
  ToFloat atof = KindToFunc(PyArray_DESCR(start));
  
 // If spatial is null create it...
  if (self->spatial==NULL)
  {
   self->spatial = Spatial_new(self->spatial_type, &self->dm, self->spatial_param); 
  }

 // Create an output matrix...  
  PyArrayObject * ret = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_FLOAT32);
  
 // Calculate each mode in turn, including conversion both ways...
  int feats_int = DataMatrix_features(&self->dm);
  float * temp = (float*)malloc(feats_int * sizeof(float));
  
  int j;
  for (j=0; j<dims[0]; j++)
  {
   // Extratc data and convert to internal format...
    int i;
    for (i=0; i<dims[1]; i++)
    {
     self->fv_ext[i] = atof(PyArray_GETPTR2(start, j, i));
    }
  
    float * fv = DataMatrix_to_int(&self->dm, self->fv_ext, self->fv_int);
  
   // Find the mode...
    mode(self->spatial, self->kernel, self->config, fv, temp, self->quality, self->epsilon, self->iter_cap);
   
   // Convert back and write out...
    fv = DataMatrix_to_ext(&self->dm, fv, self->fv_ext);
  
    for (i=0; i<dims[1]; i++)
    {
     *(float*)PyArray_GETPTR2(ret, j, i) = fv[i];
    }
  }
  
  free(temp); 
  
 // Return the matrix of modes...
  return (PyObject*)ret;
}



static PyObject * MeanShift_modes_data_py(MeanShift * self, PyObject * args)
{
 // If spatial is null create it...
  if (self->spatial==NULL)
  {
   self->spatial = Spatial_new(self->spatial_type, &self->dm, self->spatial_param); 
  }

 // Work out the output matrix size...
  int nd = 1;
  int i;
  for (i=0; i<PyArray_NDIM(self->dm.array); i++)
  {
   if (self->dm.dt[i]!=DIM_FEATURE) nd += 1;
  }
  
  npy_intp * dims = (npy_intp*)malloc(nd * sizeof(npy_intp));
  
  nd = 0;
  for (i=0; i<PyArray_NDIM(self->dm.array); i++)
  {
   if (self->dm.dt[i]!=DIM_FEATURE)
   {
    dims[nd] = PyArray_DIMS(self->dm.array)[i];
    nd += 1;
   }
  }
  
  dims[nd] = DataMatrix_ext_features(&self->dm);
  nd += 1;
 
 // Create the output matrix...
  PyArrayObject * ret = (PyArrayObject*)PyArray_SimpleNew(nd, dims, NPY_FLOAT32);
 
 // Iterate and do each entry in turn...
  int feats_int = DataMatrix_features(&self->dm);
  float * temp = (float*)malloc(feats_int * sizeof(float));
  
  float * out = (float*)PyArray_DATA(ret);
  int loc = 0;
  
  while (loc<DataMatrix_exemplars(&self->dm))
  {
   // Get the relevent feature vector...
    float * fv = DataMatrix_fv(&self->dm, loc, NULL);
    for (i=0; i<feats_int; i++) self->fv_int[i] = fv[i];
   
   // Converge mean shift...
    mode(self->spatial, self->kernel, self->config, self->fv_int, temp, self->quality, self->epsilon, self->iter_cap);
    
   // Undo conversion and copy into out...
    fv = DataMatrix_to_ext(&self->dm, self->fv_int, self->fv_ext);
  
    for (i=0; i<dims[nd-1]; i++)
    {
     out[i] = fv[i];
    }
    
   // Move to next position, detecting when we are done... 
    loc += 1;
    out += dims[nd-1];
  }
 
 // Clean up...
  free(temp);
  free(dims);
 
 // Return...
  return (PyObject*)ret;
}



static PyObject * MeanShift_cluster_py(MeanShift * self, PyObject * args)
{
 // If spatial is null create it...
  if (self->spatial==NULL)
  {
   self->spatial = Spatial_new(self->spatial_type, &self->dm, self->spatial_param); 
  }

 // Create the balls...
  int feats_int = DataMatrix_features(&self->dm);
  if (self->balls!=NULL) Balls_delete(self->balls);
  self->balls = Balls_new(self->balls_type, feats_int, self->merge_range);

 // Work out the output matrix size...
  int nd = 0;
  int i;
  for (i=0; i<PyArray_NDIM(self->dm.array); i++)
  {
   if (self->dm.dt[i]!=DIM_FEATURE) nd += 1;
  }
  
  if (nd<2) nd = 2; // So the array can be reused below
  npy_intp * dims = (npy_intp*)malloc(nd * sizeof(npy_intp));
  
  nd = 0;
  for (i=0; i<PyArray_NDIM(self->dm.array); i++)
  {
   if (self->dm.dt[i]!=DIM_FEATURE)
   {
    dims[nd] = PyArray_DIMS(self->dm.array)[i];
    nd += 1;
   }
  }
  
 // Create the output matrix...
  PyArrayObject * index = (PyArrayObject*)PyArray_SimpleNew(nd, dims, NPY_INT32);
 
 // Do the work...
  cluster(self->spatial, self->kernel, self->config, self->balls, (int*)PyArray_DATA(index), self->quality, self->epsilon, self->iter_cap, self->ident_dist, self->merge_range, self->merge_check_step);
 
 // Extract the modes, which happen to be the centers of the balls...
  dims[0] = Balls_count(self->balls);
  dims[1] = DataMatrix_ext_features(&self->dm);
  
  PyArrayObject * modes = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_FLOAT32);
  
  int j;
  for (j=0; j<dims[0]; j++)
  {
   // Fetch the location...
    const float * loc = Balls_pos(self->balls, j);
    for (i=0; i<feats_int; i++)
    {
     self->fv_int[i] = loc[i]; 
    }
    
   // Convert to external... 
    float * fv = DataMatrix_to_ext(&self->dm, self->fv_int, self->fv_ext);
   
   // Store...
    for (i=0; i<dims[1]; i++)
    {
     *(float*)PyArray_GETPTR2(modes, j, i) = fv[i];
    }
  }
 
 // Clean up...
  free(dims);
 
 // Return the tuple of (modes, assignment)...
  return Py_BuildValue("(N,N)", modes, index);
}



static PyObject * MeanShift_assign_cluster_py(MeanShift * self, PyObject * args)
{
 // Get the argument - a feature vector... 
  PyArrayObject * start;
  if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &start)) return NULL;
  
 // Check the input is acceptable...
  npy_intp feats_ext = DataMatrix_ext_features(&self->dm);
  if ((PyArray_NDIM(start)!=1)||(PyArray_DIMS(start)[0]!=feats_ext))
  {
   PyErr_SetString(PyExc_RuntimeError, "input vector must be 1D with the same length as the number of features.");
   return NULL;
  }
  ToFloat atof = KindToFunc(PyArray_DESCR(start));
  
 // Verify that cluster has been run...
  if (self->balls==NULL)
  {
   PyErr_SetString(PyExc_RuntimeError, "the cluster method must be run before the assign_cluster method.");
   return NULL;
  }
 
 // If spatial is null create it...
  if (self->spatial==NULL)
  {
   self->spatial = Spatial_new(self->spatial_type, &self->dm, self->spatial_param); 
  }

 // Fetch and convert the feature vector...
  int i;
  for (i=0; i<feats_ext; i++)
  {
   self->fv_ext[i] = atof(PyArray_GETPTR1(start, i));
  }
  
  float * fv = DataMatrix_to_int(&self->dm, self->fv_ext, self->fv_int);

 // Run the algorithm...
  float * temp = (float*)malloc(DataMatrix_features(&self->dm) * sizeof(float));
  
  int cluster = assign_cluster(self->spatial, self->kernel, self->config, self->balls, fv, temp, self->quality, self->epsilon, self->iter_cap, self->merge_check_step);
  
  free(temp);
 
 // Return the assigned cluster...
  return Py_BuildValue("i", cluster);
}



static PyObject * MeanShift_assign_clusters_py(MeanShift * self, PyObject * args)
{
 // Get the argument - a feature vector... 
  PyArrayObject * start;
  if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &start)) return NULL;
  
 // Check the input is acceptable...
  npy_intp feats_ext = DataMatrix_ext_features(&self->dm);
  if ((PyArray_NDIM(start)!=2)||(PyArray_DIMS(start)[1]!=feats_ext))
  {
   PyErr_SetString(PyExc_RuntimeError, "input vector must be 2D with the second dimension the same length as the number of features.");
   return NULL;
  }
  ToFloat atof = KindToFunc(PyArray_DESCR(start));
  
 // Verify that cluster has been run...
  if (self->balls==NULL)
  {
   PyErr_SetString(PyExc_RuntimeError, "the cluster method must be run before the assign_cluster method.");
   return NULL; 
  }
  
 // If spatial is null create it...
  if (self->spatial==NULL)
  {
   self->spatial = Spatial_new(self->spatial_type, &self->dm, self->spatial_param); 
  }
  
 // Create a temporary array of floats...
  float * temp = (float*)malloc(DataMatrix_features(&self->dm) * sizeof(float));

 // Create the output array... 
  PyArrayObject * cluster = (PyArrayObject*)PyArray_SimpleNew(1, PyArray_DIMS(start), NPY_INT32);
  
 // Run the algorithm...
  int j;
  for (j=0; j<PyArray_DIMS(start)[0]; j++)
  {
   // Fetch and convert the feature vector...
    int i;
    for (i=0; i<feats_ext; i++)
    {
     self->fv_ext[i] = atof(PyArray_GETPTR2(start, j, i));
    }
    
    float * fv = DataMatrix_to_int(&self->dm, self->fv_ext, self->fv_int);
   
   // Run it...
    int c = assign_cluster(self->spatial, self->kernel, self->config, self->balls, fv, temp, self->quality, self->epsilon, self->iter_cap, self->merge_check_step);
   
   // Store the result...
    *(int*)PyArray_GETPTR1(cluster, j) = c;
  }
  
 // Clean up...
  free(temp);
 
 // Return the assigned clusters...
  return (PyObject*)cluster;
}


static PyObject * MeanShift_cluster_on_py(MeanShift * self, PyObject * args)
{
 // Get the argument - a data matrix... 
  PyArrayObject * dm;
  if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &dm)) return NULL;

 // Check the input is acceptable...
  int feats_ext = DataMatrix_ext_features(&self->dm);
  if ((PyArray_NDIM(dm)!=2)||(PyArray_DIMS(dm)[1]!=feats_ext))
  {
   PyErr_SetString(PyExc_RuntimeError, "input matrix must be 2D with the same length as the number of features in the second dimension");
   return NULL;
  }
  ToFloat atof = KindToFunc(PyArray_DESCR(dm));
  
 // If spatial is null create it...
  if (self->spatial==NULL)
  {
   self->spatial = Spatial_new(self->spatial_type, &self->dm, self->spatial_param); 
  }
  
 // We need a balls object to detect clustering...
  Balls balls = Balls_new(self->balls_type, self->dm.feats, self->merge_range);
  
 // Create the output matrix of cluster indices...
  npy_intp exemplars = PyArray_DIMS(dm)[0];
  PyArrayObject * index = (PyArrayObject*)PyArray_SimpleNew(1, &exemplars, NPY_INT32);
  
 // Go through and converge each exemplar in turn...
  int feats_int = DataMatrix_features(&self->dm);
  float * temp = (float*)malloc(feats_int * sizeof(float));
  
  int j, i;
  for (j=0; j<exemplars; j++)
  {
   // Obtain and convert the feature vector...
    for (i=0; i<feats_ext; i++)
    {
     self->fv_ext[i] = atof(PyArray_GETPTR2(dm, j, i));
    }
    
    float * fv = DataMatrix_to_int(&self->dm, self->fv_ext, self->fv_int);
    
   // Process...
    *(int*)PyArray_GETPTR1(index, j) = mode_merge(self->spatial, self->kernel, self->config, balls, fv, temp, self->quality, self->epsilon, self->iter_cap, self->merge_range, self->merge_check_step);
  }
  
 // Extract the modes, which are the centers of the balls...
  npy_intp dims[2];
  dims[0] = Balls_count(balls);
  dims[1] = feats_ext;
  
  PyArrayObject * modes = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_FLOAT32);
  
  for (j=0; j<dims[0]; j++)
  {
   // Fetch the location...
    const float * loc = Balls_pos(balls, j);
    for (i=0; i<feats_int; i++)
    {
     self->fv_int[i] = loc[i]; 
    }
    
   // Convert to external... 
    float * fv = DataMatrix_to_ext(&self->dm, self->fv_int, self->fv_ext);
   
   // Store...
    for (i=0; i<dims[1]; i++)
    {
     *(float*)PyArray_GETPTR2(modes, j, i) = fv[i];
    }
  }
  
 // Clean up...
  free(temp);
  Balls_delete(balls);
  
 // Return the tuple of (modes, assignment)...
  return Py_BuildValue("(N,N)", modes, index);
}



static PyObject * MeanShift_manifold_py(MeanShift * self, PyObject * args)
{
 // Get the argument - a feature vector and degrees of freedom for the manifold... 
  PyArrayObject * start;
  int degrees;
  PyObject * always_hessian = Py_True;
  if (!PyArg_ParseTuple(args, "O!i|O", &PyArray_Type, &start, &degrees, &always_hessian)) return NULL;
  
  if (PyBool_Check(always_hessian)==0)
  {
   PyErr_SetString(PyExc_RuntimeError, "Parameter indicating if to calculate the hessian for every step or not should be boolean");
   return NULL;  
  }

 // Check the input is acceptable...
  npy_intp feats_ext = DataMatrix_ext_features(&self->dm);
  if ((PyArray_NDIM(start)!=1)||(PyArray_DIMS(start)[0]!=feats_ext))
  {
   PyErr_SetString(PyExc_RuntimeError, "input vector must be 1D with the same length as the number of features.");
   return NULL;
  }
  ToFloat atof = KindToFunc(PyArray_DESCR(start));
  
 // If spatial is null create it...
  if (self->spatial==NULL)
  {
   self->spatial = Spatial_new(self->spatial_type, &self->dm, self->spatial_param); 
  }

 // Create an output matrix, copy in the data, applying the scale change...  
  PyArrayObject * ret = (PyArrayObject*)PyArray_SimpleNew(1, &feats_ext, NPY_FLOAT32);
  
 // Obtain the data, convert to internal...
  int i;
  for (i=0; i<feats_ext; i++)
  {
   self->fv_ext[i] = atof(PyArray_GETPTR1(start, i));
  }
  
  float * fv = DataMatrix_to_int(&self->dm, self->fv_ext, self->fv_int);
 
 // Run the algorithm; we need some temporary storage...
  int feats_int = DataMatrix_features(&self->dm);
  
  float * grad = (float*)malloc(feats_int * sizeof(float));
  float * hess = (float*)malloc(feats_int * feats_int * sizeof(float));
  float * eigen_vec = (float*)malloc(feats_int * feats_int * sizeof(float));
  float * eigen_val = (float*)malloc(feats_int * sizeof(float));
  
  manifold(self->spatial, degrees, fv, grad, hess, eigen_val, eigen_vec, self->quality, self->epsilon, self->iter_cap, (always_hessian==Py_False) ? 0 : 1);
  
  free(eigen_val);
  free(eigen_vec);
  free(hess);
  free(grad);
  
 // Convert back and store into output...
  fv = DataMatrix_to_ext(&self->dm, fv, self->fv_ext);
  
  for (i=0; i<feats_ext; i++)
  {
   *(float*)PyArray_GETPTR1(ret, i) = fv[i];
  }
 
 // Return the array...
  return (PyObject*)ret;
}



static PyObject * MeanShift_manifolds_py(MeanShift * self, PyObject * args)
{
 // Get the argument - a data matrix and degrees of freedom for the manifold... 
  PyArrayObject * start;
  int degrees;
  PyObject * always_hessian = Py_True;
  if (!PyArg_ParseTuple(args, "O!i|O", &PyArray_Type, &start, &degrees, &always_hessian)) return NULL;

  if (PyBool_Check(always_hessian)==0)
  {
   PyErr_SetString(PyExc_RuntimeError, "Parameter indicating if to calculate the hessian for every step or not should be boolean");
   return NULL;  
  }
  
 // Check the input is acceptable...
  npy_intp dims[2];
  dims[0] = PyArray_DIMS(start)[0];
  dims[1] = DataMatrix_ext_features(&self->dm);
  
  if ((PyArray_NDIM(start)!=2)||(PyArray_DIMS(start)[1]!=dims[1]))
  {
   PyErr_SetString(PyExc_RuntimeError, "input matrix must be 2D with the same length as the number of features in the second dimension");
   return NULL;
  }
  ToFloat atof = KindToFunc(PyArray_DESCR(start));
  
 // If spatial is null create it...
  if (self->spatial==NULL)
  {
   self->spatial = Spatial_new(self->spatial_type, &self->dm, self->spatial_param); 
  }

 // Create an output matrix...  
  PyArrayObject * ret = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_FLOAT32);
  
 // Calculate each mode in turn, including undo any scale changes...
  int feats_int = DataMatrix_features(&self->dm);
  
  float * grad = (float*)malloc(feats_int * sizeof(float));
  float * hess = (float*)malloc(feats_int * feats_int * sizeof(float));
  float * eigen_vec = (float*)malloc(feats_int * feats_int * sizeof(float));
  float * eigen_val = (float*)malloc(feats_int * sizeof(float));
  
  int j, i;
  for (j=0; j<dims[0]; j++)
  {
   // Obtain and convert the feature vector...
    for (i=0; i<dims[1]; i++)
    {
     self->fv_ext[i] = atof(PyArray_GETPTR2(start, j, i));
    }
    
    float * fv =  DataMatrix_to_int(&self->dm, self->fv_ext, self->fv_int);
   
   // Find convergance point...
    manifold(self->spatial, degrees, fv, grad, hess, eigen_val, eigen_vec, self->quality, self->epsilon, self->iter_cap, (always_hessian==Py_False) ? 0 : 1);
    
   // Convert back and store...
    fv = DataMatrix_to_ext(&self->dm, fv, self->fv_ext);
   
    for (i=0; i<dims[1]; i++)
    {
     *(float*)PyArray_GETPTR2(ret, j, i) = fv[i]; 
    }
  }
  
  free(eigen_val);
  free(eigen_vec);
  free(hess);
  free(grad);
  
 // Return the matrix of modes...
  return (PyObject*)ret;
}



static PyObject * MeanShift_manifolds_data_py(MeanShift * self, PyObject * args)
{
 // Get the argument - a data matrix... 
  int degrees;
  PyObject * always_hessian = Py_True;
  if (!PyArg_ParseTuple(args, "i|O", &degrees, &always_hessian)) return NULL;
 
  if (PyBool_Check(always_hessian)==0)
  {
   PyErr_SetString(PyExc_RuntimeError, "Parameter indicating if to calculate the hessian for every step or not should be boolean");
   return NULL;  
  }
  
 // If spatial is null create it...
  if (self->spatial==NULL)
  {
   self->spatial = Spatial_new(self->spatial_type, &self->dm, self->spatial_param); 
  }

 // Work out the output matrix size...
  int nd = 1;
  int i;
  for (i=0; i<PyArray_NDIM(self->dm.array); i++)
  {
   if (self->dm.dt[i]!=DIM_FEATURE) nd += 1;
  }
  
  npy_intp * dims = (npy_intp*)malloc(nd * sizeof(npy_intp));
  
  nd = 0;
  for (i=0; i<PyArray_NDIM(self->dm.array); i++)
  {
   if (self->dm.dt[i]!=DIM_FEATURE)
   {
    dims[nd] = PyArray_DIMS(self->dm.array)[i];
    nd += 1;
   }
  }
  
  dims[nd] = DataMatrix_ext_features(&self->dm);
  nd += 1;
 
 // Create the output matrix...
  PyArrayObject * ret = (PyArrayObject*)PyArray_SimpleNew(nd, dims, NPY_FLOAT32);
 
 // Iterate and do each entry in turn...
  int feats_int = DataMatrix_features(&self->dm);
  
  float * grad = (float*)malloc(feats_int * sizeof(float));
  float * hess = (float*)malloc(feats_int * feats_int * sizeof(float));
  float * eigen_vec = (float*)malloc(feats_int * feats_int * sizeof(float));
  float * eigen_val = (float*)malloc(feats_int * sizeof(float));
  
  float * out = (float*)PyArray_DATA(ret);
  int loc = 0;
  while (loc<DataMatrix_exemplars(&self->dm))
  {
   // Obtain the relevant feature vector in internal space...
    float * fv = DataMatrix_fv(&self->dm, loc, NULL);
    for (i=0; i<feats_int; i++) self->fv_int[i] = fv[i];
    
   // Converge mean shift...
    manifold(self->spatial, degrees, self->fv_int, grad, hess, eigen_val, eigen_vec, self->quality, self->epsilon, self->iter_cap, (always_hessian==Py_False) ? 0 : 1);
    
   // Convert to external space and store in output...
    fv = DataMatrix_to_ext(&self->dm, self->fv_int, self->fv_ext);
    for (i=0; i<dims[nd-1]; i++) out[i] = fv[i];

   // Move to next position, detecting when we are done... 
    loc += 1;
    out += dims[nd-1];
  }
 
 // Clean up...
  free(eigen_val);
  free(eigen_vec);
  free(hess);
  free(grad);
  free(dims);
 
 // Return...
  return (PyObject*)ret;
}



static PyObject * MeanShift_mult_py(MeanShift * self, PyObject * args, PyObject * kw)
{
 // Handle the parameters...
  PyObject * multiplicands;
  PyArrayObject * output;
  
  int gibbs = 16;
  int mci = 64;
  int mh = 8;
  int fake = 0;
  
  static char * kw_list[] = {"multiplicands", "output", "gibbs", "mci", "mh", "fake", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OO!|iiii", kw_list, &multiplicands, &PyArray_Type, &output, &gibbs, &mci, &mh, &fake)) return NULL;
  
 // Verify the parameters are all good...
  if (PySequence_Check(multiplicands)==0)
  {
   PyErr_SetString(PyExc_RuntimeError, "First argument does not impliment the sequence protocol, e.g. its not a list, tuple, or equivalent.");
   return NULL; 
  }
  
  int terms = PySequence_Size(multiplicands);
  if (terms<1)
  {
   PyErr_SetString(PyExc_RuntimeError, "Need some MeanShift objects to multiply - identity is not defined.");
   return NULL;
  }
  
  if (PyObject_IsInstance(PySequence_GetItem(multiplicands, 0), (PyObject*)&MeanShiftType)!=1)
  {
   PyErr_SetString(PyExc_RuntimeError, "First item in multiplicand list is not a MeanShift object");
   return NULL; 
  }
  
  self = (MeanShift*)PySequence_GetItem(multiplicands, 0); // Bit weird, but why not? - self is avaliable and free to dance!
  int feats_ext = DataMatrix_ext_features(&self->dm);
  int feats_int = DataMatrix_features(&self->dm);
  
  int longest = DataMatrix_exemplars(&self->dm);
  if (longest==0)
  {
   PyErr_SetString(PyExc_RuntimeError, "First item in multiplicand list has no exemplars in its KDE");
   return NULL;
  }
  
  int i;
  for (i=1; i<terms; i++)
  {
   PyObject * temp = PySequence_GetItem(multiplicands, i);
   if (PyObject_IsInstance(temp, (PyObject*)&MeanShiftType)!=1)
   {
    PyErr_SetString(PyExc_RuntimeError, "Multiplicand list contains an entity that is not a MeanShift object");
    return NULL;
   }
   
   MeanShift * targ = (MeanShift*)temp;
   if (DataMatrix_features(&targ->dm)!=feats_int)
   {
    PyErr_SetString(PyExc_RuntimeError, "All the input KDEs must have the same number of internal features (dimensions)");
    return NULL;
   }
   
   int length = DataMatrix_exemplars(&targ->dm);
   if (length==0)
   {
    PyErr_SetString(PyExc_RuntimeError, "Item in multiplicand list has no exemplars in its KDE");
    return NULL;
   }
   
   if (length>longest) longest = length;
  }
  
  if (PyArray_NDIM(output)!=2)
  {
   PyErr_SetString(PyExc_RuntimeError, "Output array must have two dimensions");
   return NULL;
  }
  
  if (PyArray_DIMS(output)[1]!=feats_ext)
  {
   PyErr_SetString(PyExc_RuntimeError, "Output array must have the same number of columns as the input KDEs have features");
   return NULL; 
  }
  
  if (PyArray_DESCR(output)->kind!='f')
  {
   PyErr_SetString(PyExc_RuntimeError, "Output array must be a floating point type");
   return NULL;  
  }
  
  if ((PyArray_DESCR(output)->elsize!=4)&&(PyArray_DESCR(output)->elsize!=8))
  {
   PyErr_SetString(PyExc_RuntimeError, "Output array must be of type float or double");
   return NULL;  
  }
  
  if (gibbs<1)
  {
   PyErr_SetString(PyExc_RuntimeError, "gibbs sampling count must be positive");
   return NULL;
  }
  
  if (mci<1)
  {
   PyErr_SetString(PyExc_RuntimeError, "monte carlo integration sampling count must be positive");
   return NULL; 
  }
  
  if (mh<1)
  {
   PyErr_SetString(PyExc_RuntimeError, "Metropolis Hastings proposal count must be positive");
   return NULL;
  }
  
  if ((fake<0)||(fake>2))
  {
   PyErr_SetString(PyExc_RuntimeError, "fake parameter must be 0, 1 or 2");
   return NULL;
  }
  
 // Prepare the rng...
  PhiloxRNG rng;
  PhiloxRNG_init(&rng, (self->rng_link!=NULL)?self->rng_link->rng:self->rng);
  
 // Check for the degenerate situation of only one multiplicand, in which case we can just draw from it to generate the output...
  int j;
  if (terms==1)
  {
   char cd = PyArray_DESCR(output)->elsize!=4;
   
   for (j=0;j<PyArray_DIMS(output)[0]; j++)
   {
    // Code depends on status of fake...
     float * fv;
     if (fake==0)
     {
      // Correct approach - a real draw...
       draw(&self->dm, self->kernel, self->config, &rng, self->fv_int);
      
      // Convert to external format...
       fv = DataMatrix_to_ext(&self->dm, self->fv_int, self->fv_ext);
     }
     else
     {
      // Do a bootstrap draw...
       int ind = DataMatrix_draw(&self->dm, &rng);
       fv = DataMatrix_ext_fv(&self->dm, ind, NULL);
     }
     
    // Store...
     if (cd)
     {
      for (i=0; i<feats_ext; i++)
      {
       *(double*)PyArray_GETPTR2(output, j, i) = fv[i];
      }
     }
     else
     {
      for (i=0; i<feats_ext; i++)
      {
       *(float*)PyArray_GETPTR2(output, j, i) = fv[i];
      }
     }
   }
   
   Py_INCREF(Py_None);
   return Py_None;
  }
    
 // Create the MultCache, fill in parameters from the args...
  MultCache mc;
  MultCache_new(&mc);
  
  mc.rng = &rng;
  
  mc.gibbs_samples = gibbs;
  mc.mci_samples = mci;
  mc.mh_proposals = mh;
  
 // Make sure all the MeanShift objects have a Spatial initialised; create the list of Spatials; also get the kernel configuration list setup at the same time...
  Spatial * sl = (Spatial)malloc(terms * sizeof(Spatial));
  
  KernelConfig * config = NULL;
  if (self->config!=NULL)
  {
   config = (KernelConfig*)malloc(terms * sizeof(KernelConfig));
  }
  
  for (i=0; i<terms; i++)
  {
   MeanShift * targ = (MeanShift*)PySequence_GetItem(multiplicands, i);
   
   if (targ->spatial==NULL)
   {
    targ->spatial = Spatial_new(targ->spatial_type, &targ->dm, self->spatial_param);
   }
    
   sl[i] = targ->spatial;
   if (config!=NULL) config[i] = targ->config;
  }
 
 // Call the multiplication method for each draw and let it do the work...
  int * temp1 = (int*)malloc(longest * sizeof(int));
  float * temp2 = (float*)malloc(longest * sizeof(float));
  
  char cd = PyArray_DESCR(output)->elsize!=4;
  
  for (j=0; j<PyArray_DIMS(output)[0]; j++)
  {
   // Do multiplication...
    mult(self->kernel, config, terms, sl, self->fv_int, &mc, temp1, temp2, self->quality, fake);
    
   // Multiply it back to self's space...
    for (i=0; i<feats_int; i++)
    {
     self->fv_int[i] *= self->dm.mult[i]; 
    }
    
   // Convert to external format (inefficient given above!)...
    float * fv = DataMatrix_to_ext(&self->dm, self->fv_int, self->fv_ext);
   
   // Copy output to actual output array...
    if (cd)
    {
     for (i=0; i<feats_ext; i++)
     {
      *(double*)PyArray_GETPTR2(output, j, i) = fv[i];
     }
    }
    else
    {
     for (i=0; i<feats_ext; i++)
     {
      *(float*)PyArray_GETPTR2(output, j, i) = fv[i];
     }
    }
  }
 
 // Clean up the MultCache object and other stuff...
  free(temp2);
  free(temp1);
  free(config);
  free(sl);
  MultCache_delete(&mc);
 
 // Return None...
  Py_INCREF(Py_None);
  return Py_None;
}



static PyObject * MeanShift_sizeof_py(MeanShift * self, PyObject * args)
{
 size_t mem = sizeof(MeanShift) - sizeof(DataMatrix); // The DataMatrix has its own size method.
 
 int dims = DataMatrix_features(&self->dm);
 int ref_count;
 size_t shared_mem = self->kernel->byte_size(dims, self->config, &ref_count);
 if (self->name!=NULL) shared_mem += PyString_Size(self->name); // Wrong, but can't figure out the right way!
 mem += (size_t)ceil(shared_mem / (float)ref_count);
 
 mem += DataMatrix_byte_size(&self->dm);
 if (self->dm.array!=NULL) mem += PyArray_NBYTES(self->dm.array);
 
 if (self->spatial!=NULL) mem += Spatial_byte_size(self->spatial);
 if (self->balls!=NULL) mem += Balls_byte_size(self->balls);
 
 if (self->fv_int!=NULL) mem += dims * sizeof(float);
 if (self->fv_ext!=NULL) mem += DataMatrix_ext_features(&self->dm) * sizeof(float);
 
 return Py_BuildValue("n", mem);
}


static PyObject * MeanShift_memory_py(MeanShift * self, PyObject * args)
{
 int dims = DataMatrix_features(&self->dm);
 
 size_t self_mem = sizeof(MeanShift) - sizeof(DataMatrix);
 if (self->fv_int!=NULL) self_mem += dims * sizeof(float);
 if (self->fv_ext!=NULL) self_mem += DataMatrix_ext_features(&self->dm) * sizeof(float);
 
 int ref_count;
 size_t kernel_mem = self->kernel->byte_size(dims, self->config, &ref_count);
 if (self->name!=NULL) kernel_mem += PyString_Size(self->name); // Wrong, but can't figure out the right way!
 
 size_t dm_mem = DataMatrix_byte_size(&self->dm);
 
 size_t data_mem = 0;
 if (self->dm.array!=NULL) data_mem += PyArray_NBYTES(self->dm.array);
 
 size_t spatial_mem = 0;
 if (self->spatial!=NULL) spatial_mem += Spatial_byte_size(self->spatial);
 
 size_t balls_mem = 0;
 if (self->balls!=NULL) balls_mem += Balls_byte_size(self->balls);
 
 size_t total_mem = self_mem + (size_t)ceil(kernel_mem / (float)ref_count) + dm_mem + data_mem + spatial_mem + balls_mem;
  
 return Py_BuildValue("{snsnsisnsnsnsnsn}", "data", data_mem, "kernel", kernel_mem, "kernel_ref_count", ref_count, "dm", dm_mem, "spatial", spatial_mem, "balls", balls_mem, "self", self_mem, "total", total_mem);
}



static Py_ssize_t MeanShift_Length(MeanShift * self)
{
 return DataMatrix_exemplars(&self->dm);
}

static PyObject * MeanShift_GetItem(MeanShift * self, Py_ssize_t i)
{
 // Safety tests...
  Py_ssize_t exemplars = DataMatrix_exemplars(&self->dm);
 
  if (i<0) i += exemplars;
  if ((i<0)||(i>=exemplars))
  {
   PyErr_SetString(PyExc_IndexError, "MeanShift index out of range.");
   return NULL; 
  }

 // Get the actual data...
  float * fv = DataMatrix_ext_fv(&self->dm, i, NULL);
 
 // Convert into a numpy array...
  npy_intp feats = DataMatrix_ext_features(&self->dm);
  PyArrayObject * ret = (PyArrayObject*)PyArray_SimpleNew(1, &feats, NPY_FLOAT32);
  
  npy_intp j;
  for (j=0; j<feats; j++)
  {
   *(float*)PyArray_GETPTR1(ret, j) = fv[j];
  }
 
 // Return the shiny new numpy array...
  return (PyObject*)ret;
}


static PySequenceMethods MeanShift_as_sequence =
{
 (lenfunc)MeanShift_Length,          /* sq_length */
 NULL,                               /* sq_concat */
 NULL,                               /* sq_repeat */
 (ssizeargfunc)MeanShift_GetItem,    /* sq_item */
 NULL,                               /* sq_slice */
 NULL,                               /* sq_ass_item */
 NULL,                               /* sq_ass_slice */
 NULL,                               /* sq_contains */
 NULL,                               /* sq_inplace_concat */
 NULL,                               /* sq_inplace_repeat */
};



static PyMemberDef MeanShift_members[] =
{
 {"quality", T_FLOAT, offsetof(MeanShift, quality), 0, "Value between 0 and 1, inclusive - for kernel types that have an infinite domain this controls how much of that domain to use for the calculations - 0 for lowest quality, 1 for the highest quality. (Ignored by kernel types that have a finite kernel.) In the gaussian case it is equivalent of 1 sd for quality 0, 3 sd for quality 1; the other kernels are comparable."},
 {"epsilon", T_FLOAT, offsetof(MeanShift, epsilon), 0, "For convergance detection - when the step size is smaller than this it stops."},
 {"iter_cap", T_INT, offsetof(MeanShift, iter_cap), 0, "Maximum number of iterations to do before stopping, a hard limit on computation."},
 {"spatial_param", T_FLOAT, offsetof(MeanShift, spatial_param), 0, "A parameter passed through to the spatial data structure. Currently only used by the kd tree spatial, as the minimum dimension range that it will split - it defaults to 0.1, which almost switches this off. (There is also a depth limit and 8 node per leaf limit)"},
 {"ident_dist", T_FLOAT, offsetof(MeanShift, ident_dist), 0, "If two exemplars are found at any point to have a distance less than this from each other whilst clustering it is assumed they will go to the same destination, saving computation."},
 {"merge_range", T_FLOAT, offsetof(MeanShift, merge_range), 0, "Controls how close two mean shift locations have to be to be merged in the clustering method."},
 {"merge_check_step", T_INT, offsetof(MeanShift, merge_check_step), 0, "When clustering this controls how many mean shift iterations it does between checking for convergance - simply a tradeoff between wasting time doing mean shift when it has already converged and doing proximity checks for convergance. Should only affect runtime."},
 {"rng0", T_UINT, offsetof(MeanShift, rng[0]), 0, "Lets you set the random number generators position index - defaults to 0. Position 0 - the highest 32 bits."},
 {"rng1", T_UINT, offsetof(MeanShift, rng[1]), 0, "Lets you set the random number generators position index - defaults to 0. Position 1."},
 {"rng2", T_UINT, offsetof(MeanShift, rng[2]), 0, "Lets you set the random number generators position index - defaults to 0. Position 2."},
 {"rng3", T_UINT, offsetof(MeanShift, rng[3]), 0, "Lets you set the random number generators position index - defaults to 0. Position 3 - the lowest 32 bits."},
 {"link", T_OBJECT, offsetof(MeanShift, rng_link), READONLY, "Allows you to access the linked MeanShift object from which its getting its random number generator, None if its using its own rng0..rng3 values."},
 {NULL}
};



static PyMethodDef MeanShift_methods[] =
{
 {"kernels", (PyCFunction)MeanShift_kernels_py, METH_NOARGS | METH_STATIC, "A static method that returns a list of kernel types, as strings."},
 {"get_kernel", (PyCFunction)MeanShift_get_kernel_py, METH_NOARGS, "Returns the string that identifies the current kernel; for complex kernels this may be a complex string containing parameters etc."},
 {"get_range", (PyCFunction)MeanShift_get_range_py, METH_NOARGS, "Returns the range of the current kernel, taking into account the current quality value - this is how far out it searches for relevant exemplars from a point in space, eucledian distance. Note that the range is the raw internal value - you need to divide by the scale vector to get the true scale for each dimension. Provided for diagnostic purposes."},
 {"set_kernel", (PyCFunction)MeanShift_set_kernel_py, METH_VARARGS, "Sets the current kernel, as identified by a string. For complex kernels this will probably need to include extra information - e.g. the fisher kernel is given as fisher(alpha) where alpha is a floating point concentration parameter. Note that some kernels (e.g. fisher) take into account the number of features in the data when set - in such cases you must set the kernel type after calling set_data (An error will be thrown if set_data was never called)."},
 {"copy_kernel", (PyCFunction)MeanShift_copy_kernel_py, METH_VARARGS, "Given another MeanShift object this copies the settings from it. This is highly recomended when speed matters and you have lots of kernels, as it copies pointers to the internal configuration object and reference counts - for objects with complex configurations this can be an order of magnitude faster. It can also save a lot of memory, via shared caches."},
 
 
 {"link_rng", (PyCFunction)MeanShift_link_rng_py, METH_VARARGS, "Links this MeanShift object to use the same random number generator as the given MeanShift object, or unlinks the RNG if no argument given. Will do path shortening if you attempt to chain a bunch together. Will ignore attempts to create loops. The other MeanShift does not even have to be initialised - they don't need to share any common characteristics. Setting the rng* values of a linked MeanShift object acheives nothing."},
 
 {"spatials", (PyCFunction)MeanShift_spatials_py, METH_NOARGS | METH_STATIC, "A static method that returns a list of spatial indexing structures you can use, as strings."},
 {"get_spatial", (PyCFunction)MeanShift_get_spatial_py, METH_NOARGS, "Returns the string that identifies the current spatial indexing structure."},
 {"set_spatial", (PyCFunction)MeanShift_set_spatial_py, METH_VARARGS, "Sets the current spatial indexing structure, as identified by a string."},
 
 {"balls", (PyCFunction)MeanShift_balls_py, METH_NOARGS | METH_STATIC, "Returns a list of ball indexing techneques - this is the structure used when clustering to represent the hyper-sphere around the mode that defines a cluster in terms of merging distance."},
 {"get_balls", (PyCFunction)MeanShift_get_balls_py, METH_NOARGS, "Returns the current ball indexing structure, as a string."},
 {"set_balls", (PyCFunction)MeanShift_set_balls_py, METH_VARARGS, "Sets the current ball indexing structure, as identified by a string."},
 
 {"info", (PyCFunction)MeanShift_info_py, METH_VARARGS | METH_STATIC, "A static method that is given the name of a kernel, spatial or ball. It then returns a human readable description of that entity."},
 {"info_config", (PyCFunction)MeanShift_info_config_py, METH_VARARGS | METH_STATIC, "Given the name of a kernel this returns None if the kernel does not require any configuration, or a string describing how to configure it if it does."},
 
 {"copy_all", (PyCFunction)MeanShift_copy_all_py, METH_VARARGS, "Copies everything from another mean shift object except for the data structure - kernel, spatial, balls and all exposed parameters. Faster than doing things manually, including the sharing of caches - if you are making lots of MeanShift objects with the same parameters it is strongly recomended to use this."},
 {"reset", (PyCFunction)MeanShift_reset_py, METH_VARARGS, "Changing the contents of the numpy array that contains the samples wrapped by a MeanShift object will break things, potentially even causing a crash. However, you can change the contents of the array then call this - it will reset all data structures that are built on assumptions about the numpy array. Note that this only works for changing the contents - resizing it will not do anything as the MeanShift object will still have a pointer to the original."},
 
 {"converters", (PyCFunction)MeanShift_converters_py, METH_NOARGS | METH_STATIC, "Returns a list of converters that can be passed into the set_data method, as their single-character code strings. The converter method allows you to get details."},
 {"converter", (PyCFunction)MeanShift_converter_py, METH_VARARGS | METH_STATIC, "Given the code for a converter this returns None if its not recognised or a dictionary: {'code' : The code used to select it, single character string., 'name' : Name, provided for documentation purposes - no real use., 'description' : A human-consumable text description of the converter., 'external' : How many features it expects the external representation to have (as provided in the data matrix)., 'internal' : How many features it provides to the actual kernel - the scale and kernel itself match up with this.}."},
 
 {"set_data", (PyCFunction)MeanShift_set_data_py, METH_VARARGS, "Sets the data matrix, which defines the probability distribution via a kernel density estimate that everything is using. The data matrix is used directly, so it should not be modified during use as it could break the data structures created to accelerate question answering (unless you call reset after modification). First parameter is a numpy matrix (Any normal numerical type), the second a string with its length matching the number of dimensions of the matrix. The characters in the string define the meaning of each dimension: 'd' (data) - changing the index into this dimension changes which exemplar you are indexing; 'f' (feature) - changing the index into this dimension changes which feature you are indexing; 'b' (both) - same as d, except it also contributes an item to the feature vector, which is essentially the position in that dimension (used on the dimensions of an image for instance, to include pixel position in the feature vector). The system unwraps all data indices and all feature indices in row major order to hallucinate a standard data matrix, with all 'both' features at the start of the feature vector. Note that calling this resets scale. A third optional parameter sets an index into the original feature vector (Including the dual dimensions, so you can use one of them to provide weight) that is to be the weight of the feature vector - this effectivly reduces the length of the feature vector, as used by all other methods, by one. A fourth optional parameter can provide conversion codes, for a conversion provided after weight extraction but before scaling and the kernel is applied, so you can hallucinate the data is in a different format that is more amenable to a pdf being applied. Most common use is angles, which need to be converted into vectors for the directional kernels to make sense. The conversion is applied for all inputs and outputs so you only have to worry about the conversion when setting up scale and the kernel. There are static methods to query the conversion options."},
 {"get_dm", (PyCFunction)MeanShift_get_dm_py, METH_NOARGS, "Returns the current data matrix, which will be some kind of numpy ndarray."},
 {"get_dim", (PyCFunction)MeanShift_get_dim_py, METH_NOARGS, "Returns the string that gives the meaning of each dimension, as matched to the number of dimensions in the data matrix."},
 {"get_weight_dim", (PyCFunction)MeanShift_get_weight_dim_py, METH_NOARGS, "Returns the feature vector index that provides the weight of each sample, or None if there is not one and they are all fixed to 1."},
 
 {"fetch_dm", (PyCFunction)MeanShift_fetch_dm_py, METH_NOARGS, "Returns a new numpy ndarray, of float32 type to be indexed [exemplar, feature], regardless of what was provided to the set_data method. Basically a predicatable version of get_dm that hides weirdness. Note that the order is the same as if you were to use the numpy flatten() method on the data matrix, if you extracted one feature at a time first."},
 {"fetch_weight", (PyCFunction)MeanShift_fetch_weight_py, METH_NOARGS, "Returns a new numpy ndarray, of float32 type indexed by [exemplar] - each entry will be the weight assigned to the given exemplar, noting that if no weights are given then its just going to be a vector of ones. Partner to fetch_dm."},
 
 {"set_scale", (PyCFunction)MeanShift_set_scale_py, METH_VARARGS, "Given two parameters. First is an array indexed by feature to get a multiplier that is applied before the kernel (Which is always of radius 1, or some approximation of.) is considered - effectivly an inverse bandwidth in kernel density estimation terms or the inverse standard deviation if you are using the Gaussian kernel. Second is an optional scale for the weight assigned to each feature vector via the set_data method (In the event that no weight is assigned this parameter is the weight of each feature vector, as the default is 1). Scaling occurs after conversion, so the provided vector may not match the number of true features."},
 {"get_scale", (PyCFunction)MeanShift_get_scale_py, METH_NOARGS, "Returns a copy of the scale array (Inverse bandwidth)."},
 {"get_weight_scale", (PyCFunction)MeanShift_get_weight_scale_py, METH_NOARGS, "Returns the scalar for the weight of each sample - typically left as 1."},
 {"copy_scale", (PyCFunction)MeanShift_copy_scale_py, METH_VARARGS, "Given another MeanShift object this copies its scale parameters - a touch faster than using get then set as it avoids an intermediate numpy array."},
 
 {"exemplars", (PyCFunction)MeanShift_exemplars_py, METH_NOARGS, "Returns how many exemplars are in the hallucinated data matrix."},
 {"features", (PyCFunction)MeanShift_features_py, METH_NOARGS, "Returns how many features are in the hallucinated data matrix, before any conversion is applied."},
 {"features_internal", (PyCFunction)MeanShift_features_internal_py, METH_NOARGS, "Returns how many features are in the hallucinated data matrix as seen internally after any conversion filters have been applied. This is the length required of a scale vector."},
 {"weight", (PyCFunction)MeanShift_weight_py, METH_NOARGS, "Returns the total weight of the included data, taking into account the weight channel if provided."},
 {"stats", (PyCFunction)MeanShift_stats_py, METH_NOARGS, "Returns some basic stats about the data set - (mean, standard deviation). These are per channel and for the internaly seen features, after conversion."},

 {"scale_silverman", (PyCFunction)MeanShift_scale_silverman_py, METH_NOARGS, "Sets the scale for the current data using Silverman's rule of thumb, generalised to multidimensional data (Multidimensional version often attributed to Wand & Jones.). Note that this is assuming you are using Gaussian kernels and that the samples have been drawn from a Gaussian - if these asumptions are valid you should probably just fit a Gaussian in the first place, if they are not you should not use this method. Basically, do not use!"},
 {"scale_scott", (PyCFunction)MeanShift_scale_scott_py, METH_NOARGS, "Alternative to scale_silverman - assumptions are very similar and it is hence similarly crap - would recomend against this, though maybe prefered to Silverman."},
 {"loo_nll", (PyCFunction)MeanShift_loo_nll_py, METH_VARARGS, "Calculate the negative log liklihood of the model where it leaves out the sample whos probability is being calculated and then muliplies together the probability of all samples calculated independently. This can be used for model comparison, to see which is better out of several configurations, be that kernel size, kernel type etc. Takes two optional parameters: First, the lower bound on probability, to avoid outliers causing problems - defaults to 1e-16. Second, a limit on how many exemplars to use, rather than the default of using all of them (a negative value) - allows for an even more approximate calculation in considerably less time. The exemplars are drawn with uniform probability and replacement."},
 
 {"entropy", (PyCFunction)MeanShift_entropy_py, METH_VARARGS, "Calculates and returns an approximation of the entropy of the distribution represented by this object. As it uses the samples contained within its accuracy will improve with the number of them, much like for the rest of the system. Uses the natural logarithm, so the return is measured in nats. Has one optional parameter - a limit on how many exemplars to use, which will make it take a bootstrap draw from the exemplars and calculate the entropy from that, rather using all exemplars. This makes it more noisy, but can save a lot of computation."},
 {"kl", (PyCFunction)MeanShift_kl_py, METH_VARARGS, "Calculates and returns an approximation of the kullback leibler divergance, of the first parameter from self - D(self||arg1). In other words, it returns the average number of extra nats for encoding draws from p if you encode them optimally under the assumption they come from the density estimate of the mean shift object given as the first parameter. Uses the samples within self and solves using them as a sample from the distribution - consequntially the constraint the the KL-divergance be positive is broken by this estimate and you can get negative values out. What to do about this is left to the user. An optional second parameter provides a clamp on how low probability calculations for arg1 values are allowed to get, to avoid divide by zero - it defaults to 1e-16. An optional third parameter switches it from using all exemplars in its estiamte to using a bootstrap draw of the given size instead - saves time at the expense of more noise in the estimate."},
 
 {"prob", (PyCFunction)MeanShift_prob_py, METH_VARARGS, "Given a feature vector returns its probability, as calculated by the kernel density estimate that is defined by the data and kernel. Be warned that the return value can be zero."},
 {"probs", (PyCFunction)MeanShift_probs_py, METH_VARARGS, "Given a data matrix returns an array (1D) containing the probability of each feature, as calculated by the kernel density estimate that is defined by the data and kernel. Be warned that the return values can include zeros."},
 
 {"draw", (PyCFunction)MeanShift_draw_py, METH_NOARGS, "Allows you to draw from the distribution represented by the kernel density estimate. Returns a vector and makes use of the internal RNG."},
 {"draws", (PyCFunction)MeanShift_draws_py, METH_VARARGS, "Allows you to draw from the distribution represented by the kernel density estimate. Same as draw except it returns a matrix - you provide a single argument of how many draws to make. Returns an array, <# draws>X<# features> and makes use of the internal RNG."},
 {"bootstrap", (PyCFunction)MeanShift_bootstrap_py, METH_VARARGS, "Does a bootstrap draw from the samples - essentially the same as draws but assuming a Dirac delta function for the kernel. You provide the number of draws; it returns an array, <# draws>X<# features>. Makes use of the contained Philox RNG."},
 
 {"mode", (PyCFunction)MeanShift_mode_py, METH_VARARGS, "Given a feature vector returns its mode as calculated using mean shift - essentially the maxima in the kernel density estimate to which you converge by climbing the gradient."},
 {"modes", (PyCFunction)MeanShift_modes_py, METH_VARARGS, "Given a data matrix [exemplar, feature] returns a matrix of the same size, where each feature has been replaced by its mode, as calculated using mean shift."},
 {"modes_data", (PyCFunction)MeanShift_modes_data_py, METH_NOARGS, "Runs mean shift on the contained data set, returning a feature vector for each data point. The return value will be indexed in the same way as the provided data matrix, but without the feature dimensions, with an extra dimension at the end to index features. Note that the resulting output will contain a lot of effective duplication, making this a very inefficient method - your better off using the cluster method."},
 
 {"cluster", (PyCFunction)MeanShift_cluster_py, METH_NOARGS, "Clusters the exemplars provided by the data matrix - returns a two tuple (data matrix of all the modes in the dataset, indexed [mode, feature], A matrix of integers, indicating which mode each one has been assigned to by indexing the mode array. Indexing of this array is identical to the provided data matrix, with any feature dimensions removed.). The clustering is replaced each time this is called - do not expect cluster indices to remain consistant after calling this."},
 {"assign_cluster", (PyCFunction)MeanShift_assign_cluster_py, METH_VARARGS, "After the cluster method has been called this can be called with a single feature vector. It will then return the index of the cluster to which it has been assigned, noting that this will map to the mode array returned by the cluster method. In the event it does not map to a pre-existing cluster it will return a negative integer - this usually means it is so far from the provided data that the kernel does not include any samples."},
 {"assign_clusters", (PyCFunction)MeanShift_assign_clusters_py, METH_VARARGS, "After the cluster method has been called this can be called with a data matrix. It will then return the indices of the clusters to which each feature vector has been assigned, as a 1D numpy array, noting that this will map to the mode array returned by the cluster method. In the event any entry does not map to a pre-existing cluster it will return a negative integer for it - this usually means it is so far from the provided data that the kernel does not include any samples."},
 {"cluster_on", (PyCFunction)MeanShift_cluster_on_py, METH_VARARGS, "Acts like cluster, but instead of clustering the contained data it clusters the exemplars provided as a data matrix (only parameter) on the surface of the contained data. This can be thought of as calling the modes method on the provided data matrix and then merging modes that are sufficiently close together to obtain a set of clusters. It returns the same output as cluster, specifically a two tuple: (data matrix of all the modes on the dataset that are represented within the given exemplars, indexed [mode, feature], A matrix of integers, matching the number of provided exemplars, indicating which mode they landed in.). Note that if a provided exemplar is too far away from the given data it will form a cluster where it started; the provided exemplars only interact via cluster merging and are not included in the KDE for which modes are being found. Mode numbers will not match anything else, either other calls to this or calls to cluster."},
 
 {"manifold", (PyCFunction)MeanShift_manifold_py, METH_VARARGS, "Given a feature vector and the dimensionality of the manifold projects the feature vector onto the manfold using subspace constrained mean shift. Returns an array with the same shape as the input. A further optional boolean parameter allows you to enable calculation of the hessain for every iteration (The default, True, correct algorithm), or only do it once at the start (False, incorrect but works for clean data.)."},
 {"manifolds", (PyCFunction)MeanShift_manifolds_py, METH_VARARGS, "Given a data matrix [exemplar, feature] and the dimensionality of the manifold projects the feature vectors onto the manfold using subspace constrained mean shift. Returns a data matrix with the same shape as the input. A further optional boolean parameter allows you to enable calculation of the hessain for every iteration (The default, True, correct algorithm), or only do it once at the start (False, incorrect but works for clean data.)."},
 {"manifolds_data", (PyCFunction)MeanShift_manifolds_data_py, METH_VARARGS, "Given the dimensionality of the manifold projects the feature vectors that are defining the density estimate onto the manfold using subspace constrained mean shift. The return value will be indexed in the same way as the provided data matrix, but without the feature dimensions, with an extra dimension at the end to index features. A further optional boolean parameter allows you to enable calculation of the hessain for every iteration (The default, True, correct algorithm), or only do it once at the start (False, incorrect but works for clean data.)."},
 
 {"mult", (PyCFunction)MeanShift_mult_py, METH_KEYWORDS | METH_VARARGS | METH_STATIC, "A static method that allows you to multiply a bunch of kernel density estimates, and draw some samples from the resulting distribution, outputing the samples into an array. The first input must be a list of MeanShift objects (At least of length 1, though if length 1 it just resamples the input), the second a numpy array for the output - it must be 2D and have the same number of columns as all the MeanShift objects have features/dims; must be float or double. Its row count is how many samples will be drawn from the distribution implied by multiplying the KDEs together. Note that the first object in the MeanShift object list gets to set the kernel - it is assumed that all further objects have the same kernel, and if thats not the case expect problems. Note that this is same structure - different scales and the same kernel with different parameters (e.g. different concentration parameter for a Fisher) is fine. Further to the first two inputs dictionary parameters it allows parameters to be set by name: {'gibbs': Number of Gibbs samples to do, noting its multiplied by the length of the multiplication list and is the number of complete passes through the state, 'mci': Number of samples to do if it has to do monte carlo integration, 'mh': Number of Metropolis-Hastings steps it will do if it has to, multiplied by the length of the multiplicand list, 'fake': Allows you to request an incorrect-but-useful result - the default of 0 is the correct output, 1 is a mode from the Gibbs sampled mixture component instead of a draw, whilst 2 is the average position of the components that made up the selected mixture component.}. Note that this method makes extensive use of the built in rng."},
 
 {"__sizeof__", (PyCFunction)MeanShift_sizeof_py, METH_NOARGS, "Returns the number of bytes used the complete object. This is a non-trivial calculation, as data can be shared etc. - all the data that it definitely owns is included, as is the byte count for the numpy array that it contains a pointer to. If the kernel type includes a chache that is shared between objects it amortises it - divides the number of bytes by how many objects are using it and rounds up. This set of asusmptions means that if each MeanShoft object has its own numpy array and you sum them all up for a running program then the result is probably reasonable; in other situations it may not be. Also note that it counts all the caches etc. - many of these are not initalised until first used, or resized at various times, so size can vary a lot as you use an object."},
 {"memory", (PyCFunction)MeanShift_memory_py, METH_NOARGS, "Does the same thing as __sizeof__, except it returns a dictionary that breaks down all the byte counts that sum together to create the final value, as well as the final value. Output is {'data' : Size of contained data matrix, 'kernel' : Size of any data from the kernel (for most kernels this is 0), 'kernel_ref_count' : Data from the kernel can be shared between multiple instances - this is that count so you can amortise it if that makes sense, 'dm' : Size of data matrix information without the actual data matrix!, 'spatial' : Size of the spatial data structure, 'balls' : Size of the balls structure, 'self' : Size of just the object without all the stuff its holding pointers to, includes some internal caches, 'total' : Final output, what __sizeof__ returns.}"},
 
 {NULL}
};



static PyTypeObject MeanShiftType =
{
 PyObject_HEAD_INIT(NULL)
 0,                                /*ob_size*/
 "ms_c.MeanShift",                 /*tp_name*/
 sizeof(MeanShift),                /*tp_basicsize*/
 0,                                /*tp_itemsize*/
 (destructor)MeanShift_dealloc_py, /*tp_dealloc*/
 0,                                /*tp_print*/
 0,                                /*tp_getattr*/
 0,                                /*tp_setattr*/
 0,                                /*tp_compare*/
 0,                                /*tp_repr*/
 0,                                /*tp_as_number*/
 &MeanShift_as_sequence,           /*tp_as_sequence*/
 0,                                /*tp_as_mapping*/
 0,                                /*tp_hash */
 0,                                /*tp_call*/
 0,                                /*tp_str*/
 0,                                /*tp_getattro*/
 0,                                /*tp_setattro*/
 0,                                /*tp_as_buffer*/
 Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
 "An object implimenting mean shift; also includes kernel density estimation and subspace constrained mean shift using the same object, such that they are all using the same underlying density estimate. Has multiplication capabilities, such that you can multiply density estimates to get a further density estimate. Includes multiple spatial indexing schemes and kernel types, including ones for directional data. Clustering is supported, with a choice of cluster intersection tests, as well as the ability to interpret exemplar indexing dimensions of the data matrix as extra features, so it can handle the traditional image segmentation scenario. Note that it pretends to be a readonly list, from which you can get copies of each feature vector (len(self) will return self.exemplars()). Note that you must set the kernel after setting the data in some cases, so that the kernel knows the correct number of dimensions it needs to support.", /* tp_doc */
 0,                                /* tp_traverse */
 0,                                /* tp_clear */
 0,                                /* tp_richcompare */
 0,                                /* tp_weaklistoffset */
 0,                                /* tp_iter */
 0,                                /* tp_iternext */
 MeanShift_methods,                /* tp_methods */
 MeanShift_members,                /* tp_members */
 0,                                /* tp_getset */
 0,                                /* tp_base */
 0,                                /* tp_dict */
 0,                                /* tp_descr_get */
 0,                                /* tp_descr_set */
 0,                                /* tp_dictoffset */
 0,                                /* tp_init */
 0,                                /* tp_alloc */
 MeanShift_new_py,                 /* tp_new */
};



static PyMethodDef ms_c_methods[] =
{
 {NULL}
};



#include "bessel.h"



#ifndef PyMODINIT_FUNC
#define PyMODINIT_FUNC void
#endif

PyMODINIT_FUNC initms_c(void)
{
 PyObject * mod = Py_InitModule3("ms_c", ms_c_methods, "Primarily provides a mean shift implementation, but also includes kernel density estimation and subspace constrained mean shift using the same object, such that they are all using the same underlying density estimate. Includes multiple spatial indexing schemes and kernel types, including support for directional data. Clustering is supported, with a choice of cluster intersection tests, as well as the ability to interpret exemplar indexing dimensions of the data matrix as extra features, so it can handle the traditional image segmentation scenario efficiently. Exemplars can also be weighted. There is extensive support for particle filters as well, including multiplication of distributions for non-parametric belief propagation. Note that this module is not multithread safe - use multiprocessing instead.");
 
 import_array();
 
 if (PyType_Ready(&MeanShiftType) < 0) return;
 
 Py_INCREF(&MeanShiftType);
 PyModule_AddObject(mod, "MeanShift", (PyObject*)&MeanShiftType);
 
 // Fun little hack - there is some memory in a global pointer, so we add a capsule object to the module for no other purpose than to make sure it gets free-ed via the capsule destructor when the module is put down...
  PyObject * bessel_death = PyCapsule_New("Ignore me", NULL, FreeBesselMemory);
  PyModule_AddObject(mod, "__bessel_death", bessel_death);
}
