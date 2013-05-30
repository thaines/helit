// Copyright 2013 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



#include "ms_c.h"

#include <string.h>



void MeanShift_new(MeanShift * this)
{
 this->kernel = &Uniform;
 this->alpha = 1.0;
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
 this->ident_dist = 0.0;
 this->merge_range = 0.5;
 this->merge_check_step = 4;
}

void MeanShift_dealloc(MeanShift * this)
{
 DataMatrix_deinit(&this->dm);
 if (this->spatial!=NULL) Spatial_delete(this->spatial);
 if (this->balls!=NULL) Balls_delete(this->balls);
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
 return Py_BuildValue("s", self->kernel->name);
}


static PyObject * MeanShift_set_kernel_py(MeanShift * self, PyObject * args)
{
 // Parse the parameters...
  char * kname;
  if (!PyArg_ParseTuple(args, "s|f", &kname, &self->alpha)) return NULL;
 
 // Try and find the relevant kernel - if found assign it and return...
  int i = 0;
  while (ListKernel[i]!=NULL)
  {
   if (strcmp(ListKernel[i]->name, kname)==0)
   {
    self->kernel = ListKernel[i];
    self->norm = -1.0;
    
    Py_INCREF(Py_None);
    return Py_None;
   }
   
   ++i; 
  }
  
 // Was not succesful - throw an error...
  PyErr_SetString(PyExc_RuntimeError, "unrecognised kernel type");
  return NULL; 
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



static PyObject * MeanShift_set_data_py(MeanShift * self, PyObject * args)
{
 // Extract the parameters...
  PyArrayObject * data;
  char * dim_types;
  PyObject * weight_index = NULL;
  if (!PyArg_ParseTuple(args, "O!s|O", &PyArray_Type, &data, &dim_types, &weight_index)) return NULL;
  
 // Check its all ok...
  if (strlen(dim_types)!=data->nd)
  {
   PyErr_SetString(PyExc_RuntimeError, "dimension type string must be the same length as the number of dimensions in the data matrix");
   return NULL;
  }
  
  if ((data->descr->kind!='b')&&(data->descr->kind!='i')&&(data->descr->kind!='u')&&(data->descr->kind!='f'))
  {
   PyErr_SetString(PyExc_RuntimeError, "provided data matrix is not of a supported type");
   return NULL; 
  }
  
  int i;
  for (i=0; i<data->nd; i++)
  {
   if ((dim_types[i]!='d')&&(dim_types[i]!='f')&&(dim_types[i]!='b'))
   {
    PyErr_SetString(PyExc_RuntimeError, "dimension type string includes an unrecognised code"); 
   }
  }
 
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
  }
  
 // Make the assignment...
  DimType * dt = (DimType*)malloc(data->nd * sizeof(DimType));
  for (i=0; i<data->nd; i++)
  {
   switch (dim_types[i])
   {
    case 'd': dt[i] = DIM_DATA;    break;
    case 'f': dt[i] = DIM_FEATURE; break;
    case 'b': dt[i] = DIM_DUAL;    break;
   }
  }
  
  DataMatrix_set(&self->dm, data, dt, weight_i);
  free(dt);
  
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


static PyObject * MeanShift_set_scale_py(MeanShift * self, PyObject * args)
{
 // Extract the parameters...
  PyArrayObject * scale;
  float weight_scale = 1.0;
  if (!PyArg_ParseTuple(args, "O!|f", &PyArray_Type, &scale, &weight_scale)) return NULL;
 
 // Handle the scale...
  if ((scale->nd!=1)||(scale->dimensions[0]!=DataMatrix_features(&self->dm)))
  {
   PyErr_SetString(PyExc_RuntimeError, "scale vector must be a simple 1D numpy array with length matching the number of features.");
   return NULL;
  }
  ToFloat atof = KindToFunc(scale->descr);
  
  float * s = (float*)malloc(scale->dimensions[0] * sizeof(float));
  int i;
  for (i=0; i<scale->dimensions[0]; i++)
  {
   s[i] = atof(scale->data + i*scale->strides[0]);
  }
  
  DataMatrix_set_scale(&self->dm, s, weight_scale);
  free(s);
  
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
 return Py_BuildValue("i", DataMatrix_features(&self->dm));
}

float MeanShift_weight(MeanShift * this)
{
 if (this->weight<0.0)
 {
  this->weight = 0.0;
  int i;
  for (i=0; i<DataMatrix_exemplars(&this->dm); i++)
  {
   float w;
   DataMatrix_fv(&this->dm, i, &w);
   this->weight += w; 
  }
 }
  
 return this->weight;
}

static PyObject * MeanShift_weight_py(MeanShift * self, PyObject * args)
{
 return Py_BuildValue("f", MeanShift_weight(self));
}



static PyObject * MeanShift_prob_py(MeanShift * self, PyObject * args)
{
 // Get the argument - a feature vector... 
  PyArrayObject * start;
  if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &start)) return NULL;
  
 // Check the input is acceptable...
  npy_intp feats = DataMatrix_features(&self->dm);
  if ((start->nd!=1)||(start->dimensions[0]!=feats))
  {
   PyErr_SetString(PyExc_RuntimeError, "input vector must be 1D with the same length as the number of features.");
   return NULL;
  }
  ToFloat atof = KindToFunc(start->descr);
  
 // If spatial is null create it...
  if (self->spatial==NULL)
  {
   self->spatial = Spatial_new(self->spatial_type, &self->dm); 
  }
  
 // Calculate the normalising term if needed...
  int i;
  if (self->norm<0.0)
  {
   self->norm = self->kernel->norm(feats, self->alpha) / MeanShift_weight(self);
   for (i=0; i<feats; i++) self->norm /= self->dm.mult[i];
  }
 
 // Create a temporary to hold the feature vector...
  float * fv = (float*)malloc(feats * sizeof(float));
  
  for (i=0; i<feats; i++)
  {
   fv[i] = atof(start->data + i*start->strides[0]) * self->dm.mult[i];
  }
  
 // Calculate the probability...
  float p = prob(self->spatial, self->kernel, self->alpha, fv, self->norm, self->quality);
 
 // Return the calculated probability...
  return Py_BuildValue("f", p);
}



static PyObject * MeanShift_probs_py(MeanShift * self, PyObject * args)
{
 // Get the argument - a data matrix... 
  PyArrayObject * start;
  if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &start)) return NULL;

 // Check the input is acceptable...
  npy_intp feats = DataMatrix_features(&self->dm);
  if ((start->nd!=2)||(start->dimensions[1]!=feats))
  {
   PyErr_SetString(PyExc_RuntimeError, "input matrix must be 2D with the same length as the number of features in the second dimension");
   return NULL;
  }
  ToFloat atof = KindToFunc(start->descr);

 // If spatial is null create it...
  if (self->spatial==NULL)
  {
   self->spatial = Spatial_new(self->spatial_type, &self->dm); 
  }
  
 // Calculate the normalising term if needed...
  int i;
  if (self->norm<0.0)
  {
   self->norm = self->kernel->norm(feats, self->alpha) / MeanShift_weight(self);
   for (i=0; i<feats; i++) self->norm /= self->dm.mult[i]; 
  }
  
 // Create a temporary array of floats...
  float * fv = (float*)malloc(feats * sizeof(float));

 // Create the output array... 
  PyArrayObject * out = (PyArrayObject*)PyArray_SimpleNew(1, start->dimensions, NPY_FLOAT32);
  
  
 // Run the algorithm...
  for (i=0; i<start->dimensions[0]; i++)
  {
   // Copy the feature vector into the temporary storage...
    int j;
    for (j=0; j<feats; j++)
    {
     fv[j] = atof(start->data + i*start->strides[0] + j*start->strides[1]) * self->dm.mult[j];
    }
   
   // Calculate the probability...
    float p = prob(self->spatial, self->kernel, self->alpha, fv, self->norm, self->quality);
   
   // Store it...
    *(float*)(out->data + i * out->strides[0]) = p;
  }
  
 // Clean up...
  free(fv);
 
 // Return the assigned clusters...
  return (PyObject*)out;
}



static PyObject * MeanShift_mode_py(MeanShift * self, PyObject * args)
{
 // Get the argument - a feature vector... 
  PyArrayObject * start;
  if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &start)) return NULL;
  
 // Check the input is acceptable...
  npy_intp feats = DataMatrix_features(&self->dm);
  if ((start->nd!=1)||(start->dimensions[0]!=feats))
  {
   PyErr_SetString(PyExc_RuntimeError, "input vector must be 1D with the same length as the number of features.");
   return NULL;
  }
  ToFloat atof = KindToFunc(start->descr);
  
 // Create an output matrix, copy in the data, applying the scale change...  
  PyArrayObject * ret = (PyArrayObject*)PyArray_SimpleNew(1, &feats, NPY_FLOAT32);
  
  int i;
  for (i=0; i<feats; i++)
  {
   float * out = (float*)(ret->data + i*ret->strides[0]);
   *out = atof(start->data + i*start->strides[0]) * self->dm.mult[i];
  }
 
 // If spatial is null create it...
  if (self->spatial==NULL)
  {
   self->spatial = Spatial_new(self->spatial_type, &self->dm); 
  }
  
 // Run the agorithm; we need some temporary storage...
  float * temp = (float*)malloc(feats * sizeof(float));
  mode(self->spatial, self->kernel, self->alpha, (float*)ret->data, temp, self->quality, self->epsilon, self->iter_cap);
  free(temp);
  
 // Undo the scale change...
  for (i=0; i<feats; i++)
  {
   float * out = (float*)(ret->data + i*ret->strides[0]);
   *out /= self->dm.mult[i];
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
  dims[0] = start->dimensions[0];
  dims[1] = DataMatrix_features(&self->dm);
  
  if ((start->nd!=2)||(start->dimensions[1]!=dims[1]))
  {
   PyErr_SetString(PyExc_RuntimeError, "input matrix must be 2D with the same length as the number of features in the second dimension");
   return NULL;
  }
  ToFloat atof = KindToFunc(start->descr);
  
 // Create an output matrix, copy in the data, applying the scale change...  
  PyArrayObject * ret = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_FLOAT32);
  
  int i,j;
  for (i=0; i<dims[0]; i++)
  {
   for (j=0; j<dims[1]; j++)
   {
    *(float*)(ret->data + i*ret->strides[0] + j*ret->strides[1]) = atof(start->data + i*start->strides[0] + j*start->strides[1]) * self->dm.mult[j];
   }
  }
  
 // If spatial is null create it...
  if (self->spatial==NULL)
  {
   self->spatial = Spatial_new(self->spatial_type, &self->dm); 
  }
  
 // Calculate each mode in turn, including undo any scale changes...
  float * temp = (float*)malloc(dims[1] * sizeof(float));
  for (i=0; i<dims[0]; i++)
  {
   float * out = (float*)(ret->data + i*ret->strides[0]);
   
   mode(self->spatial, self->kernel, self->alpha, out, temp, self->quality, self->epsilon, self->iter_cap);
   
   for (j=0; j<dims[1]; j++) out[j] /= self->dm.mult[j];
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
   self->spatial = Spatial_new(self->spatial_type, &self->dm); 
  }

 // Work out the output matrix size...
  int nd = 1;
  int i;
  for (i=0; i<self->dm.array->nd; i++)
  {
   if (self->dm.dt[i]!=DIM_FEATURE) nd += 1;
  }
  
  npy_intp * dims = (npy_intp*)malloc(nd * sizeof(npy_intp));
  
  nd = 0;
  for (i=0; i<self->dm.array->nd; i++)
  {
   if (self->dm.dt[i]!=DIM_FEATURE)
   {
    dims[nd] = self->dm.array->dimensions[i];
    nd += 1;
   }
  }
  
  dims[nd] = DataMatrix_features(&self->dm);
  nd += 1;
 
 // Create the output matrix...
  PyArrayObject * ret = (PyArrayObject*)PyArray_SimpleNew(nd, dims, NPY_FLOAT32);
 
 // Iterate and do each entry in turn...
  float * temp = (float*)malloc(dims[nd-1] * sizeof(float));
  
  float * out = (float*)ret->data;
  int loc = 0;
  while (loc<DataMatrix_exemplars(&self->dm))
  {
   // Copy in the relevent feature vector...
    float * fv = DataMatrix_fv(&self->dm, loc, NULL);
    for (i=0; i<dims[nd-1]; i++) out[i] = fv[i];
   
   // Converge mean shift...
    mode(self->spatial, self->kernel, self->alpha, out, temp, self->quality, self->epsilon, self->iter_cap);
    
   // Undo any scale change...
    for (i=0; i<dims[nd-1]; i++) out[i] /= self->dm.mult[i];
    
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
   self->spatial = Spatial_new(self->spatial_type, &self->dm); 
  }

 // Work out the output matrix size...
  int nd = 0;
  int i;
  for (i=0; i<self->dm.array->nd; i++)
  {
   if (self->dm.dt[i]!=DIM_FEATURE) nd += 1;
  }
  
  if (nd<2) nd = 2; // So the array can be abused below
  npy_intp * dims = (npy_intp*)malloc(nd * sizeof(npy_intp));
  
  nd = 0;
  for (i=0; i<self->dm.array->nd; i++)
  {
   if (self->dm.dt[i]!=DIM_FEATURE)
   {
    dims[nd] = self->dm.array->dimensions[i];
    nd += 1;
   }
  }
  
 // Create the output matrix...
  PyArrayObject * index = (PyArrayObject*)PyArray_SimpleNew(nd, dims, NPY_INT32);
 
 // Create the balls...
  if (self->balls!=NULL) Balls_delete(self->balls);
  self->balls = Balls_new(self->balls_type, self->dm.feats, self->merge_range);
 
 // Do the work...
  cluster(self->spatial, self->kernel, self->alpha, self->balls, (int*)index->data, self->quality, self->epsilon, self->iter_cap, self->ident_dist, self->merge_range, self->merge_check_step);
 
 // Extract the modes, which happen to be the centers of the balls...
  dims[0] = Balls_count(self->balls);
  dims[1] = Balls_dims(self->balls);
  
  PyArrayObject * modes = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_FLOAT32);
  
  for (i=0; i<dims[0]; i++)
  {
   const float * loc = Balls_pos(self->balls, i);
   
   int j;
   for (j=0; j<dims[1]; j++)
   {
    ((float*)modes->data)[i*dims[1] + j] = loc[j] / self->dm.mult[j]; 
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
  npy_intp feats = DataMatrix_features(&self->dm);
  if ((start->nd!=1)||(start->dimensions[0]!=feats))
  {
   PyErr_SetString(PyExc_RuntimeError, "input vector must be 1D with the same length as the number of features.");
   return NULL;
  }
  ToFloat atof = KindToFunc(start->descr);
  
 // Verify that cluster has been run...
  if (self->balls==NULL)
  {
   PyErr_SetString(PyExc_RuntimeError, "the cluster method must be run before the assign_cluster method.");
   return NULL; 
  }
  
 // Create two temporary array fo floats, putting the feature vector into one of them...
  float * fv = (float*)malloc(feats * sizeof(float));
  float * temp = (float*)malloc(feats * sizeof(float));

  int i;
  for (i=0; i<feats; i++)
  {
   fv[i] = atof(start->data + i*start->strides[0]) * self->dm.mult[i];
  }
 
 // If spatial is null create it...
  if (self->spatial==NULL)
  {
   self->spatial = Spatial_new(self->spatial_type, &self->dm); 
  }
  
 // Run the algorithm...
  int cluster = assign_cluster(self->spatial, self->kernel, self->alpha, self->balls, fv, temp, self->quality, self->epsilon, self->iter_cap, self->merge_check_step);
  
 // Clean up...
  free(temp);
  free(fv);
 
 // Return the assigned cluster...
  return Py_BuildValue("i", cluster);
}



static PyObject * MeanShift_assign_clusters_py(MeanShift * self, PyObject * args)
{
 // Get the argument - a feature vector... 
  PyArrayObject * start;
  if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &start)) return NULL;
  
 // Check the input is acceptable...
  npy_intp feats = DataMatrix_features(&self->dm);
  if ((start->nd!=2)||(start->dimensions[1]!=feats))
  {
   PyErr_SetString(PyExc_RuntimeError, "input vector must be 2D with the second dimension the same length as the number of features.");
   return NULL;
  }
  ToFloat atof = KindToFunc(start->descr);
  
 // Verify that cluster has been run...
  if (self->balls==NULL)
  {
   PyErr_SetString(PyExc_RuntimeError, "the cluster method must be run before the assign_cluster method.");
   return NULL; 
  }
  
 // Create two temporary array of floats...
  float * fv = (float*)malloc(feats * sizeof(float));
  float * temp = (float*)malloc(feats * sizeof(float));

 // Create the output array... 
  PyArrayObject * cluster = (PyArrayObject*)PyArray_SimpleNew(1, start->dimensions, NPY_INT32);
 
 // If spatial is null create it...
  if (self->spatial==NULL)
  {
   self->spatial = Spatial_new(self->spatial_type, &self->dm); 
  }
  
 // Run the algorithm...
  int i;
  for (i=0; i<start->dimensions[0]; i++)
  {
   // Copy the feature vector into the temporary storage...
    int j;
    for (j=0; j<feats; j++)
    {
     fv[j] = atof(start->data + i*start->strides[0] + j*start->strides[1]) * self->dm.mult[j];
    }
   
   // Run it...
    int c = assign_cluster(self->spatial, self->kernel, self->alpha, self->balls, fv, temp, self->quality, self->epsilon, self->iter_cap, self->merge_check_step);
   
   // Store the result...
    *(int*)(cluster->data + i * cluster->strides[0]) = c;
  }
  
 // Clean up...
  free(temp);
  free(fv);
 
 // Return the assigned clusters...
  return (PyObject*)cluster;
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
  npy_intp feats = DataMatrix_features(&self->dm);
  if ((start->nd!=1)||(start->dimensions[0]!=feats))
  {
   PyErr_SetString(PyExc_RuntimeError, "input vector must be 1D with the same length as the number of features.");
   return NULL;
  }
  ToFloat atof = KindToFunc(start->descr);
  
 // Create an output matrix, copy in the data, applying the scale change...  
  PyArrayObject * ret = (PyArrayObject*)PyArray_SimpleNew(1, &feats, NPY_FLOAT32);
  
  int i;
  for (i=0; i<feats; i++)
  {
   float * out = (float*)(ret->data + i*ret->strides[0]);
   *out = atof(start->data + i*start->strides[0]) * self->dm.mult[i];
  }
 
 // If spatial is null create it...
  if (self->spatial==NULL)
  {
   self->spatial = Spatial_new(self->spatial_type, &self->dm); 
  }
  
 // Run the agorithm; we need some temporary storage...
  float * grad = (float*)malloc(feats * sizeof(float));
  float * hess = (float*)malloc(feats * feats * sizeof(float));
  float * eigen_vec = (float*)malloc(feats * feats * sizeof(float));
  float * eigen_val = (float*)malloc(feats * sizeof(float));
  
  manifold(self->spatial, degrees, (float*)ret->data, grad, hess, eigen_val, eigen_vec, self->quality, self->epsilon, self->iter_cap, (always_hessian==Py_False) ? 0 : 1);
  
  free(eigen_val);
  free(eigen_vec);
  free(hess);
  free(grad);
  
 // Undo the scale change...
  for (i=0; i<feats; i++)
  {
   float * out = (float*)(ret->data + i*ret->strides[0]);
   *out /= self->dm.mult[i];
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
  dims[0] = start->dimensions[0];
  dims[1] = DataMatrix_features(&self->dm);
  
  if ((start->nd!=2)||(start->dimensions[1]!=dims[1]))
  {
   PyErr_SetString(PyExc_RuntimeError, "input matrix must be 2D with the same length as the number of features in the second dimension");
   return NULL;
  }
  ToFloat atof = KindToFunc(start->descr);
  
 // Create an output matrix, copy in the data, applying the scale change...  
  PyArrayObject * ret = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_FLOAT32);
  
  int i,j;
  for (i=0; i<dims[0]; i++)
  {
   for (j=0; j<dims[1]; j++)
   {
    *(float*)(ret->data + i*ret->strides[0] + j*ret->strides[1]) = atof(start->data + i*start->strides[0] + j*start->strides[1]) * self->dm.mult[j];
   }
  }
  
 // If spatial is null create it...
  if (self->spatial==NULL)
  {
   self->spatial = Spatial_new(self->spatial_type, &self->dm); 
  }
  
 // Calculate each mode in turn, including undo any scale changes...
  float * grad = (float*)malloc(dims[1] * sizeof(float));
  float * hess = (float*)malloc(dims[1] * dims[1] * sizeof(float));
  float * eigen_vec = (float*)malloc(dims[1] * dims[1] * sizeof(float));
  float * eigen_val = (float*)malloc(dims[1] * sizeof(float));
  
  for (i=0; i<dims[0]; i++)
  {
   float * out = (float*)(ret->data + i*ret->strides[0]);
   
   manifold(self->spatial, degrees, out, grad, hess, eigen_val, eigen_vec, self->quality, self->epsilon, self->iter_cap, (always_hessian==Py_False) ? 0 : 1);
   
   for (j=0; j<dims[1]; j++) out[j] /= self->dm.mult[j];
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
   self->spatial = Spatial_new(self->spatial_type, &self->dm); 
  }

 // Work out the output matrix size...
  int nd = 1;
  int i;
  for (i=0; i<self->dm.array->nd; i++)
  {
   if (self->dm.dt[i]!=DIM_FEATURE) nd += 1;
  }
  
  npy_intp * dims = (npy_intp*)malloc(nd * sizeof(npy_intp));
  
  nd = 0;
  for (i=0; i<self->dm.array->nd; i++)
  {
   if (self->dm.dt[i]!=DIM_FEATURE)
   {
    dims[nd] = self->dm.array->dimensions[i];
    nd += 1;
   }
  }
  
  dims[nd] = DataMatrix_features(&self->dm);
  nd += 1;
 
 // Create the output matrix...
  PyArrayObject * ret = (PyArrayObject*)PyArray_SimpleNew(nd, dims, NPY_FLOAT32);
 
 // Iterate and do each entry in turn...
  float * grad = (float*)malloc(dims[1] * sizeof(float));
  float * hess = (float*)malloc(dims[1] * dims[1] * sizeof(float));
  float * eigen_vec = (float*)malloc(dims[1] * dims[1] * sizeof(float));
  float * eigen_val = (float*)malloc(dims[1] * sizeof(float));
  
  float * out = (float*)ret->data;
  int loc = 0;
  while (loc<DataMatrix_exemplars(&self->dm))
  {
   // Copy in the relevent feature vector...
    float * fv = DataMatrix_fv(&self->dm, loc, NULL);
    for (i=0; i<dims[nd-1]; i++) out[i] = fv[i];
   
   // Converge mean shift...
    manifold(self->spatial, degrees, out, grad, hess, eigen_val, eigen_vec, self->quality, self->epsilon, self->iter_cap, (always_hessian==Py_False) ? 0 : 1);
    
   // Undo any scale change...
    for (i=0; i<dims[nd-1]; i++) out[i] /= self->dm.mult[i];

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



static PyMemberDef MeanShift_members[] =
{
 {"alpha", T_FLOAT, offsetof(MeanShift, alpha), 0, "Arbitrary parameter that is passed through to the kernel - most kernels ignore it. Only current use is for the 'fisher' kernel, where it is the concentration of the von-Mises Fisher distribution."},
 {"quality", T_FLOAT, offsetof(MeanShift, quality), 0, "Value between 0 and 1, inclusive - for kernel types that have an infinite domain this controls how much of that domain to use for the calculations - 0 for lowest quality, 1 for the highest quality. (Ignored by kernel types that have a finite kernel.)"},
 {"epsilon", T_FLOAT, offsetof(MeanShift, epsilon), 0, "For convergance detection - when the step size is smaller than this it stops."},
 {"iter_cap", T_INT, offsetof(MeanShift, iter_cap), 0, "Maximum number of iterations to do before stopping, a hard limit on computation."},
 {"ident_dist", T_FLOAT, offsetof(MeanShift, ident_dist), 0, "If two exemplars are found at any point to have a distance less than this from each other whilst clustering it is assumed they will go to the same destination, saving computation."},
 {"merge_range", T_FLOAT, offsetof(MeanShift, merge_range), 0, "Controls how close two mean shift locations have to be to be merged in the clustering method."},
 {"merge_check_step", T_INT, offsetof(MeanShift, merge_check_step), 0, "When clustering this controls how many mean shift iterations it does between checking for convergance - simply a tradeoff between wasting time doing mean shift when it has already converged and doing proximity checks for convergance. Should only affects runtime."},
 {NULL}
};



static PyMethodDef MeanShift_methods[] =
{
 {"kernels", (PyCFunction)MeanShift_kernels_py, METH_NOARGS | METH_STATIC, "A static method that returns a list of kernel types, as strings."},
 {"get_kernel", (PyCFunction)MeanShift_get_kernel_py, METH_NOARGS, "Returns the string that identifies the current kernel."},
 {"set_kernel", (PyCFunction)MeanShift_set_kernel_py, METH_VARARGS, "Sets the current kernel, as identified by a string. An optional second parameter exists, for the alpha parameter, which is passed through to the kernel. Most kernels ignore this parameter - right now only the 'fisher' kernel uses it, where it is the concentration parameter of the von-Mises Fisher distribution that is used."},
 
 {"spatials", (PyCFunction)MeanShift_spatials_py, METH_NOARGS | METH_STATIC, "A static method that returns a list of spatial indexing structures you can use, as strings."},
 {"get_spatial", (PyCFunction)MeanShift_get_spatial_py, METH_NOARGS, "Returns the string that identifies the current spatial indexing structure."},
 {"set_spatial", (PyCFunction)MeanShift_set_spatial_py, METH_VARARGS, "Sets the current spatial indexing structure, as identified by a string."},
 
 {"balls", (PyCFunction)MeanShift_balls_py, METH_NOARGS | METH_STATIC, "Returns a list of ball indexing techneques - this is the structure used when clustering to represent the hyper-sphere around the mode that defines a cluster in terms of merging distance."},
 {"get_balls", (PyCFunction)MeanShift_get_balls_py, METH_NOARGS, "Returns the current ball indexing structure, as a string."},
 {"set_balls", (PyCFunction)MeanShift_set_balls_py, METH_VARARGS, "Sets the current ball indexing structure, as identified by a string."},
 
 {"info", (PyCFunction)MeanShift_info_py, METH_VARARGS | METH_STATIC, "A static method that is given the name of a kernel, spatial or ball. It then returns a human readable description of that entity."},
 
 {"set_data", (PyCFunction)MeanShift_set_data_py, METH_VARARGS, "Sets the data matrix, which defines the probability distribution via a kernel density estimate that everything is using. First parameter is a numpy matrix (Any normal numerical type), the second a string with its length matching the number of dimensions of the matrix. The characters in the string define the meaning of each dimension: 'd' (data) - changing the index into this dimension changes which exemplar you are indexing; 'f' (feature) - changing the index into this dimension changes which feature you are indexing; 'b' (both) - same as d, except it also contributes an item to the feature vector, which is essentially the position in that dimension (used on the dimensions of an image for instance, to include pixel position in the feature vector). The system unwraps all data indices and all feature indices in row major order to hallucinate a standard data matrix, with all 'both' features at the start of the feature vector. Note that calling this resets scale. A third optional parameter sets an index into the original feature vector that is to be the weight of the feature vector - this effectivly reduces the length of the feature vector, as used by all other methods, by one."},
 {"set_scale", (PyCFunction)MeanShift_set_scale_py, METH_VARARGS, "Given two parameters. First is an array indexed by feature to get a multiplier that is applied before the kernel (Which is always of radius 1, or some approximation of.) is considered - effectivly an inverse bandwidth in kernel density estimation terms. Second is an optional scale for the weight assigned to each feature vector via the set_data method (In the event that no weight is assigned this parameter is the weight of each feature vector, as the default is 1)."},
 
 {"exemplars", (PyCFunction)MeanShift_exemplars_py, METH_NOARGS, "Returns how many exemplars are in the hallucinated data matrix."},
 {"features", (PyCFunction)MeanShift_features_py, METH_NOARGS, "Returns how many features are in the hallucinated data matrix."},
 {"weight", (PyCFunction)MeanShift_weight_py, METH_NOARGS, "Returns the total weight of the included data, taking into account a weight channel if provided."},
 
 {"prob", (PyCFunction)MeanShift_prob_py, METH_VARARGS, "Given a feature vector returns its probability, as calculated by the kernel density estimate that is defined by the data and kernel. Be warned that the return value can be zero."},
 {"probs", (PyCFunction)MeanShift_probs_py, METH_VARARGS, "Given a data matrix returns an array (1D) containing the probability of each feature, as calculated by the kernel density estimate that is defined by the data and kernel. Be warned that the return value can be zero."},
 
 {"mode", (PyCFunction)MeanShift_mode_py, METH_VARARGS, "Given a feature vector returns its mode as calculated using mean shift - essentially the maxima in the kernel density estimate to which you converge by climbing the gradient."},
 {"modes", (PyCFunction)MeanShift_modes_py, METH_VARARGS, "Given a data matrix [exemplar, feature] returns a matrix of the same size, where each feature has been replaced by its mode, as calculated using mean shift."},
 {"modes_data", (PyCFunction)MeanShift_modes_data_py, METH_NOARGS, "Runs mean shift on the contained data set, returning a feature vector for each data point. The return value will be indexed in the same way as the provided data matrix, but without the feature dimensions, with an extra dimension at the end to index features. Note that the resulting output will contain a lot of effective duplication, making this a very inefficient method - your better off using the cluster method."},
 
 {"cluster", (PyCFunction)MeanShift_cluster_py, METH_NOARGS, "Clusters the exemplars provided by the data matrix - returns a two tuple (data matrix of all the modes in the dataset, indexed [mode, feature], A matrix of integers, indicating which mode each one has been assigned to by indexing the mode array. Indexing of this array is identical to the provided data matrix, with any feature dimensions removed.). The clustering is replaced each time this is called - do not expect cluster indices to remain consistant after calling this."},
 {"assign_cluster", (PyCFunction)MeanShift_assign_cluster_py, METH_VARARGS, "After the cluster method has been called this can be called with a single feature vector. It will then return the index of the cluster to which it has been assigned, noting that this will map to the mode array returned by the cluster method. In the event it does not map to a pre-existing cluster it will return a negative integer - this usually means it is so far from the provided data that the kernel does not include any samples."},
 {"assign_clusters", (PyCFunction)MeanShift_assign_clusters_py, METH_VARARGS, "After the cluster method has been called this can be called with a data matrix. It will then return the indices of the clusters to which each feature vector has been assigned, as a 1D numpy array, noting that this will map to the mode array returned by the cluster method. In the event any entry does not map to a pre-existing cluster it will return a negative integer for it - this usually means it is so far from the provided data that the kernel does not include any samples."},
 
 {"manifold", (PyCFunction)MeanShift_manifold_py, METH_VARARGS, "Given a feature vector and the dimensionality of the manifold projects the feature vector onto the manfold using subspace constrained mean shift. Returns an array with the same shape as the input. A further optional boolean parameter allows you to enable calculation of the hessain for every iteration (The default, True, correct algorithm), or only do it once at the start (False, incorrect but works for clean data.)."},
 {"manifolds", (PyCFunction)MeanShift_manifolds_py, METH_VARARGS, "Given a data matrix [exemplar, feature] and the dimensionality of the manifold projects the feature vectors onto the manfold using subspace constrained mean shift. Returns a data matrix with the same shape as the input. A further optional boolean parameter allows you to enable calculation of the hessain for every iteration (The default, True, correct algorithm), or only do it once at the start (False, incorrect but works for clean data.)."},
 {"manifolds_data", (PyCFunction)MeanShift_manifolds_data_py, METH_VARARGS, "Given the dimensionality of the manifold projects the feature vectors that are defining the density estimate onto the manfold using subspace constrained mean shift. The return value will be indexed in the same way as the provided data matrix, but without the feature dimensions, with an extra dimension at the end to index features. A further optional boolean parameter allows you to enable calculation of the hessain for every iteration (The default, True, correct algorithm), or only do it once at the start (False, incorrect but works for clean data.)."},
 
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
 0,                                /*tp_as_sequence*/
 0,                                /*tp_as_mapping*/
 0,                                /*tp_hash */
 0,                                /*tp_call*/
 0,                                /*tp_str*/
 0,                                /*tp_getattro*/
 0,                                /*tp_setattro*/
 0,                                /*tp_as_buffer*/
 Py_TPFLAGS_DEFAULT,               /*tp_flags*/
 "An object implimenting mean shift; also includes kernel density estimation and subspace constrained mean shift using the same object, such that they are all using the same underlying density estimate. Includes multiple spatial indexing schemes and kernel types, including one for directional data. Clustering is supported, with a choice of cluster intersection tests, as well as the ability to interpret exemplar indexing dimensions of the data matrix as extra features, so it can handle the traditional image segmentation scenario.", /* tp_doc */
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



#ifndef PyMODINIT_FUNC
#define PyMODINIT_FUNC void
#endif

PyMODINIT_FUNC initms_c(void)
{
 PyObject * mod = Py_InitModule3("ms_c", ms_c_methods, "Primarily provides a mean shift implementation, but also includes kernel density estimation and subspace constrained mean shift using the same object, such that they are all using the same underlying density estimate. Includes multiple spatial indexing schemes and kernel types, including one for directional data. Clustering is supported, with a choice of cluster intersection tests, as well as the ability to interpret exemplar indexing dimensions of the data matrix as extra features, so it can handle the traditional image segmentation scenario.");
 
 import_array();
 
 if (PyType_Ready(&MeanShiftType) < 0) return;
 
 Py_INCREF(&MeanShiftType);
 PyModule_AddObject(mod, "MeanShift", (PyObject*)&MeanShiftType);
}
