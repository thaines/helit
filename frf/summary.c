// Copyright 2014 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



#include "summary.h"

#include "data_matrix.h"
#include "index_set.h"



// The helper methods for calling the methods of arbitrary summary objects...
size_t Summary_init_size(char code, DataMatrix * dm, IndexView * view, int feature)
{
 return CodeSummary[(unsigned char)code]->init_size(dm, view, feature);  
}

void Summary_init(char code, Summary this, DataMatrix * dm, IndexView * view, int feature)
{
 CodeSummary[(unsigned char)code]->init(this, dm, view, feature);
}

float Summary_error(char code, Summary this, DataMatrix * dm, IndexView * view, int feature)
{
 return CodeSummary[(unsigned char)code]->error(this, dm, view, feature); 
}

PyObject * Summary_merge_py(char code, int trees, Summary * sums, SummaryMagic magic, int extra)
{
 return CodeSummary[(unsigned char)code]->merge_py(trees, sums, magic, extra);
}

PyObject * Summary_merge_many_py(char code, int exemplars, int trees, Summary * sums, SummaryMagic magic, int extra)
{
 return CodeSummary[(unsigned char)code]->merge_many_py(exemplars, trees, sums, magic, extra);
}

size_t Summary_size(char code, Summary this)
{
 return CodeSummary[(unsigned char)code]->size(this);
}

PyObject * Summary_string(char code, Summary this)
{
 return CodeSummary[(unsigned char)code]->string(this); 
}



// The nothing summary type - I feel as empty writting this as I am sure you do reading it...
static size_t Nothing_init_size(DataMatrix * dm, IndexView * view, int feature)
{
 return 0; 
}

static void Nothing_init(Summary self, DataMatrix * dm, IndexView * view, int feature)
{
 // No-op
}

static float Nothing_error(Summary self, DataMatrix * dm, IndexView * view, int feature)
{
 return 0.0;  
}

static PyObject * Nothing_merge_py(int trees, Summary * sums, SummaryMagic magic, int extra)
{
 Py_INCREF(Py_None);
 return Py_None;
}

static PyObject * Nothing_merge_many_py(int exemplars, int trees, Summary * sums, SummaryMagic magic, int extra)
{
 Py_INCREF(Py_None);
 return Py_None;
}

static size_t Nothing_size(Summary this)
{
 return 0;  
}

static PyObject * Nothing_string(Summary this)
{
 return PyString_FromFormat("nothing()");
}


const SummaryType NothingSummary =
{
 'N',
 "Nothing",
 "A summary type that does nothing - does not in any way summarise the feature index it is assigned to. For if you either have a multi-index summary type on an earlier feature, and hence don't need to summarise this feature index twice, or have some excess feature in your data structure and just want to ignore it.",
 Nothing_init_size,
 Nothing_init,
 Nothing_error,
 Nothing_merge_py,
 Nothing_merge_many_py,
 Nothing_size,
 Nothing_string,
};



// The categorical summary type - a categorical distribution built from a discrete feature. Note that it ignores negative feature values, making that equivalent to an unknown value (Similarly, if the user sets a low max value for the datamatrix then high values will be ignored.), and assumes they are 0 based so it can use a simple array...
typedef struct Categorical Categorical;

struct Categorical
{
 int count; // # Of samples that went into the distribution - has its uses.
 int cats; // Number of categories.
 float prob[0]; // Probability of each category.
};


static size_t Categorical_init_size(DataMatrix * dm, IndexView * view, int feature)
{
 int cats = DataMatrix_Max(dm, feature) + 1;
 return sizeof(Categorical) + cats * sizeof(float);
}

static void Categorical_init(Summary self, DataMatrix * dm, IndexView * view, int feature)
{
 Categorical * this = (Categorical*)self;
 
 this->count = 0;
 this->cats = DataMatrix_Max(dm, feature) + 1;
 
 int i;
 for (i=0; i<this->cats; i++)
 {
  this->prob[i] = 0.0;
 }
   
 for (i=0; i<view->size; i++)
 {
  int exemplar = view->vals[i];
  int value = DataMatrix_GetDiscrete(dm, exemplar, feature);
  
  if ((value>=0)&&(value<this->cats))
  {
   this->count += 1;
   this->prob[value] += 1.0;
  }
 }
 
 if (this->count!=0)
 {
  for (i=0; i<this->cats; i++)
  {
   this->prob[i] /= this->count;
  }
 }
 else
 {
  // No data - set to a uniform as a stupid fallback...
   for (i=0; i<this->cats; i++)
   {
    this->prob[i] = 1.0 / this->cats;
   }
 }
}

static float Categorical_error(Summary self, DataMatrix * dm, IndexView * view, int feature)
{
 Categorical * this = (Categorical*)self;
 
 int best = 0;
 int i;
 for (i=1; i<this->cats; i++)
 {
  if (this->prob[i]>this->prob[best])
  {
   best = i;
  }
 }
 
 float ret = 0.0;
 for (i=0; i<view->size; i++)
 {
  int v = DataMatrix_GetDiscrete(dm, view->vals[i], feature);
  if (best!=v) ret += 1.0;
 }
 
 return ret;
}

static PyObject * Categorical_merge_py(int trees, Summary * sums, SummaryMagic magic, int extra)
{
 // Setup the output...
  int count = 0;
  
  Categorical * first = (Categorical*)sums[0];
  if (magic!=NULL) first = (Categorical*)magic(first, extra);
  
  npy_intp size = first->cats;
  PyArrayObject * prob = (PyArrayObject*)PyArray_SimpleNew(1, &size, NPY_FLOAT32);
 
  int i, j;
  for (i=0; i<size; i++)
  {
   *(float*)PyArray_GETPTR1(prob, i) = 0.0;
  }
  
 // Loop and record the values...
  float total = 0.0;
  for (j=0; j<trees; j++)
  {
   Categorical * targ = (Categorical*)sums[j];
   if (magic!=NULL) targ = (Categorical*)magic(targ, extra);
   
   count += targ->count;
   
   for (i=0; i<size; i++)
   {
    *(float*)PyArray_GETPTR1(prob, i) += targ->prob[i];
    total += targ->prob[i];
   }
  }
  
  for (i=0; i<size; i++)
  {
   *(float*)PyArray_GETPTR1(prob, i) /= total;
  }
 
 // Build and return the dictionary...
  return Py_BuildValue("{sisN}", "count", count, "prob", prob);
}

static PyObject * Categorical_merge_many_py(int exemplars, int trees, Summary * sums, SummaryMagic magic, int extra)
{
 // Setup the output...
  Categorical * first = (Categorical*)sums[0];
  if (magic!=NULL) first = (Categorical*)magic(first, extra);
  
  npy_intp size[2] = {exemplars, first->cats};
  PyArrayObject * count = (PyArrayObject*)PyArray_SimpleNew(1, size, NPY_INT32);
  PyArrayObject * prob = (PyArrayObject*)PyArray_SimpleNew(2, size, NPY_FLOAT32);
 
  int i, j, k;
  
  for (k=0; k<size[0]; k++)
  {
   *(int*)PyArray_GETPTR1(count, k) = 0;
   for (i=0; i<size[1]; i++)
   {
    *(float*)PyArray_GETPTR2(prob, k, i) = 0.0;
   }
  }
  
 // Loop and record the values...
  for (k=0; k<size[0]; k++)
  {
   float total = 0.0;
   for (j=0; j<trees; j++)
   {
    Categorical * targ = (Categorical*)sums[k*trees + j];
    if (magic!=NULL) targ = (Categorical*)magic(targ, extra);
   
    *(int*)PyArray_GETPTR1(count, k) += targ->count;
   
    for (i=0; i<size[1]; i++)
    {
     *(float*)PyArray_GETPTR2(prob, k, i) += targ->prob[i];
     total += targ->prob[i];
    }
   }
  
   for (i=0; i<size[1]; i++)
   {
    *(float*)PyArray_GETPTR2(prob, k, i) /= total;
   }
  }
 
 // Build and return the dictionary...
  return Py_BuildValue("{sNsN}", "count", count, "prob", prob);
}

static size_t Categorical_size(Summary self)
{
 Categorical * this = (Categorical*)self;
 return sizeof(Categorical) + this->cats * sizeof(float);  
}

static PyObject * Categorical_string(Summary self)
{
 Categorical * this = (Categorical*)self;
 
 PyObject * ret = NULL;
 int i;
 for (i=0; i<this->cats; i++)
 {
  char * s = PyOS_double_to_string(this->prob[i], 'f', 3, 0, NULL);
  
  if (ret==NULL) ret = PyString_FromFormat("categorical([%s", s);
  else
  {
   PyString_ConcatAndDel(&ret, PyString_FromFormat(",%s", s));
  }
  
  PyMem_Free(s);
 }
 
 PyString_ConcatAndDel(&ret, PyString_FromFormat("]|%i)", this->count));
 return ret;
}


const SummaryType CategoricalSummary =
{
 'C',
 "Categorical",
 "A standard categorical distribution for discrete features. The indices are taken to go from 1 to the maximum given by the datamatrix, inclusive - any value outside this is ignored, effectivly being treated as unknown. Output when converted to a python object is a dictionary - the key 'count' gets the number of samples that went into the distribution, whilst 'prob' gets an array, indexed by category, of the probabilities of each. For the array case the count gets a 1D array and cat becomes 2D, indexed [exemplar, cat]. The error calculation is simply zero for most probable value matching, 1 for it not matching.",
 Categorical_init_size,
 Categorical_init,
 Categorical_error,
 Categorical_merge_py,
 Categorical_merge_many_py,
 Categorical_size,
 Categorical_string,
};



// The univariate Gaussian for real features - exactly what you would expect...
typedef struct Gaussian Gaussian;

struct Gaussian
{
 int count; // # Of samples that went into the distribution - has its uses.
 float mean;
 float var;
};


static size_t Gaussian_init_size(DataMatrix * dm, IndexView * view, int feature)
{
 return sizeof(Gaussian); 
}

static void Gaussian_init(Summary self, DataMatrix * dm, IndexView * view, int feature)
{
 Gaussian * this = (Gaussian*)self;
 
 this->count = 0;
 this->mean = 0.0;
 this->var = 0.0;
 
 int i;
 for (i=0; i<view->size; i++)
 {
  int exemplar = view->vals[i];
  float value = DataMatrix_GetContinuous(dm, exemplar, feature);
  
  this->count += 1;
  float delta = value - this->mean;
  this->mean += delta / this->count;
  this->var += delta * (value - this->mean);
 }
 
 if (this->count!=0) this->var /= this->count;
}

static float Gaussian_error(Summary self, DataMatrix * dm, IndexView * view, int feature)
{
 Gaussian * this = (Gaussian*)self;
 
 float ret = 0.0;
 
 int i;
 for (i=0; i<view->size; i++)
 {
  float v = DataMatrix_GetContinuous(dm, view->vals[i], feature);
  ret += fabs(v - this->mean);
 }
 
 return ret;
}

static PyObject * Gaussian_merge_py(int trees, Summary * sums, SummaryMagic magic, int extra)
{
 // Combine from all trees...
  int count = 0;
  float mean = 0.0;
  float var = 0.0;
  
  int i;
  for (i=0; i<trees; i++)
  {
   Gaussian * targ = (Gaussian*)sums[i];
   if (magic!=NULL) targ = (Gaussian*)magic(targ, extra);
   
   int new_count = count + targ->count;
   float delta = targ->mean - mean;
   float offset = delta * targ->count / (float)new_count;
   mean += offset;
   var += (targ->var * targ->count) + offset * count * delta;
   count = new_count;
  }
  
  if (count!=0) var /= count;
 
 // Build and return a dictionary containing the required values...
  return Py_BuildValue("{sisfsf}", "count", count, "mean", mean, "var", var);
}

static PyObject * Gaussian_merge_many_py(int exemplars, int trees, Summary * sums, SummaryMagic magic, int extra)
{
 // Create the output numpy arrays...
  npy_intp size = exemplars;
  PyArrayObject * count_arr = (PyArrayObject*)PyArray_SimpleNew(1, &size, NPY_INT32);
  PyArrayObject * mean_arr  = (PyArrayObject*)PyArray_SimpleNew(1, &size, NPY_FLOAT32);
  PyArrayObject * var_arr   = (PyArrayObject*)PyArray_SimpleNew(1, &size, NPY_FLOAT32);
 
 // Go through and fill in the values for each row at a time...
  int i, j;
  for (j=0; j<exemplars; j++)
  {
   int * count  = (int*)PyArray_GETPTR1(count_arr, j);
   float * mean = (float*)PyArray_GETPTR1(mean_arr, j);
   float * var  = (float*)PyArray_GETPTR1(var_arr, j);
   
   *count = 0;
   *mean = 0.0;
   *var = 0.0;
    
   for (i=0; i<trees; i++)
   {
    Gaussian * targ = (Gaussian*)sums[j*trees + i];
    if (magic!=NULL) targ = (Gaussian*)magic(targ, extra);
    
    int new_count = *count + targ->count;
    float delta = targ->mean - (*mean);
    float offset = delta * targ->count / (float)new_count;
    *mean += offset;
    *var += (targ->var * targ->count) + offset * (*count) * delta;
    *count = new_count;
   }
   
   if (*count!=0) *var /= *count;
  }  
 
 // Build and return the dictionary of arrays...
  return Py_BuildValue("{sNsNsN}", "count", count_arr, "mean", mean_arr, "var", var_arr);
}

static size_t Gaussian_size(Summary self)
{
 return sizeof(Gaussian);
}

static PyObject * Gaussian_string(Summary self)
{
 Gaussian * this = (Gaussian*)self;
 
 char * ms = PyOS_double_to_string(this->mean, 'f', 3, 0, NULL);
 char * vs = PyOS_double_to_string(this->var, 'f', 3, 0, NULL);
 
 PyObject * ret = PyString_FromFormat("gaussian(%s,%s|%i)", ms, vs, this->count);
 
 PyMem_Free(ms);
 PyMem_Free(vs);
 
 return ret;
}


const SummaryType GaussianSummary =
{
 'G',
 "Gaussian",
 "Expects continuous valued values, which it models with a Gaussian distribution. For output it dumps a dictionary - indexed by 'count' for the number of samples that went into the calculation, 'mean' for the mean and 'var' for the variance. For a single sample these will go to standard python floats, for an array evaluation to numpy arrays. When returning errors it returns the absolute difference between the mean and actual value.",
 Gaussian_init_size,
 Gaussian_init,
 Gaussian_error,
 Gaussian_merge_py,
 Gaussian_merge_many_py,
 Gaussian_size,
 Gaussian_string,
};




// The bivariate Gaussian for real features - uses the given feature index as well as the next feature to construct a bivariate Gaussian...
typedef struct BiGaussian BiGaussian;

struct BiGaussian
{
 int count; // # Of samples that went into the distribution - has its uses.
 float mean[2];
 float var[2];
 float covar;
};


static size_t BiGaussian_init_size(DataMatrix * dm, IndexView * view, int feature)
{
 return sizeof(BiGaussian);
}

static void BiGaussian_init(Summary self, DataMatrix * dm, IndexView * view, int feature)
{
 BiGaussian * this = (BiGaussian*)self;
 
 int i, k;
 
 this->count = 0;
 for (k=0; k<2; k++)
 {
  this->mean[k] = 0.0;
  this->var[k] = 0.0;
 }
 this->covar = 0.0;
 
 for (i=0; i<view->size; i++)
 {
  int exemplar = view->vals[i];
  float value[2];
  
  value[0] = DataMatrix_GetContinuous(dm, exemplar, feature);
  value[1] = DataMatrix_GetContinuous(dm, exemplar, feature+1);
  
  this->count += 1;
  float delta[2];
  
  for (k=0; k<2; k++)
  {
   delta[k] = value[k] - this->mean[k];
   this->mean[k] += delta[k] / this->count;
   this->var[k] += delta[k] * (value[k] - this->mean[k]);
  }
  
  this->covar += delta[0] * (value[1] - this->mean[1]);
 }
 
 if (this->count!=0)
 {
  for (k=0; k<2; k++)
  {
   this->var[k] /= this->count;
  }
  this->covar /= this->count;
 }
}

static float BiGaussian_error(Summary self, DataMatrix * dm, IndexView * view, int feature)
{
 BiGaussian * this = (BiGaussian*)self;
 
 float ret = 0.0;
 
 int i;
 for (i=0; i<view->size; i++)
 {
  int ei = view->vals[i];
  float v1 = DataMatrix_GetContinuous(dm, ei, feature);
  float v2 = DataMatrix_GetContinuous(dm, ei, feature+1);
  
  float d1 = v1 - this->mean[0];
  float d2 = v2 - this->mean[1];
  ret += sqrt(d1*d1 + d2*d2);
 }
 
 return ret;
}


static PyObject * BiGaussian_merge_py(int trees, Summary * sums, SummaryMagic magic, int extra)
{
 int i, k;
 
 // Create the output variables...
  npy_intp dims[2] = {2, 2};
  
  int count = 0;
  PyArrayObject * mean_arr = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_FLOAT32);
  PyArrayObject * covar_arr = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_FLOAT32);
  
  float * mean[2];
  float * var[2];
  float * covar;
  
  for (k=0; k<2; k++)
  {
   mean[k] = (float*)PyArray_GETPTR1(mean_arr, k);
   var[k] = (float*)PyArray_GETPTR2(covar_arr, k, k);
   
   *mean[k] = 0.0;
   *var[k] = 0.0;
  }
  covar = (float*)PyArray_GETPTR2(covar_arr, 0, 1);
  *covar = 0.0;
 
 
 // Combine from all trees...
  for (i=0; i<trees; i++)
  {
   BiGaussian * targ = (BiGaussian*)sums[i];
   if (magic!=NULL) targ = (BiGaussian*)magic(targ, extra);
   
   int new_count = count + targ->count;
   float delta[2];
   
   for (k=0; k<2; k++)
   {
    delta[k] = targ->mean[k] - (*mean[k]);
    float offset = delta[k] * targ->count / (float)new_count;
    *mean[k] += offset;
    *var[k] += (targ->var[k] * targ->count) + offset * count * delta[k];
   }
   
   *covar += (targ->covar * targ->count) + delta[0] * delta[1] * count * targ->count / (float)new_count;
   
   count = new_count;
  }
  
  if (count!=0)
  {
   *var[0] /= count;
   *var[1] /= count;
   *covar /= count;
  }
  *(float*)PyArray_GETPTR2(covar_arr, 1, 0) = *covar;
  
 // Build and return a dictionary containing the required values...
  return Py_BuildValue("{sisNsN}", "count", count, "mean", mean_arr, "covar", covar_arr);
}

static PyObject * BiGaussian_merge_many_py(int exemplars, int trees, Summary * sums, SummaryMagic magic, int extra)
{
 int i, j, k;
 
 // Create the output variables...
  npy_intp dims[3] = {exemplars, 2, 2};
  
  PyArrayObject * count_arr = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_INT32);
  PyArrayObject * mean_arr = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_FLOAT32);
  PyArrayObject * covar_arr = (PyArrayObject*)PyArray_SimpleNew(3, dims, NPY_FLOAT32);
  
 // Loop and process each exemplar in turn... 
  for (j=0; j<exemplars; j++)
  {
   // Get pointers to all the output variables...
    int * count = (int*)PyArray_GETPTR1(count_arr, j);
    float * mean[2];
    float * var[2];
    float * covar;
  
    for (k=0; k<2; k++)
    {
     mean[k] = (float*)PyArray_GETPTR2(mean_arr, j, k);
     var[k] = (float*)PyArray_GETPTR3(covar_arr, j, k, k);
   
     *mean[k] = 0.0;
     *var[k] = 0.0;
    }
    covar = (float*)PyArray_GETPTR3(covar_arr, j, 0, 1);
    *covar = 0.0;
 
   // Combine from all trees...
    for (i=0; i<trees; i++)
    {
     BiGaussian * targ = (BiGaussian*)sums[j*trees+i];
     if (magic!=NULL) targ = (BiGaussian*)magic(targ, extra);
   
     int new_count = *count + targ->count;
     float delta[2];
   
     for (k=0; k<2; k++)
     {
      delta[k] = targ->mean[k] - (*mean[k]);
      float offset = delta[k] * targ->count / (float)new_count;
      *mean[k] += offset;
      *var[k] += (targ->var[k] * targ->count) + offset * (*count) * delta[k];
     }
   
     *covar += (targ->covar * targ->count) + delta[0] * delta[1] * (*count) * targ->count / (float)new_count;
   
     *count = new_count;
    }
  
    if (*count!=0)
    {
     *var[0] /= *count;
     *var[1] /= *count;
     *covar /= *count;
    }
    *(float*)PyArray_GETPTR3(covar_arr, j, 1, 0) = *covar;
  }
  
 // Build and return a dictionary containing the required values...
  return Py_BuildValue("{sNsNsN}", "count", count_arr, "mean", mean_arr, "covar", covar_arr);
}

static size_t BiGaussian_size(Summary self)
{
 return sizeof(BiGaussian);  
}

static PyObject * BiGaussian_string(Summary self)
{
 BiGaussian * this = (BiGaussian*)self;
 
 char * m0s = PyOS_double_to_string(this->mean[0], 'f', 3, 0, NULL);
 char * m1s = PyOS_double_to_string(this->mean[1], 'f', 3, 0, NULL);
 
 char * v0s = PyOS_double_to_string(this->var[0], 'f', 3, 0, NULL);
 char * v1s = PyOS_double_to_string(this->var[1], 'f', 3, 0, NULL);
 
 char * cvs = PyOS_double_to_string(this->covar, 'f', 3, 0, NULL);
 
 PyObject * ret = PyString_FromFormat("gaussian([%s,%s],[[%s,%s],[%s,%s]]|%i)", m0s, m1s, v0s, cvs, cvs, v1s, this->count);
 
 PyMem_Free(m0s);
 PyMem_Free(m1s);
 PyMem_Free(v0s);
 PyMem_Free(v1s);
 PyMem_Free(cvs);
 
 return ret;
}


const SummaryType BiGaussianSummary =
{
 'B',
 "BiGaussian",
 "A bivariate verison of Gaussian - uses the given feature index and the next one as well. Same output format-ish, except you get a length 2 array for mean and a 2x2 array indexed by 'covar' intead of the var entry, with one variable, and those with the extra dimension for the array version. Error is the Euclidean distance from the mean.",
 BiGaussian_init_size,
 BiGaussian_init,
 BiGaussian_error,
 BiGaussian_merge_py,
 BiGaussian_merge_many_py,
 BiGaussian_size,
 BiGaussian_string,
};



// List of summary types that the system knows about...
const SummaryType * ListSummary[] =
{
 &NothingSummary,
 &CategoricalSummary,
 &GaussianSummary,
 &BiGaussianSummary,
 NULL
};

const SummaryType * CodeSummary[256];



// Implimentation of summary set...
int SummarySet_validate(DataMatrix * dm, const char * codes)
{
 if (codes!=NULL)
 {
  // Validate length matches...
   if (strlen(codes)!=dm->features)
   {
    PyErr_SetString(PyExc_ValueError, "Summary codes do not match datamatrix feature count");
    return 0; 
   }
   
  // Validate all codes go to existing Summary types...
   int i;
   for (i=0; i<dm->features; i++)
   {
    if (CodeSummary[(unsigned char)codes[i]]==NULL) 
    {
     PyErr_SetString(PyExc_TypeError, "Unrecognised summary type code.");
     return 0; 
    }
   }
 }
 
 // No error has been raised - return success...
  return 1;
}

size_t SummarySet_init_size(DataMatrix * dm, IndexView * view, const char * codes)
{
 size_t ret = sizeof(SummarySet);
 
 // Add in the feature offsets...
  ret += dm->features * sizeof(int);
  
 // Add in the codes...
  ret += sizeof(int) * ((dm->features / sizeof(int)) + (((dm->features%sizeof(int))==0)?0:1));
 
 // Add in the Summary object sizes...
  int i;
  for (i=0; i<dm->features; i++)
  {
   if (codes)
   {
    ret += Summary_init_size(codes[i], dm, view, i);
   }
   else
   {
    if (DataMatrix_Type(dm, i)==DISCRETE)
    {
     ret += CategoricalSummary.init_size(dm, view, i);
    }
    else
    {
     ret += GaussianSummary.init_size(dm, view, i);
    }
   }
  }
 
 // Return the final size...
  return ret;
}

inline char * CodePtr(SummarySet * this)
{
 return (char*)this + sizeof(SummarySet) + this->features * sizeof(int); 
}

inline Summary SummaryPtr(SummarySet * this, int i)
{
 return (char*)this + this->offset[i]; 
}

void SummarySet_init(SummarySet * this, DataMatrix * dm, IndexView * view, const char * codes)
{
 int i; 
 
 // Setup the basic variables...
  this->size = sizeof(SummarySet) + dm->features * sizeof(int); // Grows as we go.
  this->features = dm->features;
 
 // Find the location of the code book and drop it in...
  char * code = CodePtr(this);
  if (codes!=NULL)
  {
   for (i=0; i<dm->features; i++)
   {
    code[i] = codes[i]; 
   }
  }
  else
  {
   for (i=0; i<dm->features; i++)
   {
    if (DataMatrix_Type(dm, i)==DISCRETE)
    {
     code[i] = CategoricalSummary.code;
    }
    else
    {
     code[i] = GaussianSummary.code; 
    }
   }
  }
  
  this->size += sizeof(int) * ((dm->features / sizeof(int)) + (((dm->features%sizeof(int))==0)?0:1));
  
 // Create the summary object for each feature in the DataMatrix...
  for (i=0; i<dm->features; i++)
  {
   this->offset[i] = this->size;
   Summary target = SummaryPtr(this, i);
   
   Summary_init(code[i], target, dm, view, i);
   
   this->size += Summary_init_size(code[i], dm, view, i);
  }
}

void SummarySet_error(SummarySet * this, DataMatrix * dm, IndexView * view, float * out)
{
 char * code = CodePtr(this);
 
 int i;
 for (i=0; i<this->features; i++)
 {
  out[i] += Summary_error(code[i], SummaryPtr(this, i), dm, view, i); 
 }
}

static Summary SummarySet_magic(void * self, int i)
{
 SummarySet * this = (SummarySet*)self;
 return (char*)this + this->offset[i];
}

PyObject * SummarySet_merge_py(int trees, SummarySet ** sum_sets)
{
 char * code = CodePtr(sum_sets[0]);
 
 // Create the return tuple...
  int feats = sum_sets[0]->features;
  PyObject * ret = PyTuple_New(feats);
 
 // Iterate and fill the tuple in...
  int i;
  for (i=0; i<feats; i++)
  {
   PyObject * obj = Summary_merge_py(code[i], trees, (Summary*)sum_sets, SummarySet_magic, i);
   PyTuple_SetItem(ret, i, obj);
  }
  
 // Return the tuple...
  return ret;
}

PyObject * SummarySet_merge_many_py(int exemplars, int trees, SummarySet ** sum_sets)
{
 char * code = CodePtr(sum_sets[0]);
  
 // Create the return tuple...
  int feats = sum_sets[0]->features;
  PyObject * ret = PyTuple_New(feats);
 
 // Iterate and fill the tuple in...
  int i;
  for (i=0; i<feats; i++)
  {
   PyObject * obj = Summary_merge_many_py(code[i], exemplars, trees, (Summary*)sum_sets, SummarySet_magic, i);
   PyTuple_SetItem(ret, i, obj);
  }
  
 // Return the tuple...
  return ret;
}

size_t SummarySet_size(SummarySet * this)
{
 return this->size; 
}

PyObject * SummarySet_string(SummarySet * this)
{
 char * code = CodePtr(this);
 int i;
 
 PyObject * ret = PyString_FromString("[");
 for (i=0; i<this->features; i++)
 {
  if (i!=0) PyString_ConcatAndDel(&ret, PyString_FromString(","));
  PyString_ConcatAndDel(&ret, Summary_string(code[i], SummaryPtr(this, i)));
 }
 PyString_ConcatAndDel(&ret, PyString_FromString("]"));
 
 return ret;
}



void Setup_Summary(void)
{
 import_array();  
}
