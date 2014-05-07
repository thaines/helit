// Copyright 2014 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



#include "summary.h"

#include "data_matrix.h"
#include "index_set.h"



// The helper methods for calling the methods of arbitrary summary objects...
Summary Summary_new(char code, DataMatrix * dm, IndexView * view, int feature)
{
 int i = 0;
 while(ListSummary[i]!=NULL)
 {
  if (ListSummary[i]->code==code)
  {
   return ListSummary[i]->init(dm, view, feature);
  }
  ++i;
 }
 
 PyErr_SetString(PyExc_TypeError, "Unrecognised summary type code."); 
 return NULL;
}

void Summary_delete(Summary this)
{
 const SummaryType * type = *(const SummaryType**)this;
 type->deinit(this);
}

const SummaryType * Summary_type(Summary this)
{
 return *(const SummaryType**)this;
}

PyObject * Summary_merge_py(int trees, Summary * sums, int offset)
{
 Summary * first = (Summary)((char*)sums[0] + offset);
 const SummaryType * type = *(const SummaryType**)first;
 return type->merge_py(trees, sums, offset);
}

PyObject * Summary_merge_many_py(int exemplars, int trees, Summary * sums, int offset)
{
 Summary * first = (Summary)((char*)sums[0] + offset);
 const SummaryType * type = *(const SummaryType**)first;
 return type->merge_many_py(exemplars, trees, sums, offset);
}

Summary Summary_from_bytes(void * in, size_t * ate)
{
 int i = 0;
 while(ListSummary[i]!=NULL)
 {
  if (ListSummary[i]->code==(*(char*)in))
  {
   Summary ret = ListSummary[i]->from_bytes((char*)in + 1, ate);
   if ((ret!=NULL)&&(ate!=NULL)) *ate += sizeof(char);
   return ret;
  }
  ++i;
 }
 
 PyErr_SetString(PyExc_ValueError, "Unrecognised summary type code in byte stream."); 
 return NULL;
}

size_t Summary_size(Summary this)
{
 const SummaryType * type = *(const SummaryType**)this;
 return type->size(this) + sizeof(char); // extra to store type code.
}

void Summary_to_bytes(Summary this, void * out)
{
 const SummaryType * type = *(const SummaryType**)this;
 *(char*)out = type->code;
 type->to_bytes(this, (char*)out + 1);
}



// The nothing summary type - I feel as empty writting this as I am sure you do reading it...
typedef struct Nothing Nothing;

struct Nothing
{
 const SummaryType * type;
};


static Summary Nothing_new(DataMatrix * dm, IndexView * view, int feature)
{
 Nothing * this = (Nothing*)malloc(sizeof(Nothing));
 this->type = &NothingSummary;
 return this;
}

static void Nothing_delete(Summary this)
{
 free(this); 
}

static PyObject * Nothing_merge_py(int trees, Summary * sums, int offset)
{
 Py_INCREF(Py_None);
 return Py_None;
}

static PyObject * Nothing_merge_many_py(int exemplars, int trees, Summary * sums, int offset)
{
 Py_INCREF(Py_None);
 return Py_None;
}

static Summary Nothing_from_bytes(void * in, size_t * ate)
{
 Nothing * this = (Nothing*)malloc(sizeof(Nothing));
 this->type = &NothingSummary;
 if (ate!=NULL) *ate = 0;
 return this;
}

static size_t Nothing_size(Summary this)
{
 return 0;  
}

static void Nothing_to_bytes(Summary this, void * out)
{
 // No-op 
}


const SummaryType NothingSummary =
{
 'N',
 "Nothing",
 "A summary type that does nothing - does not in any way summarise the feature index it is assigned to. For if you either have a multi-index summary type on an earlier feature, and hence don't need to summarise this feature index twice, or have some excess feature in your data structure and just want to ignore it.",
 Nothing_new,
 Nothing_delete,
 Nothing_merge_py,
 Nothing_merge_many_py,
 Nothing_from_bytes,
 Nothing_size,
 Nothing_to_bytes,
};



// The categorical summary type - a categorical distribution built from a discrete feature. Note that it ignores negative feature values, making that equivalent to an unknown value (Similarly, if the user sets a low max value for the datamatrix then high values will be ignored.), and assumes they are 0 based so it can use a simple array...
typedef struct Categorical Categorical;

struct Categorical
{
 const SummaryType * type;
 
 int count; // # Of samples that went into the distribution - has its uses.
 int cats; // Number of categories.
 float prob[0]; // Probability of each category.
};


static Summary Categorical_new(DataMatrix * dm, IndexView * view, int feature)
{
 int cats = DataMatrix_Max(dm, feature) + 1;
 
 Categorical * this = (Categorical*)malloc(sizeof(Categorical) + cats * sizeof(float));
 this->type = &CategoricalSummary;
 
 this->count = 0;
 this->cats = cats;
 
 int i;
 for (i=0; i<this->cats; i++)
 {
  this->prob[i] = 0.0;
 }
   
 for (i=0; i<view->size; i++)
 {
  int exemplar = view->vals[i];
  int value = DataMatrix_GetDiscrete(dm, exemplar, feature);
  
  if ((value>=0)&&(value<cats))
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
 
 return this;
}

static void Categorical_delete(Summary this)
{
 free(this); 
}

static PyObject * Categorical_merge_py(int trees, Summary * sums, int offset)
{
 // Setup the output...
  int count = 0;
  npy_intp size = ((Categorical*)sums[0])->cats;
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
   Categorical * targ = (Categorical*)((char*)sums[j] + offset);
   
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

static PyObject * Categorical_merge_many_py(int exemplars, int trees, Summary * sums, int offset)
{
 // Setup the output...
  npy_intp size[2] = {exemplars, ((Categorical*)sums[0])->cats};
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
    Categorical * targ = (Categorical*)((char*)sums[k*trees + j] + offset);
   
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

static Summary Categorical_from_bytes(void * in, size_t * ate)
{
 int cats = ((int*)in)[1];
  
 Categorical * this = (Categorical*)malloc(sizeof(Categorical) + cats * sizeof(float));
 this->type = &CategoricalSummary;
 
 this->count = ((int*)in)[1];
 this->cats = cats;
 
 int i;
 for (i=0; i<cats; i++)
 {
  this->prob[i] = ((int*)in+2)[i];
 }
 
 if (ate!=NULL) *ate = 2*sizeof(int) + cats*sizeof(float);
 return this;
}

static size_t Categorical_size(Summary self)
{
 Categorical * this = (Categorical*)self;
 return 2*sizeof(int) + this->cats*sizeof(float);  
}

static void Categorical_to_bytes(Summary self, void * out)
{
 Categorical * this = (Categorical*)self;
 
 ((int*)out)[0] = this->count;
 ((int*)out)[1] = this->cats;
 
 int i;
 for (i=0; i<this->cats; i++)
 {
  ((float*)((int*)out + 2))[i] = this->prob[i];
 }
}


const SummaryType CategoricalSummary =
{
 'C',
 "Categorical",
 "A standard categorical distribution for discrete features. The indices are taken to go from 1 to the maximum given by the datamatrix, inclusive - any value outside this is ignored, effectivly being treated as unknown. Output when converted to a python object is a dictionary - the key 'count' gets the number of samples that went into the distribution, whilst 'prob' gets an array, indexed by category, of the probabilities of each. For the array case the count gets a 1D array and cat becomes 2D, indexed [exemplar, cat].",
 Categorical_new,
 Categorical_delete,
 Categorical_merge_py,
 Categorical_merge_many_py,
 Categorical_from_bytes,
 Categorical_size,
 Categorical_to_bytes,
};



// The univariate Gaussian for real features - exactly what you would expect...
typedef struct Gaussian Gaussian;

struct Gaussian
{
 const SummaryType * type;
 
 int count; // # Of samples that went into the distribution - has its uses.
 float mean;
 float var;
};


static Summary Gaussian_new(DataMatrix * dm, IndexView * view, int feature)
{
 Gaussian * this = (Gaussian*)malloc(sizeof(Gaussian));
 this->type = &GaussianSummary;
 
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
 
 return this;
}

static void Gaussian_delete(Summary this)
{
 free(this); 
}

static PyObject * Gaussian_merge_py(int trees, Summary * sums, int offset)
{
 // Combine from all trees...
  int count = 0;
  float mean = 0.0;
  float var = 0.0;
  
  int i;
  for (i=0; i<trees; i++)
  {
   Gaussian * targ = (Gaussian*)((char*)sums[i] + offset);
   
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

static PyObject * Gaussian_merge_many_py(int exemplars, int trees, Summary * sums, int offset)
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
    Gaussian * targ = (Gaussian*)((char*)sums[j*trees + i] + offset);
    
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

static Summary Gaussian_from_bytes(void * in, size_t * ate)
{
 Gaussian * this = (Gaussian*)malloc(sizeof(Gaussian));
 this->type = &GaussianSummary;
 
 this->count = *(int*)in;
 this->mean = ((float*)((int*)in + 1))[0];
 this->var = ((float*)((int*)in + 1))[1];
 
 if (ate!=NULL) *ate = sizeof(int) + 2*sizeof(float);
 return this;
}

static size_t Gaussian_size(Summary self)
{
 return sizeof(int) + 2*sizeof(float);  
}

static void Gaussian_to_bytes(Summary self, void * out)
{
 Gaussian * this = (Gaussian*)self;
 *(int*)out = this->count;
 ((float*)((int*)out + 1))[0] = this->mean;
 ((float*)((int*)out + 1))[1] = this->var;
}


const SummaryType GaussianSummary =
{
 'G',
 "Gaussian",
 "Expects continuous valued values, which it models with a Gaussian distribution. For output it dumps a dictionary - indexed by 'count' for the number of samples that went into the calculation, 'mean' for the mean and 'var' for the variance. For a single sample these will go to standard python floats, for an array evaluation to numpy arrays.",
 Gaussian_new,
 Gaussian_delete,
 Gaussian_merge_py,
 Gaussian_merge_many_py,
 Gaussian_from_bytes,
 Gaussian_size,
 Gaussian_to_bytes,
};




// The bivariate Gaussian for real features - uses the given feature index as well as the next feature to construct a bivariate Gaussian...
typedef struct BiGaussian BiGaussian;

struct BiGaussian
{
 const SummaryType * type;
 
 int count; // # Of samples that went into the distribution - has its uses.
 float mean[2];
 float var[2];
 float covar;
};


static Summary BiGaussian_new(DataMatrix * dm, IndexView * view, int feature)
{
 BiGaussian * this = (BiGaussian*)malloc(sizeof(BiGaussian));
 this->type = &BiGaussianSummary;
 
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
 
 return this;
}

static void BiGaussian_delete(Summary this)
{
 free(this); 
}

static PyObject * BiGaussian_merge_py(int trees, Summary * sums, int offset)
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
   BiGaussian * targ = (BiGaussian*)((char*)sums[i] + offset);
   
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

static PyObject * BiGaussian_merge_many_py(int exemplars, int trees, Summary * sums, int offset)
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
     BiGaussian * targ = (BiGaussian*)((char*)sums[j*trees+i] + offset);
   
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

static Summary BiGaussian_from_bytes(void * in, size_t * ate)
{
 BiGaussian * this = (BiGaussian*)malloc(sizeof(BiGaussian));
 this->type = &BiGaussianSummary;
 
 this->count = *(int*)in;
 this->mean[0] = ((float*)((int*)in + 1))[0];
 this->mean[1] = ((float*)((int*)in + 1))[1];
 this->var[0] = ((float*)((int*)in + 1))[2];
 this->var[1] = ((float*)((int*)in + 1))[3];
 this->covar = ((float*)((int*)in + 1))[4];
 
 if (ate!=NULL) *ate = sizeof(int) + 5*sizeof(float);
 return this;
}

static size_t BiGaussian_size(Summary self)
{
 return sizeof(int) + 5*sizeof(float);  
}

static void BiGaussian_to_bytes(Summary self, void * out)
{
 BiGaussian * this = (BiGaussian*)self;
 *(int*)out = this->count;
 ((float*)((int*)out + 1))[0] = this->mean[0];
 ((float*)((int*)out + 1))[1] = this->mean[1];
 ((float*)((int*)out + 1))[2] = this->var[0];
 ((float*)((int*)out + 1))[3] = this->var[1];
 ((float*)((int*)out + 1))[4] = this->covar;
}


const SummaryType BiGaussianSummary =
{
 'B',
 "BiGaussian",
 "A bivariate verison of Gaussian - uses the given feature index and the next one as well. Same output format-ish, except you get a length 2 array for mean and a 2x2 array indexed by 'covar' intead of the var entry, with one variable, and those with the extra dimension for the array version.",
 BiGaussian_new,
 BiGaussian_delete,
 BiGaussian_merge_py,
 BiGaussian_merge_many_py,
 BiGaussian_from_bytes,
 BiGaussian_size,
 BiGaussian_to_bytes,
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



// Implimentation of summary set...
SummarySet * SummarySet_new(DataMatrix * dm, IndexView * view, const char * codes)
{
 // Verify the codes are of the right length...
  if ((codes!=NULL)&&(strlen(codes)!=dm->features))
  {
   PyErr_SetString(PyExc_ValueError, "Summary codes do not match datamatrix feature count");
   return NULL; 
  }
  
 // Create the object...
  SummarySet * this = (SummarySet*)malloc(sizeof(SummarySet) + dm->features * sizeof(Summary));
  this->features = dm->features;
  
 // Initialise each of the entrys, roling back if an error occurs...
  int i;
  for (i=0; i<this->features; i++)
  {
   if (codes!=NULL)
   {
    this->feature[i] = Summary_new(codes[i], dm, view, i); 
   }
   else
   {
    if (DataMatrix_Type(dm, i)==DISCRETE)
    {
     this->feature[i] = CategoricalSummary.init(dm, view, i);
    }
    else
    {
     this->feature[i] = GaussianSummary.init(dm, view, i);
    }
   }
   
   if (this->feature[i]==NULL)
   {
    // Error - rollback and deallocate everything that worked!..
     i -= 1;
     for (; i>=0; i--)
     {
      Summary_delete(this->feature[i]);
     }
     free(this);
     
    return NULL; 
   }
  }
  
 // Return...
  return this;
}

void SummarySet_delete(SummarySet * this)
{
 int i;
 for (i=0; i<this->features; i++)
 {
  Summary_delete(this->feature[i]); 
 }
 free(this);
}

PyObject * SummarySet_merge_py(int trees, SummarySet ** sum_sets)
{
 // Create the return tuple...
  int feats = sum_sets[0]->features;
  PyObject * ret = PyTuple_New(feats);
 
 // Iterate and fill the tuple in...
  int i;
  for (i=0; i<feats; i++)
  {
   int offset = offsetof(SummarySet, feature) + i * sizeof(Summary);
   PyObject * obj = Summary_merge_py(trees, (Summary*)sum_sets, offset);
   PyTuple_SetItem(ret, i, obj);
  }
  
 // Return the tuple...
  return ret;
}

PyObject * SummarySet_merge_many_py(int exemplars, int trees, SummarySet ** sum_sets)
{
 // Create the return tuple...
  int feats = sum_sets[0]->features;
  PyObject * ret = PyTuple_New(feats);
 
 // Iterate and fill the tuple in...
  int i;
  for (i=0; i<feats; i++)
  {
   int offset = offsetof(SummarySet, feature) + i * sizeof(Summary);
   PyObject * obj = Summary_merge_many_py(exemplars, trees, (Summary*)sum_sets, offset);
   PyTuple_SetItem(ret, i, obj);
  }
  
 // Return the tuple...
  return ret;
}

SummarySet * SummarySet_from_bytes(void * in, size_t * ate)
{
 // Eat the length...
  int feats = *(int*)in;
  in = (char*)in + sizeof(int);
  int used = sizeof(int);
 
 // Create the return object...
  SummarySet * this = (SummarySet*)malloc(sizeof(SummarySet) + feats*sizeof(Summary));
  this->features = feats;
  
  int i;
  for (i=0; i<feats; i++)
  {
   size_t used;
   this->feature[i] = Summary_from_bytes(in, &used);
   in = (char*)in + used;
  }
  
 // Return it...
  if (ate!=NULL) *ate = used;
  return this;
}

size_t SummarySet_size(SummarySet * this)
{
 size_t ret = sizeof(int);
 
 int i;
 for (i=0; i<this->features; i++)
 {
  ret += Summary_size(this->feature[i]); 
 }
 
 return ret; 
}

void SummarySet_to_bytes(SummarySet * this, void * out)
{
 *(int*)out = this->features;
 out = (char*)out + sizeof(int);
 
 int i;
 for (i=0; i<this->features; i++)
 {
  Summary_to_bytes(this->feature[i], out);
  out = (char*)out + Summary_size(this->feature[i]);
 }
}



// Makes a warning go away...
void DoNotUse_summary_h(void)
{
 import_array();  
}
