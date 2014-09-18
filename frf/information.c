// Copyright 2014 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



#include "information.h"



// The helper methods for generic Info objects...
Info Info_new(char code, DataMatrix * dm, int feature)
{
 int i = 0;
 while (ListInfo[i]!=NULL)
 {
  if (ListInfo[i]->code==code)
  {
   return ListInfo[i]->init(dm, feature);
  }
  i += 1;
 }
 
 PyErr_SetString(PyExc_TypeError, "Unrecognised information type code.");
 return NULL;
}

void Info_reset(Info this)
{
 const InfoType * type = *(const InfoType**)this;
 type->reset(this);
}

void Info_delete(Info this)
{
 const InfoType * type = *(const InfoType**)this;
 type->deinit(this);
}

void Info_add(Info this, int exemplar)
{
 const InfoType * type = *(const InfoType**)this;
 type->add(this, exemplar);
}

void Info_remove(Info this, int exemplar)
{
 const InfoType * type = *(const InfoType**)this;
 type->remove(this, exemplar);
}

float Info_count(Info this)
{
 const InfoType * type = *(const InfoType**)this;
 return type->count(this);
}

float Info_entropy(Info this)
{
 const InfoType * type = *(const InfoType**)this;
 return type->entropy(this);
}



// The nothing information type - still records count as some optimisers could get irrate otherwise...
typedef struct Nothing Nothing;

struct Nothing
{
 const InfoType * type;
 
 DataMatrix * dm;
 float count;
};


static Info Nothing_new(DataMatrix * dm, int feature)
{
 Nothing * this = (Nothing*)malloc(sizeof(Nothing));
 this->type = &NothingInfo;
 
 this->dm = dm;
 this->count = 0.0;
 
 return this;  
}

static void Nothing_reset(Info self)
{
 Nothing * this = (Nothing*)self;
 this->count = 0.0;
}

static void Nothing_delete(Info this)
{
 free(this);
}

static void Nothing_add(Info self, int exemplar)
{
 Nothing * this = (Nothing*)self;
 this->count += DataMatrix_GetWeight(this->dm, exemplar);
}

static void Nothing_remove(Info self, int exemplar)
{
 Nothing * this = (Nothing*)self;
 this->count -= DataMatrix_GetWeight(this->dm, exemplar);
}

static float Nothing_count(Info self)
{
 Nothing * this = (Nothing*)self;
 return this->count;
}

static float Nothing_entropy(Info this)
{
 return 0.0; 
}


const InfoType NothingInfo =
{
 'N',
 "Nothing",
 "Does nothing - always returns an entropy of zero, to say we are not encoding this information. Can be used to hide an output channel, either because you don't want it included or another channel is already using it.",
 Nothing_new,
 Nothing_reset,
 Nothing_delete,
 Nothing_add,
 Nothing_remove,
 Nothing_count,
 Nothing_entropy,
};



// The categorical information type...
typedef struct CategoricalInner CategoricalInner;
typedef struct Categorical Categorical;

struct CategoricalInner
{
 float count;
 float log_count; // Only valid when count!=0.
};

struct Categorical
{
 const InfoType * type;
 
 DataMatrix * dm;
 int feature;
 
 float exemplars; // Number of exemplars in the system.
 float total; // Sum if you add up count - can be less than exemplars due to unknown values.
 
 int cats;
 CategoricalInner cat[0];
};


static Info Categorical_new(DataMatrix * dm, int feature)
{
 int cats = DataMatrix_Max(dm, feature) + 1;
 Categorical * this = (Categorical*)malloc(sizeof(Categorical) + cats * sizeof(CategoricalInner));
 this->type = &CategoricalInfo;
 
 this->dm = dm;
 this->feature = feature;
 
 this->exemplars = 0.0;
 this->total = 0.0;
 
 this->cats = cats;
 int i;
 for (i=0; i<cats; i++)
 {
  this->cat[i].count = 0.0;
 }
 
 return this;  
}

static void Categorical_reset(Info self)
{
 Categorical * this = (Categorical*)self;
 
 this->exemplars = 0.0;
 this->total = 0.0;
 
 int i;
 for (i=0; i<this->cats; i++)
 {
  this->cat[i].count = 0.0; 
 }
}

static void Categorical_delete(Info this)
{
 free(this);
}

static void Categorical_add(Info self, int exemplar)
{
 Categorical * this = (Categorical*)self;
 float w = DataMatrix_GetWeight(this->dm, exemplar);
 
 this->exemplars += w;
 
 int val = DataMatrix_GetDiscrete(this->dm, exemplar, this->feature);
 if ((val>=0)&&(val<this->cats))
 {
  this->total += w;
  this->cat[val].count += w;
  if (this->cat[val].count>1e-6)
  {
   this->cat[val].log_count = log(this->cat[val].count);
  }
 }
}

static void Categorical_remove(Info self, int exemplar)
{
 Categorical * this = (Categorical*)self;
 float w = DataMatrix_GetWeight(this->dm, exemplar);
 
 this->exemplars -= w;
 
 int val = DataMatrix_GetDiscrete(this->dm, exemplar, this->feature);
 if ((val>=0)&&(val<this->cats))
 {
  this->total -= w;
  this->cat[val].count -= w;
  if (this->cat[val].count>1e-6)
  {
   this->cat[val].log_count = log(this->cat[val].count); 
  }
 }
}

static float Categorical_count(Info self)
{
 Categorical * this = (Categorical*)self;
 return this->exemplars;
}

static float Categorical_entropy(Info self)
{
 Categorical * this = (Categorical*)self;
 float ret = 0.0;
 
 float mult = 1.0 / (float)this->total;
 
 int i;
 for (i=0; i<this->cats; i++)
 {
  if (this->cat[i].count>1e-6)
  {
   ret -= mult * this->cat[i].count * this->cat[i].log_count;
  }
 }
 
 return ret + log(this->total);
}


const InfoType CategoricalInfo =
{
 'C',
 "Categorical",
 "Provides the entropy of a categorical distribution for the optimisation subsystem - the default for discrete distributions.",
 Categorical_new,
 Categorical_reset,
 Categorical_delete,
 Categorical_add,
 Categorical_remove,
 Categorical_count,
 Categorical_entropy,
};



// The Gaussian information type...
typedef struct Gaussian Gaussian;

struct Gaussian
{
 const InfoType * type;
 
 DataMatrix * dm;
 int feature;
 
 float exemplars; // Number of exemplars in the system.
 
 float mean;
 float scatter; // i.e. variance multiplied by exemplars.
};


static Info Gaussian_new(DataMatrix * dm, int feature)
{
 Gaussian * this = (Gaussian*)malloc(sizeof(Gaussian));
 this->type = &GaussianInfo;
 
 this->dm = dm;
 this->feature = feature;
 
 this->exemplars = 0.0;
 
 this->mean = 0.0;
 this->scatter = 0.0;
 
 return this;  
}

static void Gaussian_reset(Info self)
{
 Gaussian * this = (Gaussian*)self;
 
 this->exemplars = 0.0;
 
 this->mean = 0.0;
 this->scatter = 0.0;
}

static void Gaussian_delete(Info this)
{
 free(this);
}

static void Gaussian_add(Info self, int exemplar)
{
 Gaussian * this = (Gaussian*)self;
 float val = DataMatrix_GetContinuous(this->dm, exemplar, this->feature);
 float w = DataMatrix_GetWeight(this->dm, exemplar);
 
 float new_exemplars = this->exemplars + w;
 float delta = val - this->mean;
 float offset = (w * delta) / new_exemplars;
 
 this->mean += offset;
 this->scatter += this->exemplars * delta * offset;
 this->exemplars = new_exemplars;
}

static void Gaussian_remove(Info self, int exemplar)
{
 Gaussian * this = (Gaussian*)self;
 float val = DataMatrix_GetContinuous(this->dm, exemplar, this->feature);
 float w = DataMatrix_GetWeight(this->dm, exemplar);
 
 float new_exemplars = this->exemplars - w;
 float delta = val - this->mean;
 float offset = (w * delta) / new_exemplars;
 
 this->mean -= offset;
 this->scatter -= this->exemplars * delta * offset;
 this->exemplars = new_exemplars;
}

static float Gaussian_count(Info self)
{
 Gaussian * this = (Gaussian*)self;
 return this->exemplars; 
}

static float Gaussian_entropy(Info self)
{
 Gaussian * this = (Gaussian*)self;

 float var = this->scatter / this->exemplars;
 if (var<1e-6) var = 1e-6; // To avoid log(0) - bit of light regularisation basically.
 
 return 0.5 * log(2*M_PI*M_E*var);
}


const InfoType GaussianInfo =
{
 'G',
 "Gaussian",
 "Given the entropy of a Gaussian distribution over the data - default for continuous data.",
 Gaussian_new,
 Gaussian_reset,
 Gaussian_delete,
 Gaussian_add,
 Gaussian_remove,
 Gaussian_count,
 Gaussian_entropy,
};



// The bivariate-Gaussian information type...
typedef struct BiGaussian BiGaussian;

struct BiGaussian
{
 const InfoType * type;
 
 DataMatrix * dm;
 int feature;
 
 float exemplars; // Number of exemplars in the system.
 
 float mean[2];
 float scatter_var[2]; // i.e. variance multiplied by exemplars.
 float scatter_co; // i.e. the covariance multiplied by exemplars.
};


static Info BiGaussian_new(DataMatrix * dm, int feature)
{
 BiGaussian * this = (BiGaussian*)malloc(sizeof(BiGaussian));
 this->type = &BiGaussianInfo;
 
 this->dm = dm;
 this->feature = feature;
 
 this->exemplars = 0.0;
 
 int i;
 for (i=0; i<2; i++)
 {
  this->mean[i] = 0.0;
  this->scatter_var[i] = 0.0;
 }
 this->scatter_co = 0.0;
 
 return this;  
}

static void BiGaussian_reset(Info self)
{
 BiGaussian * this = (BiGaussian*)self;
 
 this->exemplars = 0.0;
 
 int i;
 for (i=0; i<2; i++)
 {
  this->mean[i] = 0.0;
  this->scatter_var[i] = 0.0;
 }
 this->scatter_co = 0.0;
}

static void BiGaussian_delete(Info this)
{
 free(this);
}

static void BiGaussian_add(Info self, int exemplar)
{
 BiGaussian * this = (BiGaussian*)self;
 
 float val[2];
 val[0] = DataMatrix_GetContinuous(this->dm, exemplar, this->feature);
 val[1] = DataMatrix_GetContinuous(this->dm, exemplar, this->feature+1);
 float w = DataMatrix_GetWeight(this->dm, exemplar);
 
 int i;
 float delta[2];
 float offset[2];
 
 float new_exemplars = this->exemplars + w;
 for (i=0; i<2; i++)
 {
  delta[i] = val[i] - this->mean[i];
  offset[i] = delta[i] * w / new_exemplars;
  
  this->mean[i] += offset[i];
  this->scatter_var[i] += this->exemplars * delta[i] * offset[i];
 }
 
 this->scatter_co += this->exemplars * delta[0] * offset[1];
 this->exemplars = new_exemplars;
}

static void BiGaussian_remove(Info self, int exemplar)
{
 BiGaussian * this = (BiGaussian*)self;
 
 float val[2];
 val[0] = DataMatrix_GetContinuous(this->dm, exemplar, this->feature);
 val[1] = DataMatrix_GetContinuous(this->dm, exemplar, this->feature+1);
 float w = DataMatrix_GetWeight(this->dm, exemplar);
 
 int i;
 float delta[2];
 float offset[2];
 
 float new_exemplars = this->exemplars - w;
 for (i=0; i<2; i++)
 {
  delta[i] = val[i] - this->mean[i];
  offset[i] = delta[i] * w / new_exemplars;
  
  this->mean[i] -= offset[i];
  this->scatter_var[i] -= this->exemplars * delta[i] * offset[i];
 }
 
 this->scatter_co -= this->exemplars * delta[0] * offset[1];
 this->exemplars = new_exemplars;
}

static float BiGaussian_count(Info self)
{
 BiGaussian * this = (BiGaussian*)self;
 return this->exemplars; 
}

static float BiGaussian_entropy(Info self)
{
 BiGaussian * this = (BiGaussian*)self;
 if (this->exemplars<1e-6) return 0.0;
 
 float covar = this->scatter_co / this->exemplars;
 float det = (this->scatter_var[0] / this->exemplars) * (this->scatter_var[1] / this->exemplars) - covar*covar;
 
 det *= 2 * M_PI * M_E;
 if (det<1e-6) det = 1e-6; // To avoid log zero.

 return 0.5 * log(det);
}


const InfoType BiGaussianInfo =
{
 'B',
 "BiGaussian",
 "Gaussian of two variables - first is the feature index its at, the second is the next one along - don't put in the last slot as that will go wrong.",
 BiGaussian_new,
 BiGaussian_reset,
 BiGaussian_delete,
 BiGaussian_add,
 BiGaussian_remove,
 BiGaussian_count,
 BiGaussian_entropy,
};



// The list of info types...
const InfoType * ListInfo[] =
{
 &NothingInfo,
 &CategoricalInfo,
 &GaussianInfo,
 &BiGaussianInfo,
 NULL
};



// Info set methods...
InfoSet * InfoSet_new(DataMatrix * dm, const char * codes, PyArrayObject * ratios)
{
 // Get the number of features, verify the inputs...
  int feats = dm->features;
  if ((codes!=NULL)&&(strlen(codes)!=feats))
  {
   PyErr_SetString(PyExc_ValueError, "Information measure codes wrong length for data matrix");
   return NULL; 
  }
  
  if (ratios!=NULL)
  {
   if (PyArray_NDIM(ratios)!=2)
   {
    PyErr_SetString(PyExc_TypeError, "Ratio matrix for information measure must be 2D, indexed [depth, feature]."); 
    return NULL;
   }
   
   if (PyArray_DIMS(ratios)[1]!=feats)
   {
    PyErr_SetString(PyExc_TypeError, "2nd dimension of ratio matrix must match number of features in data matrix."); 
    return NULL;
   }
  }
 
 // Create the object...
  InfoSet * this = (InfoSet*)malloc(sizeof(InfoSet) + feats * sizeof(InfoPair));
  this->ratios = ratios;
  Py_XINCREF(this->ratios);
  if (this->ratios!=NULL) this->rat_func = KindToContinuousFunc(PyArray_DESCR(this->ratios));
    
  this->features = feats;
  
 // Zero out the array of info objects (makes error handling easier...)...
  int i;
  for (i=0; i<feats; i++)
  {
   this->pair[i].pass = NULL;
   this->pair[i].fail = NULL;
  }
  
 // Fill in the array of Info objects - need to be ready to rollback on error...
  int error = 0;
  
  for (i=0; i<feats; i++)
  {
   if (codes!=NULL)
   {
    this->pair[i].pass = Info_new(codes[i], dm, i);
    if (this->pair[i].pass==NULL) {error = 1; break;}
    
    this->pair[i].fail = Info_new(codes[i], dm, i);
    if (this->pair[i].fail==NULL) {error = 1; break;}
   }
   else
   {
    if (DataMatrix_Type(dm, i)==DISCRETE)
    {
     this->pair[i].pass = CategoricalInfo.init(dm, i);
     this->pair[i].fail = CategoricalInfo.init(dm, i);
    }
    else
    {
     this->pair[i].pass = GaussianInfo.init(dm, i);
     this->pair[i].fail = GaussianInfo.init(dm, i);
    }
   }
  }
  
 // If there has been a problem clean up and error...
  if (error!=0)
  {
   for (i=0; i<feats; i++)
   {
    if (this->pair[i].pass!=NULL) Info_delete(this->pair[i].pass);
    if (this->pair[i].fail!=NULL) Info_delete(this->pair[i].fail);
   }
   
   Py_XDECREF(this->ratios);
   free(this);
   
   return NULL; 
  }
  
 // Return...
  return this;
}

void InfoSet_delete(InfoSet * this)
{
 int i;
 for (i=0; i<this->features; i++)
 {
  Info_delete(this->pair[i].pass);
  Info_delete(this->pair[i].fail);
 }
   
 Py_XDECREF(this->ratios);
 free(this); 
}

void InfoSet_reset(InfoSet * this)
{
 int i;
 for (i=0; i<this->features; i++)
 {
  Info_reset(this->pair[i].pass);
  Info_reset(this->pair[i].fail);
 }
}

void InfoSet_pass_add(InfoSet * this, int exemplar)
{
 int i;
 for (i=0; i<this->features; i++)
 {
  Info_add(this->pair[i].pass, exemplar);
 }
}

void InfoSet_pass_remove(InfoSet * this, int exemplar)
{
 int i;
 for (i=0; i<this->features; i++)
 {
  Info_remove(this->pair[i].pass, exemplar);
 }
}

void InfoSet_fail_add(InfoSet * this, int exemplar)
{
 int i;
 for (i=0; i<this->features; i++)
 {
  Info_add(this->pair[i].fail, exemplar);
 }
}

void InfoSet_fail_remove(InfoSet * this, int exemplar)
{
 int i;
 for (i=0; i<this->features; i++)
 {
  Info_remove(this->pair[i].fail, exemplar);
 }
}

float InfoSet_entropy(InfoSet * this, int depth)
{
 float ret = 0.0;
 
 int i;
 for (i=0; i<this->features; i++)
 {
  float ratio = 1.0;
  if (this->ratios!=NULL)
  {
   ratio = this->rat_func(PyArray_GETPTR2(this->ratios, depth % PyArray_DIMS(this->ratios)[0], i));
  }
  
  if (ratio>1e-6)
  {
   float fail_total = Info_count(this->pair[i].fail);
   float pass_total = Info_count(this->pair[i].pass);
   float total = fail_total + pass_total;
   if (total<1e-6) total = 1e-6;
   
   float weight = fail_total / total;
   
   float entropy = weight * Info_entropy(this->pair[i].fail) + (1.0 - weight) * Info_entropy(this->pair[i].pass);
   
   ret += ratio * entropy;
  }
 }
   
 return ret;
}

float InfoSet_view_entropy(InfoSet * this, IndexView * iv, int depth)
{
 // Reset the fail half...
  int i;
  for (i=0; i<this->features; i++)
  {
   Info_reset(this->pair[i].fail);
  }
  
 // Add everything to the fail half...
  int j;
  for (j=0; j<iv->size; j++)
  {
   for (i=0; i<this->features; i++)
   {
    Info_add(this->pair[i].fail, iv->vals[j]);
   }
  }
  
 // Calculate and return its entropy...
  float ret = 0.0;
  for (i=0; i<this->features; i++)
  {
   float ratio = 1.0;
   if (this->ratios!=NULL)
   {
    ratio = this->rat_func(PyArray_GETPTR2(this->ratios, depth % PyArray_DIMS(this->ratios)[0], i));
   }

   if (ratio>1e-6)
   {
    ret += ratio * Info_entropy(this->pair[i].fail);
   }
  }
   
 return ret;
}



void Setup_Information(void)
{
 import_array();  
}
