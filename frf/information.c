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

int Info_count(Info this)
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
 int count;
};


static Info Nothing_new(DataMatrix * dm, int feature)
{
 Nothing * this = (Nothing*)malloc(sizeof(Nothing));
 this->type = &NothingInfo;
 
 this->count = 0;
 return this;  
}

static void Nothing_reset(Info self)
{
 Nothing * this = (Nothing*)self;
 this->count = 0;
}

static void Nothing_delete(Info this)
{
 free(this);
}

static void Nothing_add(Info self, int exemplar)
{
 Nothing * this = (Nothing*)self;
 this->count += 1;
}

static void Nothing_remove(Info self, int exemplar)
{
 Nothing * this = (Nothing*)self;
 this->count -= 1;
}

static int Nothing_count(Info self)
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
 "Does nothing - always returns an entrop of zero, to say we are not encoding this information. Can be used to hide an output channel, either because you don't want it included or another channel is already using it.",
 Nothing_new,
 Nothing_reset,
 Nothing_delete,
 Nothing_add,
 Nothing_remove,
 Nothing_count,
 Nothing_entropy,
};



// The categorical information type...
typedef struct Categorical Categorical;

struct Categorical
{
 const InfoType * type;
 
 DataMatrix * dm;
 int feature;
 
 int exemplars; // Number of exemplars in the system.
 int total; // Sum if you add up count - can be less than exemplars due to unknown values.
 
 int cats;
 int count[0];
};


static Info Categorical_new(DataMatrix * dm, int feature)
{
 int cats = DataMatrix_Max(dm, feature) + 1;
 Categorical * this = (Categorical*)malloc(sizeof(Categorical) + cats * sizeof(int));
 this->type = &CategoricalInfo;
 
 this->dm = dm;
 this->feature = feature;
 
 this->exemplars = 0;
 this->total = 0;
 
 this->cats = cats;
 int i;
 for (i=0; i<cats; i++)
 {
  this->count[i] = 0;
 }
 
 return this;  
}

static void Categorical_reset(Info self)
{
 Categorical * this = (Categorical*)self;
 
 this->exemplars = 0;
 this->total = 0;
 
 int i;
 for (i=0; i<this->cats; i++)
 {
  this->count[i] = 0; 
 }
}

static void Categorical_delete(Info this)
{
 free(this);
}

static void Categorical_add(Info self, int exemplar)
{
 Categorical * this = (Categorical*)self;
 this->exemplars += 1;
 
 int val = DataMatrix_GetDiscrete(this->dm, exemplar, this->feature);
 if ((val>=0)&&(val<this->cats))
 {
  this->total += 1;
  this->count[val] += 1;
 }
}

static void Categorical_remove(Info self, int exemplar)
{
 Categorical * this = (Categorical*)self;
 this->exemplars -= 1;
 
 int val = DataMatrix_GetDiscrete(this->dm, exemplar, this->feature);
 if ((val>=0)&&(val<this->cats))
 {
  this->total -= 1;
  this->count[val] -= 1;
 }
}

static int Categorical_count(Info self)
{
 Categorical * this = (Categorical*)self;
 return this->exemplars; 
}

static float Categorical_entropy(Info self)
{
 Categorical * this = (Categorical*)self;
 float ret = 0.0;
 
 int i;
 for (i=0; i<this->cats; i++)
 {
  if (this->count[i]!=0)
  {
   float p = this->count[i] / (float)this->total;
   ret -= p * log(p);
  }
 }
 
 return ret;
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
 
 int exemplars; // Number of exemplars in the system.
 
 float mean;
 float scatter; // i.e. variance multiplied by exemplars.
};


static Info Gaussian_new(DataMatrix * dm, int feature)
{
 Gaussian * this = (Gaussian*)malloc(sizeof(Gaussian));
 this->type = &GaussianInfo;
 
 this->dm = dm;
 this->feature = feature;
 
 this->exemplars = 0;
 
 this->mean = 0.0;
 this->scatter = 0.0;
 
 return this;  
}

static void Gaussian_reset(Info self)
{
 Gaussian * this = (Gaussian*)self;
 
 this->exemplars = 0;
 
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
 
 this->exemplars += 1;
 float delta = val - this->mean;
 this->mean += delta / this->exemplars;
 this->scatter += delta * (val - this->mean);
}

static void Gaussian_remove(Info self, int exemplar)
{
 Gaussian * this = (Gaussian*)self;
 float val = DataMatrix_GetContinuous(this->dm, exemplar, this->feature);
 
 this->exemplars -= 1;
 float delta = val - this->mean;
 this->mean -= delta / this->exemplars;
 this->scatter -= delta * (val - this->mean);
}

static int Gaussian_count(Info self)
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

// ******************************



// The list of info types...
const InfoType * ListInfo[] =
{
 &NothingInfo,
 &CategoricalInfo,
 &GaussianInfo,
 //&BiGaussianInfo,
 NULL
};



// Info set methods...

// **********************************



// Makes a warning go away...
void DoNotUse_information_h(void)
{
 import_array();  
}
