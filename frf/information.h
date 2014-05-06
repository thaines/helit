#ifndef INFORMATION_H
#define INFORMATION_H

// Copyright 2014 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



// Provides measures of the information of a set of exemplars, with add/remove capability and support for multiple output information measures...

#include <Python.h>
#include <structmember.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "data_matrix.h"



// Info type, that provides information on a single channel...
typedef void * Info;



// Function pointer types for the information object...

// New, delete and reset...
typedef Info (*InfoNew)(DataMatrix * dm, int feature);
typedef void (*InfoReset)(Info this);
typedef void (*InfoDelete)(Info this);

// Add and remove for exemplars in the data set...
typedef void (*InfoAdd)(Info this, int exemplar);
typedef void (*InfoRemove)(Info this, int exemplar);

// Return how many exemplars are in the object right now...
typedef int (*InfoCount)(Info this);

// Returns the entropy (in nats) of the current set of exemplars...
typedef float (*InfoEntropy)(Info this);



// Definition of type object (v-table) for Info objects...
typedef struct InfoType InfoType;

struct InfoType
{
 const char code; // So a string can specify the info types to use.
 const char * name;
 const char * description;
 
 InfoNew init;
 InfoReset reset;
 InfoDelete deinit;
 
 InfoAdd add;
 InfoRemove remove;
 
 InfoCount count;
 InfoEntropy entropy;
};



// Standard accessor methods for all info objects that assume a pointer to an InfoType is the first entry in the Info struct...
Info Info_new(char code, DataMatrix * dm, int feature);
void Info_reset(Info this);
void Info_delete(Info this);

void Info_add(Info this, int exemplar);
void Info_remove(Info this, int exemplar);

int Info_count(Info this);
float Info_entropy(Info this);



// Basic information types...

// Big fat NO-OP - always returns 0, mainly for use with BiGaussian on the second channel...
const InfoType NothingInfo; // N

// Entropy of a Categorical distribution over the data...
const InfoType CategoricalInfo; // C

// Entropy of a Gaussian distribution over the data...
const InfoType GaussianInfo; // G

// Entropy of a bivariate Gaussian distribution over the data...
const InfoType BiGaussianInfo; // B



// List of all known information types...
extern const InfoType * ListInfo[];



// Cute-lil struct for the below...
typedef struct InfoPair InfoPair;

struct InfoPair
{
 Info pass;
 Info fail;
};



// A set of info types, two for each feature in a target DataMatrix - allows configuration and fetching the entropy of all of them combined, noting that you can configure the ratios of feature entropy on a per depth basis, to optimise for different things at different depths...
typedef struct InfoSet InfoSet;

struct InfoSet
{
 int features;
 InfoPair pair[0];
};



// Creates an info set for a datamatrix. You can optionally provide codes to choose which to use for each - if NULL then it automatically uses Categorical for discrete channels and Gaussian for continuous. Finally, you can optionally provide a numpy array of weights indexed by [depth, feature] that re-weights the entropy values - depth is accessed modulus the size of the array, so it will wrap around if it gets too deep...
InfoSet * InfoSet_new(DataMatrix * dm, const char * codes, PyArrayObject * ratios);

// Clean up an info set...
void InfoSet_delete(InfoSet * this);

// Reset, add and remove...
void InfoSet_reset(InfoSet * this);
void InfoSet_pass_add(InfoSet * this, int exemplar);
void InfoSet_pass_remove(InfoSet * this, int exemplar);
void InfoSet_fail_add(InfoSet * this, int exemplar);
void InfoSet_fail_remove(InfoSet * this, int exemplar);

// Returns the entropy of the current split, weighted by the number of samples in each half
float InfoSet_entropy(InfoSet * this);



#endif
