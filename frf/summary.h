#ifndef SUMMARY_H
#define SUMMARY_H

// Copyright 2014 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



// Given a data set and index view this can summarise the statistics of the nodes within - it is this that provides the output from a random forest. Internal object only though - it converts the output to a list for external consumption...

#include <Python.h>
#include <structmember.h>

typedef struct DataMatrix DataMatrix;
typedef struct IndexView IndexView;



// Summary type, just a void pointer used for all of the summary types...
typedef void * Summary;



// Define an access function to be used if the Summary pointer is not really a Summary pointer below, to convert whatever it really is into one...
typedef Summary (*SummaryMagic)(void * ptr, int extra);



// The function pointer typedefs required by each summary object...

// Returns how many bytes are required by the Summary object if initialised with the given parameters...
typedef size_t (*SummaryInitSize)(DataMatrix * dm, IndexView * view, int feature);

// Creates a new Summary object of the given type, storing it in the provided memory block - requires a DataMatrix to summarise, an exemplar index view to tell it which exemplars to summarise and a feature index of which index to summarise...
typedef void (*SummaryInit)(Summary this, DataMatrix * dm, IndexView * view, int feature);

// Calculates the error of the given nodes reaching this summary, as some kind of floating point value summed for all entries...
typedef float (*SummaryError)(Summary this, DataMatrix * dm, IndexView * view, int feature);

// Converts a set of summaries (trees is the number) into a python object that the user can dance with; returns a new reference. Can in principal return NULL and raise an error. For combining the summaries from the leaves of multiple trees into a single entity for a user to play with. The function SummaryMagic, and its parameter extra, exist to make use from within a SummarySet efficient - a function that converts the passed in 'fake' Summary object array into real Summary objects...
typedef PyObject * (*SummaryMergePy)(int trees, Summary * sums, SummaryMagic magic, int extra);

// As above, but for multiple test exemplars, point being the Python summary can give a datamatrix-like response and be much more efficient this way. Outer is exemplars, inner is trees when going through the sums array...
typedef PyObject * (*SummaryMergeManyPy)(int exemplars, int trees, Summary * sums, SummaryMagic magic, int extra);

// Returns how many bytes the passed Summary object is in size...
typedef size_t (*SummarySize)(Summary this);



// The summary type - basically all the function pointers and documentation required to run a summary object...
typedef struct SummaryType SummaryType;

struct SummaryType
{
 const char code; // Used for specifying the summary types as a string.
 const char * name;
 const char * description;
 
 SummaryInitSize init_size;
 SummaryInit init;
 
 SummaryError error;
 
 SummaryMergePy merge_py;
 SummaryMergeManyPy merge_many_py;
 
 SummarySize size;
};



// Define a set of standard methods for arbitrary Summary objects - all assume the first entry in the Summary structure is a pointer to its SummaryType object - match with defined function pointers...
size_t Summary_init_size(char code, DataMatrix * dm, IndexView * view, int feature);
void Summary_init(char code, Summary this, DataMatrix * dm, IndexView * view, int feature);

float Summary_error(char code, Summary this, DataMatrix * dm, IndexView * view, int feature);

PyObject * Summary_merge_py(char code, int trees, Summary * sums, SummaryMagic magic, int extra);
PyObject * Summary_merge_many_py(char code, int exemplars, int trees, Summary * sums, SummaryMagic magic, int extra);

size_t Summary_size(char code, Summary this);



// The SummaryType objects provided by the system...
// Does nothing - mostly useful if using something like the BiGaussianType...
const SummaryType NothingSummary; // Code = N

// Default for dealing with discrete variables...
const SummaryType CategoricalSummary; // Code = C

// Default for dealing with continuous variables...
const SummaryType GaussianSummary; // Code = G

// Does a bivariate Gaussian, on the provided feature index and the following one (A type code vector with this as the last entry will cause a crash.)...
const SummaryType BiGaussianSummary; // Code = B



// List of all summary types known to the system - for automatic detection, and a code to Type lookup for speed...
extern const SummaryType * ListSummary[];
extern const SummaryType * CodeSummary[256];



// Because we deal in mulivariate output we need summaries to come as a set, indexed by output feature...
typedef struct SummarySet SummarySet;

struct SummarySet
{
 int size; // Size of this entire summary.
 int features; // Number of features.
 int offset[0]; // Offset from start of this struct to find each Summary.
 // Codes go here, as array of chars immediatly after last offset. Will be padded to multiple of sizeof(int)
};



// Validates the provided codes - returns non-zero on success, zero on error, in which case it will have set a Python error. Can be called on a null codes pointer without issue...
int SummarySet_validate(DataMatrix * dm, const char * codes);

// Returns how large the memory block for a SummarySet needs to be, so the user can allocate it. Provided codes must be valid...
size_t SummarySet_init_size(DataMatrix * dm, IndexView * view, const char * codes);

// Creates a SummarySet, using the type string - if the type string is null then it uses the default, where it uses a Categorical for discrete data and a Gaussian for continuous data. It also falls back to these when the string is too short...
void SummarySet_init(SummarySet * this, DataMatrix * dm, IndexView * view, const char * codes);

// Outputs the error of the summary set when applied to the given exemplars - used for calculating the OOB error - outputs a value for each feature, into an array of floats (length must be number of features), so the user can decide what they care about and weight them accordingly. It adds its value to whatever is already in the array...
void SummarySet_error(SummarySet * this, DataMatrix * dm, IndexView * view, float * out);

// Returns a new reference to a Python object that is returned to the user to summarise the many summary sets provided - a tuple indexed by feature, with the actual objects in the tuple defined by the actual summary types. This exists to be given the summary sets at the leaves of a forest that an exemplar falls into...
PyObject * SummarySet_merge_py(int trees, SummarySet ** sum_sets);

// As above, but for when we are processing an entire data matrix and hence have an exemplars x trees array of SummarySet pointers, indexed with exemplars in the outer loop, trees in the inner...
PyObject * SummarySet_merge_many_py(int exemplars, int trees, SummarySet ** sum_sets);

// Returns how many bytes the given SummarySet consumes...
size_t SummarySet_size(SummarySet * this);



// Setup this module - for internal use only...
void Setup_Summary(void);



#endif
