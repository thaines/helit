#ifndef LEARNER_H
#define LEARNER_H

// Copyright 2014 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



// This provides tests, and code to optimise them based on information theory based measures of their value - its all fairly simple, though its designed to avoid allocating its own memory...

#include <stddef.h>

typedef struct DataMatrix DataMatrix;
typedef struct IndexView IndexView;
typedef struct InfoSet InfoSet;



// Define a learner type...
typedef void * Learner;



// Define the methods of a learner object...

// Creates a new learner for a given datamatrix/feature...
typedef Learner (*LearnerNew)(DataMatrix * dm, int feature);

// Retires a learner. In the Blade Runner sense of the word...
typedef void (*LearnerDelete)(Learner this);

// Optimises the learner for the given IndexView (Which its allowed to re-order!), and returns 1 if its generated a test, 0 otherwise. Improve is a hint that if it can't do better than improve it should just give up - could save computation if a learner is capable of using it. key is used if random number generation is needed - it should be incrimented if its used...
typedef int (*LearnerOptimise)(Learner this, InfoSet * info, IndexView * view, int depth, float improve, unsigned int key[4]);

// If the learner has just done a successful optimisation then this should return the entropy of it...
typedef float (*LearnerEntropy)(Learner this);

// If the learner has just done a successful optimisation then this should return how many bytes the test requires to be written into...
typedef size_t (*LearnerSize)(Learner this);

// If the learner has just done a successful optimisation then this will write the test into the given bytes (Does not include the test code)...
typedef void (*LearnerFetch)(Learner this, void * out);



// Define the learner type...
typedef struct LearnerType LearnerType;

struct LearnerType
{
 const char code; // Code to get this learner.
 const char code_test; // Code for the test type it outputs.
 const char * name;
 const char * description;

 LearnerNew init;
 LearnerDelete deinit;
 
 LearnerOptimise optimise;
 
 LearnerEntropy entropy;
 LearnerSize size;
 LearnerFetch fetch;
};



// Set of methods for interacting with learners...
Learner Learner_new(char code, DataMatrix * dm, int feature);
void Learner_delete(Learner this);

int Learner_optimise(Learner this, InfoSet * info, IndexView * view, int depth, float improve, unsigned int key[4]);

char Learner_test_code(Learner this);
float Learner_entropy(Learner this);
size_t Learner_size(Learner this);
void Learner_fetch(Learner this, void * out);



// The various learner types in the system...
// No-op learner - no idea why you would use it, but its here for completeness...
const LearnerType IdiotLearner; // Code = I.

// The default learner for real data - finds an optimal split point...
const LearnerType SplitLearner; // Code = S.

// The default learner for discrete data - one discrete value passes, all other fail...
const LearnerType OneCatLearner; // Code = O.



// List of learner types, for automatic detection...
extern const LearnerType * ListLearner[];



// The LearnerSet - a Learner for each feature in an input data matrix - has the ability to find the best test given an index set to optimise for, then get the test as a bunch of bytes...
typedef struct LearnerSet LearnerSet;

struct LearnerSet
{
 int best; // Index of best, negative if none.
 int * feat; // Buffer of feature indices - to avoid allocating memory each time it shuffles them. (Actually stored after the learn array...)
 
 int features;
 Learner learn[0];
};



// Methods for the LearnerSet object...

// Creates it - takes a data matrix and optional codes for if you want to configure the individual learners on a per feature basis...
LearnerSet * LearnerSet_new(DataMatrix * dm, const char * codes);

// Terminate a LearnerSet, with extreme prejudice...
void LearnerSet_delete(LearnerSet * this);

// Optimises the split for the data in the given IndexView with the metric in the InfoSet; IndexView will be super jumbled by the process. features is how many randomly selected features to try optimising (without replacement - if greater than # features it just does them all), depth is the depth this is being done at, as required by the InfoView. improve is the entropy it is aiming to better, whilst key is for the random number generator, and will be incrimented as/if its used. Returns non-zero if its found something, zero if it failed...
int LearnerSet_optimise(LearnerSet * this, InfoSet * info, IndexView * view, int features, int depth, float improve, unsigned int key[4]);

// If its found a solution this returns that solutions entropy - required to be less than improve for the last call, otherwise it would of indicated failure...
float LearnerSet_entropy(LearnerSet * this);

// If its found a solution this is the test code of the blob it will output...
char LearnerSet_code(LearnerSet * this);

// If the learner has just done a successful optimisation then this should return how many bytes the test requires to be written into...
size_t LearnerSet_size(LearnerSet * this);

// If the learner has just done a successful optimisation then this will write the test into the given bytes (Does not include the test code)...
void LearnerSet_fetch(LearnerSet * this, void * out);



// Function to run a test - takes a DataMatrix and an exemplar, returns non-zero if it passed, zero if it failed...
typedef int (*DoTest)(const void * test, DataMatrix * dm, int exemplar);

// Test code to test table for tests - allows you to get at the functions for each test...
extern DoTest CodeToTest[256];

// Function that returns the size of the datablock of a test...
typedef size_t (*TestSize)(const void * test);

// Test code to size table for tests - allows you to get at the functions for each test...
extern TestSize CodeToSize[256];

// Helper function - uses the above table to perform a test - given the tests code and test data, as generated by a Learner, then a DataMatrix and exemplar to perform the test on - returns non-zero if it passed, zero if it failed...
int Test(char code, void * test, DataMatrix * dm, int exemplar);

// Same as above, but for size...
size_t TestSize(char code, void * test);



// Internal method that fills in CodeToTest...
void SetupCodeToTest(void);



#endif
