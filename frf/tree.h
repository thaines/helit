#ifndef TREE_H
#define TREE_H

// Copyright 2014 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



// Provides the node object - the basis of a tree, and really the complete system but lacking the primary Python interface...

#include "summary.h"
#include "index_set.h"



// 'Simple' structure of stuff to pass into a tree - makes life easier...
typedef struct TreeParam TreeParam;

struct TreeParam
{
 DataMatrix * x; // We are learning a function from these values...
 DataMatrix * y; // ... to these values.
 
 LearnerSet * ls; // Defines the kind of tests to consider for each feature and does the optimisation.
 InfoSet * is; // Defines the metric to be optimised.
 const char * summary_codes; // Codes to use when making summary objects.
 
 unsigned int * key; // User is allowed to change these as it uses random data.
 
 int opt_features; // Number of (randomly selected) features to try optimising for each split.
 int min_exemplars; // Cancels a split if either half ends up smaller than this.
 int max_splits; // Maximum number of splits to do, effectivly the maximum depth.
};



// Define the tree object - its one single block of memory for all the data, plus a single pointer to some extra stuff to accelerate indexing during runtime, but its designed to be dumped to disk at will, and then recovered afterwards, and used with only the building of a simple index...
typedef struct Tree Tree;

struct Tree
{
 char magic[4]; // Magic number 'FRFT'
 int revision; // Incase there are ever multiple formats - currently 1.
 long long size; // How big entire tree blob is - assuming long long is 64 bits.
  
 int objects; // Number of entities.
 int dummy; // So the below is definitly on the 8 byte boundary - compiler would probably put it there anyway.
 void ** index; // Index - gets you a pointer to each object. Has to be rebuilt after reloading. Note - first object is an int aligned array of chars, giving types for the rest of the objects. 'N' for node, 'S' for summary.
};



// Methods to learn a new tree from some data - internally this is rather complicated, but only because it has to deal with all the crazy memory stuff - all the real work is elsewhere in this library. Return value will have been malloc'ed - user needs to free. More of the parameters are in the param struct - see it for details, but a seperate index set of exemplars to use is required. If provided oob_error will be filled in with anything in the DataMatrix not in indices...
Tree * Tree_learn(TreeParam * param, IndexSet * indices, float * oob_error);


// Returns non-zero if it thinks its a tree - i.e. the magic numbers and revision are correct, zero if there is a problem...
int Tree_safe(Tree * this);

// If you have just created a memory block to contain a tree then this rebuilds the index. Must be called before actually using the tree. Returns zero if it doesn't think you actually have a tree (and sets a python error), nonzero if all is good. On zero you don't call deinit...
int Tree_init(Tree * this);

// When done with a tree this cleans up the index, but not the memory of the actual Tree object...
void Tree_deinit(Tree * this);


// Returns how many bytes in size the Tree object should be - could be larger than sizeof(Tree) due to 32 bit/64 bit issues. Good number of bytes to read in before calling size and creating a real memory block...
size_t Tree_head_size(void);

// Returns how big the tree object is, in bytes. Note that this can be called during the loading process on the first sizeof(Tree) bytes to still get the correct answer...
size_t Tree_size(Tree * this);

// Returns how many objects are in the tree...
int Tree_objects(Tree * this);


// Given a bunch of exemplars this outputs a measure of their error, one measure for each feature - same code as oob calculation, but for arbitrary data - for n-fold validation/hold out sets etc...
void Tree_error(Tree * this, DataMatrix * x, DataMatrix * y, IndexView * view, float * out);

// Runs a Tree on a single exemplar - returns the SummarySet object that it lands in...
SummarySet * Tree_run(Tree * this, DataMatrix * x, int exemplar);

// Runs a Tree on many exemplars, recording the result into the provided array - step is how many to step between entries in out when writting the output, so you can interleave values from multiple trees as required by the SummarySet_merge_many_py method. Assumes that IndexSet is everything in the DataMatrix...
void Tree_run_many(Tree * this, DataMatrix * x, IndexSet * is, SummarySet ** out, int step);

// Converts the Tree into a Python object suitable for human consumption - tests and summaries (leaf nodes) are represented as strings, whilst non-leaf nodes are represented with dictionaries, containing 'test', 'pass' and 'fail'...
PyObject * Tree_human(Tree * this);



// Setup this module - for internal use only...
void Setup_Tree(void);



#endif
