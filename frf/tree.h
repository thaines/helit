#ifndef TREE_H
#define TREE_H

// Copyright 2014 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



// Provides the node object - the basis of a tree, and really the complete system but lacking the primary Python interface...

#include "summary.h"



// Define the tree object - its one single block of memory for all the data, plus a single pointer to some extra stuff to accelerate indexing during runtime, but its designed to be dumped to disk at will, and then recovered afterwards, and used with only the building of a simple index...
typedef struct Tree Tree;

struct Tree
{
 char magic[4]; // Magic number 'FRFT'
 int revision; // Incase there are ever multiple formats.
 long long size; // How big entire tree blob is - assuming long long is 64 bits.
  
 int objects; // Number of entities.
 void ** index; // Index - gets you a pointer to each object. Has to be rebuilt after reloading. Note - first object is an int aligned array of chars, giving types for the rest of the objects. 'N' for node, 'S' for summary.
};



// Methods to learn a new tree from some data - internally this is rather complicated, but only because it has to deal with all the crazy memory stuff - all the real work is elsewhere in this library. Return value will have been malloc'ed - user needs to free. Its learning a function from x to y, using the exemplars indicated by the view. 'ls' provides the learning/tests, 'is' defines what is to be optimised, summary_codes controls the summary objects that are created at leafs, and can be NULL. If provided oob_error will be filled in with anything in the DataMatrix not used by the view, and key is for the random number generator...
Tree * Tree_learn(DataMatrix * x, DataMatrix * y, IndexView * view, LearnerSet * ls, InfoSet * is, const char * summary_codes, float * oob_error, unsigned char key[4]);


// If you have just created a memory block to contain a tree then this rebuilds the index. Must be called before actually using the tree...
void Tree_init(Tree * this);

// When done with a tree this cleans up the index, but not the memory of the actual Tree object...
void Tree_deinit(Tree * this);


// Returns how big the tree object is, in bytes...
size_t Tree_size(Tree * this);

// Returns how many objects are in the tree...
int Tree_objects(Tree * this);


// Given a bunch of exemplars this outputs a measure of their error, one measure for each feature - same code as oob calculation, but for arbitrary data - for n-fold validation/hold out sets etc...
void Tree_error(Tree * this, DataMatrix * x, DataMatrix * y, IndexView * view, float * out);

// Runs a Tree on a single exemplar - returns the Summary object for output...
SummarySet * Tree_run(Tree * this, DataMatrix * x, int exemplar);

// Runs a Tree on many exemplars, recording the result into the provided array - step is how many to step between entries in out when writting the output, so you can interleave values from multiple trees as required by the SummarySet_merge_many_py method...
void Tree_run_many(Tree * this, DataMatrix * x, IndexView * view, SummarySet ** out, int step);



#endif
