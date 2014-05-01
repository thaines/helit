#ifndef INDEX_SET_H
#define INDEX_SET_H

// Copyright 2014 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



// Provides a set of indices to a data matrix exemplar set - used to represent the indices that have gone down a particular branch of a tree and also does bootstrap sampling/identifies the out of bag set...

#include "data_matrix.h"



// The index set object...
typedef struct IndexSet IndexSet;

struct IndexSet
{
 int size; // Size of below array.
 int vals[0]; // The associated exemplar indices.
};


// Create and destroy an index set...
IndexSet * IndexSet_new(int size);
void IndexSet_delete(IndexSet * this);

// Initalises an index set - can either do all samples or a bootstrap draw. Note that key will be modified, and left at a position where it can be used for the next use if you want...
void IndexSet_init_all(IndexSet * this);
void IndexSet_init_bootstrap(IndexSet * this, unsigned int key[4]);



// The index view object - view of part of an index set - sent down the tree indicating the indices of the data set relevant to learning the current node - basically a pointer to a subpart of an IndexSet that will be reordered and split to be sent down to leafs...
typedef struct IndexView IndexView;

struct IndexView
{
 int size; // Size of below array.
 int * vals; // The associated exemplar indices.
};


// Initialise an index view from an index set, to view the entire set...
void IndexView_init(IndexView * this, IndexSet * source);

// Given a data matrix, and a test this splits the index view into two - a passed and a failed set...
// *************************************************



#endif
