#ifndef NODE_H
#define NODE_H

// Copyright 2014 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



// Provides the node object - the basis of a tree, and really the complete system but lacking the primary Python interface...

#include "summary.h"



// Node types - basically leaf and not leaf...
static const short TEST    = 0;
static const short SUMMARY = 1;



// Define the Node object - its required to be a single block of memory that free can safely be called on...
typedef struct Node Node;

struct Node
{
 // Indices to its children objects - either an index into the node list or an index into the summary list, depending on the given type...
  short fail_type;
  short pass_type;
  int fail;
  int pass;

 // The test to decide which direction to go - code stored here, with the actual test data stored immediatly afterwards...
  char code;
  char test[0];
};



struct Tree
{
 int nodes; // Number of nodes.
 Node ** node; // For indexing nodes - they are all in the same memory block as Tree.
 
 int summaries; // Number of summaries.
 SummarySet ** ss; // For indexing summaries - they (curently) own their memory and need to be freed.
};



// Methods to learn a new tree from a 



#endif
