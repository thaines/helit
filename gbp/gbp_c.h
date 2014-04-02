#ifndef GBP_C_H
#define GBP_C_H

// Copyright 2014 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



// Pre-declerations...
typedef struct Node Node;
typedef struct HalfEdge HalfEdge;
typedef struct GBP GBP;



// Node - a univariate Gaussian random variable basically...
struct Node
{
 HalfEdge * first; // First half edge leaving this node. 
 
 float unary_p_mean; // p-mean for the unary term.
 float unary_prec; // Precision for the unary term; can be zero.
 
 float pmean; // Estimated p-mean for this random variable.
 float prec; // Estimated precision for this random variable.
};



// Half an edge - pairs of these create a pairwise term between nodes...
struct HalfEdge
{
 Node * dest;
 HalfEdge * reverse; // Its partner in crime, going in the other direction.
 HalfEdge * next; // Next in the list of edges leaving a given node.
 
 float pairwise; // Contains either the p-mean or precision - see accessor functions for details of this rather convoluted attempt to save 8 bytes/avoid storing the same information twice.
 
 float pmean; // p-mean of its message.
 float prec; // precision of its message.
};



// The actual object...
struct GBP
{
 PyObject_HEAD
 
 int node_count;
 Node * node;
 
 int half_edge_count; // No list - access by looping nodes.
 
 int block_size; // Number of half edge pairs to malloc at a time.
 HalfEdge * gc; // Linked list of half edge pairs that can be recycled; use next pointer.
};



#endif
