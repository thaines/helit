#ifndef GBP_C_H
#define GBP_C_H

// Copyright 2014 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



// Pre-declerations...
typedef struct Node Node;
typedef struct HalfEdge HalfEdge;
typedef struct Edge Edge;
typedef struct Block Block;
typedef struct GBP GBP;



// Node - a univariate Gaussian random variable basically...
struct Node
{
 HalfEdge * first; // First half edge leaving this node. 
 
 float unary_pmean; // p-mean for the unary term. Just mean if prec is infinity.
 float unary_prec; // Precision for the unary term; can be zero.
 
 float pmean; // Estimated p-mean for this random variable.
 float prec; // Estimated precision for this random variable.
 
 int chain_count; // Number of chains that include this node, or -1 if not calculated.
 int on; // Non-zero if its to be processed, otherwise as though it doesn't exist.
};



// Half an edge - pairs of these create a pairwise term between nodes...
struct HalfEdge
{
 Node * dest;
 HalfEdge * reverse; // Its partner in crime, going in the other direction.
 HalfEdge * next; // Next in the list of edges leaving a given node.
 
 float pmean; // p-mean of its message.
 float prec; // precision of its message.
};



// An edge - contains the details of the relationship, in terms of the two HalfEdge objects that define the directions and the pmean and precision of the avaliable information going in the forward direction...
struct Edge
{
 union
 {
  Edge * next; // For when in the gc list.
  HalfEdge forward; // for normal usage!
 };
 HalfEdge backward;
 
 // Final relationship between the two nodes is defined as:
 // pmean = [-poffset, poffset]
 // precision = [diag, co-diag; co-diag, diag]
 // i.e.
 // poffset is diag multiplied by the offset between the random variables.
 // diag is the precision of the offset between variables.
 // co is the precision between the random variables if your thinking of the GBP model as a Gaussian with a sparse precision.
  float poffset;
  float diag;
  float co;
};



// A block - a big lump of Edge structures so we don't malloc too often.
struct Block
{
 Block * next;
 Edge data[0];
};



// The actual object...
struct GBP
{
 PyObject_HEAD
 
 int node_count;
 Node * node;
 
 int edge_count; // No list - access by looping nodes.
 
 Edge * gc; // Linked list of edges that can be recycled; use next pointer on forward HalfEdge.
 
 int block_size; // Number of half edge pairs to malloc at a time.
 Block * storage;
};



#endif
