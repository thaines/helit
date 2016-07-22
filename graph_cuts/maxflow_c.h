#ifndef MAXFLOW_C_H
#define MAXFLOW_C_H

// Copyright 2016 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



// Pre-declerations...
typedef struct Node Node;
typedef struct HalfLink HalfLink;
typedef struct MaxFlow MaxFlow;



// The structure that represents a vertex...
struct Node
{
 HalfLink * first; // Linked list of half edges that leave this vertex.
 
 HalfLink * parent; // Parent of this node in the tree, communicated by the edge you follow for which the apretn is the destination.
 
 Node * prev_active; // Doubly linked list of active nodes.
 Node * next_active;
 Node * next_orphan; // Linked list of orphans.
 
 int depth_valid; // Used to cache via path shortening depth values during adoption.
 int depth;

 char owner; // -1 = source, 0 = free, 1 = sink.
};



// The structure that represents a half edge - each edge consists of two of these...
struct HalfLink
{
 Node * dest; // Node to which it goes to.
 HalfLink * other; // The other half edge which pairs with this to make an entire edge.
 
 HalfLink * next; // Next HalfLink for the vertex this is leaving.
 
 float remain; // Remaining flow that can be sent in this direction along this edge.
};



// The structure that represents the solver for maxflow...
struct MaxFlow
{
 PyObject_HEAD
 
 int vertex_count;
 int true_vertex_count; // Vertex count in memory, for the resize method.
 Node * vertex;
 
 int edge_count; // Not used, just conveniant for the api.
 int half_edge_count;
 int true_half_edge_count; // Halves in actual memory block, for resize.
 HalfLink * half_edge; // The two halfs that make up each edge are adjacent.
 
 int source;
 int sink;
 
 Node active; // Dummy node, used to do doubly connected circular linked list of active nodes.
 float max_flow; // Amount of flow that has been sent along the graph.
};



// Struct of C pointers to the various API functions...
typedef struct MaxFlowAPI MaxFlowAPI;

struct MaxFlowAPI
{
 // Create and delete the object, within a block of memory the user provides, sizeof(MaxFlow). The user is responsible for malloc/free on the basic block, these on internal structures...
  int (*init)(MaxFlow * this, int vertex_count, int edge_count);
  void (*deinit)(MaxFlow * this);
  int (*resize)(MaxFlow * this, int vertex_count, int edge_count);
  
 // For setting/getting the source/sink vertices...
  void (*set_source)(MaxFlow * this, int index);
  void (*set_sink)(MaxFlow * this, int index);
  int (*get_source)(MaxFlow * this);
  int (*get_sink)(MaxFlow * this);
  
 // To setup the edges - just lists of vertex indices (steps for if integers are not adjacent)...
  void (*set_edges)(MaxFlow * this, int * from, int * to, size_t step_from, size_t step_to);
  
 // Resets the edges...
  void (*reset_edges)(MaxFlow * this);
  
 // Sets an edge range - will not overwrite any edge taht has already been set, hence the existance of reset edges...
  void (*set_edge)(MaxFlow * this, int edge, int from, int to);
  void (*set_edges_range)(MaxFlow * this, int start, int length, int * from, int * to, size_t step_from, size_t step_to);
  
 // Sets how much can flow along each of the edges...
  void (*cap_flow)(MaxFlow * this, int edge, float neg_max, float pos_max);
  void (*set_flow_cap)(MaxFlow * this, float * neg_max, float * pos_max, size_t step_neg, size_t step_pos);
  void (*set_flow_cap_double)(MaxFlow * this, double * neg_max, double * pos_max, size_t step_neg, size_t step_pos);
  void (*set_flow_cap_range)(MaxFlow * this, int start, int length, float * neg_max, float * pos_max, size_t step_neg, size_t step_pos);
  void (*set_flow_cap_range_double)(MaxFlow * this, int start, int length, double * neg_max, double * pos_max, size_t step_neg, size_t step_pos);
  
 // Solves the maxflow problem...
  void (*solve)(MaxFlow * this);
  
 // Fetches which side a single vertex is on: -1 for source, 1 for sink...
  int (*get_side)(MaxFlow * this, int vertex);
  
 // Fetches which side all vertices are on - complex interface...
  void (*store_side)(MaxFlow * this, int * out, int * source, int * sink, size_t elem_size, size_t out_step, size_t source_step, size_t sink_step);
  void (*store_side_range)(MaxFlow * this, int start, int length, int * out, int * source, int * sink, size_t elem_size, size_t out_step, size_t source_step, size_t sink_step);
  
 // Get amount of unused flow exists for each edge...
  void (*get_unused)(MaxFlow * this, int edge, float * neg, float * pos);
  void (*store_unused)(MaxFlow * this, float * neg, float * pos, size_t neg_step, size_t pos_step);
};



#ifdef USE_MAXFLOW_C

static MaxFlowAPI * maxflow;

static int import_maxflow(void)
{
 maxflow = (MaxFlowAPI*)PyCapsule_Import("maxflow_c._C_API", 0);
 return (maxflow!=NULL) ? 0 : -1;
}

#endif



#endif
