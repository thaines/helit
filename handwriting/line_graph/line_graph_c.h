#ifndef LINE_GRAPH_C_H
#define LINE_GRAPH_C_H

// Copyright 2016 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

#include <Python.h>
#include <structmember.h>
#include <numpy/arrayobject.h>



// Pre-decleration...
typedef struct SplitTag SplitTag;

typedef struct Vertex Vertex;
typedef struct HalfEdge HalfEdge;
typedef struct Edge Edge;

typedef struct Region Region;
typedef struct LineGraph LineGraph;

typedef struct EdgeInt EdgeInt;



// Represents a split or a tag, depending on if the tag string is null or not.
struct SplitTag
{
 Edge * loc; // Which edge it is on.

 SplitTag * next; // Its pointers to adjacent elements.
 SplitTag * prev;

 char * tag; // null if it is a split, a null terminated string if it is a tag (owned by malloc).
 SplitTag * other; // For if its paired tag, used to indicate relationships between parts.
 float t; // Where along the half-edge to put it - 0 for the start, 1 for the end.
 int segment; // segment assignment after this split; only valid if it is a split.
};



// The structure that represents a vertex...
struct Vertex
{
 HalfEdge * incident; // Pointer to a half edge that leaves this node.

 float x;
 float y;

 float u; // So we can always get back to the original coordinates after it has been distorted.
 float v; // "
 float w; // Distance traveled in u,v space if we travel radius.

 float radius;
 float density;
 float weight;
 
 int source; // Index of vertex in source LineGraph this LineGraph was created from, or -1 if none.
};



// The structure that represents a half-edge...
struct HalfEdge
{
 HalfEdge * reverse; // Partner in crime that goes the other way.
 Vertex * dest; // Destination vertex.

 HalfEdge * next; // Next edge going around the hole this edge wraps (clockwise.)
 HalfEdge * prev; // Other way to the above - anti-clockwise.
};



// A complete edge - two half edges and SplitTag doubly linked list with circular dummy...
struct Edge
{
 HalfEdge pos;
 HalfEdge neg;
 
 SplitTag dummy; // A doubly linked circular list of splits/tags, dummy node. The objects are malloc-ed and owned.
 int source; // Index of edge in source LineGraph this LineGraph was created from, or -1 if none.
 int segment; // Segment assignment at the start of the edge (Closest to neg) - it can be changed by splits along the edge.
};

inline Edge * HalfToEdge(HalfEdge * half)
{
 if (half->reverse < half) half = half->reverse;
 return (Edge*)(void*)((char*)(void*)half - offsetof(Edge, pos));
}



// The spatial structure used for collisions, using ranges of edges - we actually use a binary tree and sort the edge array so we can use ranges, rather than random access...
struct Region
{
 float min_x;
 float max_x;
 float min_y;
 float max_y;

 int begin; // inclusive
 int end;   // exclusive

 Region * child_low; // Owned pointers.
 Region * child_high; // "

 Region * next; // Temporary - used to return spatial queries as a list of region objects.
};



// Actual structure that represents this data structure in python...
struct LineGraph
{
 PyObject_HEAD

 int vertex_count;
 Vertex * vertex;

 int edge_count;
 Edge * edge;

 Region * root;
 int segments; // -1 if graph is not segmented, how many segments exist if it is valid.
};



// Represents a set of edge intercepts - Edge and two t values, one for the edge, the other for the (unspecified) other entity. Forms a singularly linked list...
struct EdgeInt
{
 Edge * edge;
 float edge_t;
 float other_t;
 EdgeInt * next;
};



#endif
