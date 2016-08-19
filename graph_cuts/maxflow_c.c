// Copyright 2016 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

#include <Python.h>
#include <structmember.h>
#include <numpy/arrayobject.h>



#include "maxflow_c.h"



static int MaxFlow_init(MaxFlow * this, int vertex_count, int edge_count)
{
 int i;
 
 this->vertex = NULL;
 this->half_edge = NULL;
 
 // The vertices...
  this->vertex_count = vertex_count;
  this->true_vertex_count = this->vertex_count;
  this->vertex = (Node*)malloc(sizeof(Node)*this->vertex_count);
  if (this->vertex==NULL)
  {
   printf("Error: Out of memory allocating %lu bytes (vertices). (ptr size = %lu)\n", sizeof(Node)*this->vertex_count, sizeof(void*));
   return 1;
  }
  
  Node * vertex = this->vertex;
  for (i=0; i<this->vertex_count; i++, vertex++)
  {
   vertex->first = NULL;
   vertex->parent = NULL;
   vertex->prev_active = vertex;
   vertex->next_active = vertex;
   vertex->next_orphan = NULL;
   vertex->owner = 0;
  }
 
 // The edges...
  this->edge_count = edge_count;
  this->half_edge_count = edge_count * 2;
  this->true_half_edge_count = this->half_edge_count;
  this->half_edge = (HalfLink*)malloc(sizeof(HalfLink)*this->half_edge_count);
  if (this->half_edge==NULL)
  {
   printf("Error: Out of memory allocating %lu bytes (half edges). (ptr size = %lu)\n", sizeof(HalfLink)*this->half_edge_count, sizeof(void*)); 
   return 1;
  }
  
  HalfLink * half_edge = this->half_edge;
  for (i=0; i<this->half_edge_count; i++, half_edge++)
  {
   half_edge->dest = NULL;
   half_edge->other = ((i%2)==0) ? (half_edge+1) : (half_edge-1);
   half_edge->next = NULL;
   half_edge->remain = 0.0;
  }
  
 // The list of active vertices...
  this->active.first = NULL;
  this->active.parent = NULL;
  this->active.prev_active = &this->active;
  this->active.next_active = &this->active;
  this->active.next_orphan = NULL;
  this->active.owner = 0;
 
 // Other variables...
  this->source = -1;
  this->sink = -1;
 
  this->max_flow = 0.0;
  
 // Return 0 on success...
  return 0;
}

static void MaxFlow_deinit(MaxFlow * this)
{
 free(this->vertex);
 this->vertex = NULL;
 
 free(this->half_edge);
 this->half_edge = NULL;
}

static int MaxFlow_resize(MaxFlow * this, int vertex_count, int edge_count)
{
 int i;
 
 // The vertices...
  this->vertex_count = vertex_count;
  if (this->vertex_count>this->true_vertex_count)
  {
   Node * temp = this->vertex;
   this->vertex = (Node*)realloc(this->vertex, sizeof(Node)*this->vertex_count);
   if (this->vertex==NULL)
   {
    this->vertex = temp;
    printf("Error: Out of memory allocating %lu bytes (vertices). (ptr size = %lu)\n", sizeof(Node)*this->vertex_count, sizeof(void*));
    return 1;
   }
   this->true_vertex_count = this->vertex_count;
  }
  
  Node * vertex = this->vertex;
  for (i=0; i<this->vertex_count; i++, vertex++)
  {
   vertex->first = NULL;
   vertex->parent = NULL;
   vertex->prev_active = vertex;
   vertex->next_active = vertex;
   vertex->next_orphan = NULL;
   vertex->owner = 0;
  }
 
 // The edges...
  this->edge_count = edge_count;
  this->half_edge_count = edge_count * 2;
  if (this->half_edge_count>this->true_half_edge_count)
  {
   HalfLink * temp = this->half_edge;
   this->half_edge = (HalfLink*)realloc(this->half_edge, sizeof(HalfLink)*this->half_edge_count);
   if (this->half_edge==NULL)
   {
    this->half_edge = temp;
    printf("Error: Out of memory allocating %lu bytes (half edges). (ptr size = %lu)\n", sizeof(HalfLink)*this->half_edge_count, sizeof(void*)); 
    return 1;
   }
   this->true_half_edge_count = this->half_edge_count;
  }
  
  HalfLink * half_edge = this->half_edge;
  for (i=0; i<this->half_edge_count; i++, half_edge++)
  {
   half_edge->dest = NULL;
   half_edge->other = ((i%2)==0) ? (half_edge+1) : (half_edge-1);
   half_edge->next = NULL;
   half_edge->remain = 0.0;
  }
  
 // The list of active vertices...
  this->active.first = NULL;
  this->active.parent = NULL;
  this->active.prev_active = &this->active;
  this->active.next_active = &this->active;
  this->active.next_orphan = NULL;
  this->active.owner = 0;
 
 // Other variables...
  this->source = -1;
  this->sink = -1;
 
  this->max_flow = 0.0;
  
 // Return 0 on success...
  return 0;
}


static PyObject * MaxFlow_new_py(PyTypeObject * type, PyObject * args, PyObject * kwds)
{
 // Extract the arguments...
  int vertex_count, edge_count;
  if (!PyArg_ParseTuple(args, "ii", &vertex_count, &edge_count)) return NULL;
 
 // Allocate the object...
  MaxFlow * self = (MaxFlow*)type->tp_alloc(type, 0);
 
 // On success construct it...
  if (self!=NULL)
  {
   int res = MaxFlow_init(self, vertex_count, edge_count);
   if (res!=0)
   {
    MaxFlow_deinit(self);
    self->ob_type->tp_free((PyObject*)self);
    return PyErr_NoMemory();
   }
  }
  
 // Return the new object...
  return (PyObject*)self;
}

static void MaxFlow_dealloc_py(MaxFlow * self)
{
 MaxFlow_deinit(self);
 self->ob_type->tp_free((PyObject*)self);
}

static PyObject * MaxFlow_resize_py(MaxFlow * self, PyObject * args, PyObject * kwds)
{
 // Extract the arguments...
  int vertex_count, edge_count;
  if (!PyArg_ParseTuple(args, "ii", &vertex_count, &edge_count)) return NULL;
 
 // On success construct it...
  int res = MaxFlow_resize(self, vertex_count, edge_count);
  if (res!=0)
  {
   return PyErr_NoMemory();
  }
  
 // Return the new object...
  return (PyObject*)self;
}



static PyMemberDef MaxFlow_members[] =
{
 {"vertex_count", T_INT, offsetof(MaxFlow, vertex_count), READONLY, "Number of vertices in the graph."},
 {"edge_count", T_INT, offsetof(MaxFlow, edge_count), READONLY, "Number of edges in the graph."},
 {"half_edge_count", T_INT, offsetof(MaxFlow, half_edge_count), READONLY, "Number of half-edges in the graph - divide by two to get the actual number of edges."},
 {"max_flow", T_FLOAT, offsetof(MaxFlow, max_flow), READONLY, "Maximum flow across the graph - will be 0.0 if the algorithm has not been run."},
 {NULL}
};



static void MaxFlow_set_source(MaxFlow * this, int index)
{
 this->source = index;
}

static void MaxFlow_set_sink(MaxFlow * this, int index)
{
 this->sink = index;
}


static PyObject * MaxFlow_set_source_py(MaxFlow * self, PyObject * args)
{
 // Extract the variables...
  int index;
  if (!PyArg_ParseTuple(args, "i", &index)) return NULL;
 
 // Do the work...
  MaxFlow_set_source(self, index);
  
 // Return None...
  Py_INCREF(Py_None);
  return Py_None;
}

static PyObject * MaxFlow_set_sink_py(MaxFlow * self, PyObject * args)
{
 // Extract the variables...
  int index;
  if (!PyArg_ParseTuple(args, "i", &index)) return NULL;
 
 // Do the work...
  MaxFlow_set_sink(self, index);
  
 // Return None...
  Py_INCREF(Py_None);
  return Py_None;
}



static int MaxFlow_get_source(MaxFlow * this)
{
 return this->source; 
}

static int MaxFlow_get_sink(MaxFlow * this)
{
 return this->sink; 
}


static PyObject * MaxFlow_get_source_py(MaxFlow * self, PyObject * args)
{
 // Do the work...
  int ret = MaxFlow_get_source(self);
  
 // Return the value...
  if (ret<0)
  {
   Py_INCREF(Py_None);
   return Py_None;
  }
  else
  {
   return PyInt_FromLong(ret);
  }
}

static PyObject * MaxFlow_get_sink_py(MaxFlow * self, PyObject * args)
{
 // Do the work...
  int ret = MaxFlow_get_sink(self);
  
 // Return the value...
  if (ret<0)
  {
   Py_INCREF(Py_None);
   return Py_None;
  }
  else
  {
   return PyInt_FromLong(ret);
  }
}



static void MaxFlow_set_edges(MaxFlow * this, int * from, int * to, size_t step_from, size_t step_to)
{
 int i;
 
 // Reset the edge lists for each vertex...
  Node * vertex = this->vertex;
  for (i=0; i<this->vertex_count; i++, vertex++)
  {
   vertex->first = NULL;
  }
 
 // Loop the edges in the list, copying the connectivity over for each...
  HalfLink * target = this->half_edge;
  for (i=0; i<this->half_edge_count; i+=2, target+=2)
  {
   // Link up the two edge halfs at this position to the relevant vertices...
    target[0].dest = &this->vertex[*to];
    target[0].next = this->vertex[*from].first;
    this->vertex[*from].first = &target[0];
    
    target[1].dest = &this->vertex[*from];
    target[1].next = this->vertex[*to].first;
    this->vertex[*to].first = &target[1];
    
   // Move to the next entry in the provided arrays - complex because we need to support arrays with arbitrary strides...
    from = (int*)(void*)((char*)(void*)from + step_from);
    to = (int*)(void*)((char*)(void*)to + step_to);
  }
}


static PyObject * MaxFlow_set_edges_py(MaxFlow * self, PyObject * args)
{
 // Extract the two numpy arrays...
  PyArrayObject * from;
  PyArrayObject * to;
  if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &from, &PyArray_Type, &to)) return NULL;
  
  if (from->nd!=1 || to->nd!=1)
  {
   PyErr_SetString(PyExc_TypeError, "Links must be given using one dimensional arrays");
   return NULL;
  }
  
  if (from->dimensions[0]!=self->edge_count || to->dimensions[0]!=self->edge_count)
  {
   PyErr_SetString(PyExc_IndexError, "Link arrays are not the length of the edge count.");
   return NULL;
  }
  
  if (from->descr->kind!='i' || from->descr->elsize<sizeof(int) || to->descr->kind!='i' || to->descr->elsize<sizeof(int))
  {
   PyErr_SetString(PyExc_TypeError, "Link arrays must be of integer type.");
   return NULL;
  }
  
 // Setup the edges...
  MaxFlow_set_edges(self, (int*)(void*)from->data, (int*)(void*)to->data, from->strides[0], to->strides[0]);
 
 // Return None...
  Py_INCREF(Py_None);
  return Py_None;
}



static void MaxFlow_reset_edges(MaxFlow * this)
{
 int i;
 
 // Reset the edge lists for each vertex...
  Node * vertex = this->vertex;
  for (i=0; i<this->vertex_count; i++, vertex++)
  {
   vertex->first = NULL;
  }
  
 // Null the dest pointer for each half edge...
  HalfLink * half_edge = this->half_edge;
  for (i=0; i<this->half_edge_count; i++, half_edge++)
  {
   half_edge->dest = NULL; 
  }
}

static void MaxFlow_set_edge(MaxFlow * this, int edge, int from, int to)
{
 HalfLink * target = this->half_edge + 2*edge;
 if (target[0].dest==NULL)
 {
  target[0].dest = &this->vertex[to];
  target[0].next = this->vertex[from].first;
  this->vertex[from].first = &target[0];
    
  target[1].dest = &this->vertex[from];
  target[1].next = this->vertex[to].first;
  this->vertex[to].first = &target[1];
 }
}

static void MaxFlow_set_edges_range(MaxFlow * this, int start, int length, int * from, int * to, size_t step_from, size_t step_to)
{
 int i;

 // Loop the edges in the list, copying the connectivity over for each...
  HalfLink * target = this->half_edge + 2*start;
  for (i=0; i<length; i++, target+=2)
  {
   // Link up the two edge halfs at this position to the relevant vertices...
    if (target[0].dest==NULL)
    {
     target[0].dest = &this->vertex[*to];
     target[0].next = this->vertex[*from].first;
     this->vertex[*from].first = &target[0];
    
     target[1].dest = &this->vertex[*from];
     target[1].next = this->vertex[*to].first;
     this->vertex[*to].first = &target[1];
    }
    
   // Move to the next entry in the provided arrays - complex because we need to support arrays with arbitrary strides...
    from = (int*)(void*)((char*)(void*)from + step_from);
    to = (int*)(void*)((char*)(void*)to + step_to);
  }
}


static PyObject * MaxFlow_reset_edges_py(MaxFlow * self, PyObject * args)
{
 // Do the reset...
  MaxFlow_reset_edges(self);
 
 // Return None...
  Py_INCREF(Py_None);
  return Py_None;
}

static PyObject * MaxFlow_set_edges_range_py(MaxFlow * self, PyObject * args)
{
 // Extract the two numpy arrays...
  int start;
  PyArrayObject * from;
  PyArrayObject * to;
  if (!PyArg_ParseTuple(args, "iO!O!", &start, &PyArray_Type, &from, &PyArray_Type, &to)) return NULL;
  
  if (from->nd!=1 || to->nd!=1)
  {
   PyErr_SetString(PyExc_TypeError, "Links must be given using one dimensional arrays");
   return NULL;
  }
  
  if (from->dimensions[0]!=to->dimensions[0])
  {
   PyErr_SetString(PyExc_IndexError, "Link arrays are not the same length.");
   return NULL;
  }
  
  if (from->descr->kind!='i' || from->descr->elsize<sizeof(int) || to->descr->kind!='i' || to->descr->elsize<sizeof(int))
  {
   PyErr_SetString(PyExc_TypeError, "Link arrays must be of integer type.");
   return NULL;
  }
  
 // Setup the edges...
  MaxFlow_set_edges_range(self, start, from->dimensions[0], (int*)(void*)from->data, (int*)(void*)to->data, from->strides[0], to->strides[0]);
 
 // Return None...
  Py_INCREF(Py_None);
  return Py_None;
}



static void MaxFlow_cap_flow(MaxFlow * this, int edge, float neg_max, float pos_max)
{
 HalfLink * target = this->half_edge + 2*edge;
 
 target[0].remain = pos_max;
 target[1].remain = neg_max;
}



static void MaxFlow_set_flow_cap(MaxFlow * this, float * neg_max, float * pos_max, size_t step_neg, size_t step_pos)
{
 int i;
 
 // Loop the edges in the list, setting the remaining flow for each...
  HalfLink * target = this->half_edge;
  for (i=0; i<this->half_edge_count; i+=2, target+=2)
  {
   // Store the values...
    target[0].remain = *pos_max;
    target[1].remain = *neg_max;
    
   // Move to the next entry in the provided arrays - complex because we need to support arrays with arbitrary strides...
    neg_max = (float*)(void*)((char*)(void*)neg_max + step_neg);
    pos_max = (float*)(void*)((char*)(void*)pos_max + step_pos);
  }
}

static void MaxFlow_set_flow_cap_double(MaxFlow * this, double * neg_max, double * pos_max, size_t step_neg, size_t step_pos)
{
 int i;
 
 // Loop the edges in the list, setting the remaining flow for each...
  HalfLink * target = this->half_edge;
  for (i=0; i<this->half_edge_count; i+=2, target+=2)
  {
   // Store the values...
    target[0].remain = *pos_max;
    target[1].remain = *neg_max;
    
   // Move to the next entry in the provided arrays - complex because we need to support arrays with arbitrary strides...
    neg_max = (double*)(void*)((char*)(void*)neg_max + step_neg);
    pos_max = (double*)(void*)((char*)(void*)pos_max + step_pos);
  }
}

static void MaxFlow_set_flow_cap_range(MaxFlow * this, int start, int length, float * neg_max, float * pos_max, size_t step_neg, size_t step_pos)
{
 int i;
 
 // Loop the edges in the list, setting the remaining flow for each...
  HalfLink * target = this->half_edge + 2*start;
  for (i=0; i<length; i++, target+=2)
  {
   // Store the values...
    target[0].remain = *pos_max;
    target[1].remain = *neg_max;
    
   // Move to the next entry in the provided arrays - complex because we need to support arrays with arbitrary strides...
    neg_max = (float*)(void*)((char*)(void*)neg_max + step_neg);
    pos_max = (float*)(void*)((char*)(void*)pos_max + step_pos);
  }
}

static void MaxFlow_set_flow_cap_range_double(MaxFlow * this, int start, int length, double * neg_max, double * pos_max, size_t step_neg, size_t step_pos)
{
 int i;
 
 // Loop the edges in the list, setting the remaining flow for each...
  HalfLink * target = this->half_edge + 2*start;
  for (i=0; i<length; i++, target+=2)
  {
   // Store the values...
    target[0].remain = *pos_max;
    target[1].remain = *neg_max;
    
   // Move to the next entry in the provided arrays - complex because we need to support arrays with arbitrary strides...
    neg_max = (double*)(void*)((char*)(void*)neg_max + step_neg);
    pos_max = (double*)(void*)((char*)(void*)pos_max + step_pos);
  }
}


static PyObject * MaxFlow_set_flow_cap_py(MaxFlow * self, PyObject * args)
{
 // Extract the two numpy arrays...
  PyArrayObject * neg_max;
  PyArrayObject * pos_max;
  if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &neg_max, &PyArray_Type, &pos_max)) return NULL;
  
  if (neg_max->nd!=1 || pos_max->nd!=1)
  {
   PyErr_SetString(PyExc_TypeError, "Flow limits must be given using one dimensional arrays");
   return NULL;
  }
  
  if (neg_max->dimensions[0]!=self->edge_count || pos_max->dimensions[0]!=self->edge_count)
  {
   PyErr_SetString(PyExc_IndexError, "Flow limit arrays are not the length of the edge count.");
   return NULL;
  }
  
  if (neg_max->descr->kind!='f' || pos_max->descr->kind!='f' || neg_max->descr->elsize!=pos_max->descr->elsize)
  {
   PyErr_SetString(PyExc_TypeError, "Flow limit arrays must be floating point, of the same type.");
   return NULL;
  }
  
  int mode = 0;
  if (neg_max->descr->elsize==sizeof(float)) mode = 1;
  if (neg_max->descr->elsize==sizeof(double)) mode = 2;
  
  if (mode==0)
  {
   PyErr_SetString(PyExc_TypeError, "Flow limit arrays must use a floating point type equivalent to a c float or double.");
   return NULL; 
  }
  
 // Setup the edges...
  if (mode==1)
  {
   MaxFlow_set_flow_cap(self, (float*)(void*)neg_max->data, (float*)(void*)pos_max->data, neg_max->strides[0], pos_max->strides[0]);
  }
  else
  {
   MaxFlow_set_flow_cap_double(self, (double*)(void*)neg_max->data, (double*)(void*)pos_max->data, neg_max->strides[0], pos_max->strides[0]);
  }
 
 // Return None...
  Py_INCREF(Py_None);
  return Py_None;
}

static PyObject * MaxFlow_set_flow_cap_range_py(MaxFlow * self, PyObject * args)
{
 // Extract the two numpy arrays...
  int start;
  PyArrayObject * neg_max;
  PyArrayObject * pos_max;
  if (!PyArg_ParseTuple(args, "iO!O!", &start, &PyArray_Type, &neg_max, &PyArray_Type, &pos_max)) return NULL;
  
  if (neg_max->nd!=1 || pos_max->nd!=1)
  {
   PyErr_SetString(PyExc_TypeError, "Flow limits must be given using one dimensional arrays");
   return NULL;
  }
  
  if (neg_max->dimensions[0]!=pos_max->dimensions[0])
  {
   PyErr_SetString(PyExc_IndexError, "Flow limit arrays are not the same length.");
   return NULL;
  }
  
  if (neg_max->descr->kind!='f' || pos_max->descr->kind!='f' || neg_max->descr->elsize!=pos_max->descr->elsize)
  {
   PyErr_SetString(PyExc_TypeError, "Flow limit arrays must be floating point, of the same type.");
   return NULL;
  }
  
  int mode = 0;
  if (neg_max->descr->elsize==sizeof(float)) mode = 1;
  if (neg_max->descr->elsize==sizeof(double)) mode = 2;
  
  if (mode==0)
  {
   PyErr_SetString(PyExc_TypeError, "Flow limit arrays must use a floating point type equivalent to a c float or double.");
   return NULL; 
  }
  
 // Setup the edges...
  if (mode==1)
  {
   MaxFlow_set_flow_cap_range(self, start, neg_max->dimensions[0], (float*)(void*)neg_max->data, (float*)(void*)pos_max->data, neg_max->strides[0], pos_max->strides[0]);
  }
  else
  {
   MaxFlow_set_flow_cap_range_double(self, start, neg_max->dimensions[0], (double*)(void*)neg_max->data, (double*)(void*)pos_max->data, neg_max->strides[0], pos_max->strides[0]);
  }
 
 // Return None...
  Py_INCREF(Py_None);
  return Py_None;
}



static void MaxFlow_rem_active(Node * v)
{
 v->next_active->prev_active = v->prev_active;
 v->prev_active->next_active = v->next_active;
 
 v->next_active = v;
 v->prev_active = v;
}

static void MaxFlow_add_active(MaxFlow * this, Node * v)
{
 if (v->prev_active!=v) MaxFlow_rem_active( v);
 
 v->next_active = &this->active;
 v->prev_active = this->active.prev_active;
 
 v->next_active->prev_active = v;
 v->prev_active->next_active = v;
}

static int MaxFlow_tree_depth(MaxFlow * this, Node * v, int valid)
{
 // Returns the depth of a node in a tree, or -1 if it is in an orphan tree...
 // (valid is a key for checking cache validity, not used for orphan trees as that could change.)
 
 if (v->depth_valid==valid) return v->depth;
  
 int depth;
 if (v->parent==NULL)
 {
  int pos = v - this->vertex;
  if (v->owner==-1) depth = (pos==this->source) ? 0 : -1;
  else depth = (pos==this->sink) ? 0: -1;
 }
 else
 {
  depth = MaxFlow_tree_depth(this, v->parent->dest, valid);
  if (depth!=-1) depth += 1;
 }
 
 if (depth!=-1)
 {
  v->depth_valid = valid;
  v->depth = depth;
 }

 return depth;
}



static HalfLink * MaxFlow_grow_trees(MaxFlow * this)
{
 // Loop on grabbing an active node and checking all of its neighbours for grow space...
  while (this->active.next_active!=&this->active)
  {
   // Get the first node...
    Node * target = this->active.next_active;
    
   // Iterate and grow each edge...
    HalfLink * half_edge = target->first;
    
    while (half_edge)
    {
     // Process the connection only if flow can be sent over it...
      float flow = (target->owner==-1) ? half_edge->remain : half_edge->other->remain;
      
      if (flow>1e-12)
      {
       char dest_owner = half_edge->dest->owner;
       
       if (dest_owner==0)
       {
        // Its a free node - lets arrange for that to be in the past tense...
         half_edge->dest->parent = half_edge->other;
         half_edge->dest->owner = target->owner;
         MaxFlow_add_active(this, half_edge->dest);
       }
       else
       {
        if (dest_owner!=target->owner)
        {
         // The node belongs to the other tree - return this link as someting to send flow over (In the direction the flow is going to go.)...
          return (target->owner==-1) ? half_edge : half_edge->other;
        }
        // The else here is that both vertices already belong to the same tree - noop.
       }
      }
      
     // Move to next...
      half_edge = half_edge->next;
    }
   
   // If we have successfully dealt with every edge remove it from the active list...
    MaxFlow_rem_active(target);
  }
  
 // No tree to grow...
  return NULL;
}


static Node * MaxFlow_fill_route(MaxFlow * this, HalfLink * link)
{
 // Calculate the maximum flow that can be sent...
  float to_send = link->remain;
  
  // To source...
   Node * target = link->other->dest;
   while ((target-this->vertex)!=this->source)
   {
    float remain = target->parent->other->remain;
    if (remain<to_send) to_send = remain;
    target = target->parent->dest;
   }
  
  // To sink...
   target = link->dest;
   while ((target-this->vertex)!=this->sink)
   {
    float remain = target->parent->remain;
    if (remain<to_send) to_send = remain;
    target = target->parent->dest;
   }
   
  // Record the sent flow...
   this->max_flow += to_send;

 // Iterate the nodes and adjust the flow as needed, creating orphans as we go...
  Node * orphans = NULL;
  
  // The collision link...
   link->remain -= to_send;
   link->other->remain += to_send;
   
  // The source tree...
   HalfLink * half_edge = link->other;
   while ((half_edge->dest-this->vertex)!=this->source)
   {
    // Move to next (Done first because we start on the link)...
     half_edge = half_edge->dest->parent;
     
    // Send the flow...
     half_edge->remain += to_send;
     half_edge->other->remain -= to_send;
    
    // Check if we have created an orphan...
     if (half_edge->other->remain<1e-12)
     {
      half_edge->other->dest->next_orphan = orphans;
      orphans = half_edge->other->dest;
      orphans->parent = NULL;
     }
   }
   
  // The sink tree...
   half_edge = link;
   while ((half_edge->dest-this->vertex)!=this->sink)
   {
    // Move to next (Done first because we start on the link)...
     half_edge = half_edge->dest->parent;
    
    // Send the flow...
     half_edge->remain -= to_send;
     half_edge->other->remain += to_send;
     
    // Check if we have created an orphan...
     if (half_edge->remain<1e-12)
     {
      half_edge->other->dest->next_orphan = orphans;
      orphans = half_edge->other->dest;
      orphans->parent = NULL;
     }
   }

 // Return the orphan list...
  return orphans;
}


static void MaxFlow_adopt_orphans(MaxFlow * this, Node * orphans, int valid)
{
 while (orphans!=NULL)
 {
  // Fetch the current orphan...
   Node * target = orphans;
   orphans = orphans->next_orphan;
   
  // Check if there is an easy solution - parent it to another node in the same tree that has some spare capacity...
   HalfLink * half_edge = target->first;
   HalfLink * best = NULL;
   int bestDepth = -1;
   while (half_edge)
   {
    // Determine if the edge could point at a new parent...
     if (half_edge->dest->owner==target->owner)
     {
      float can_send = (target->owner==-1) ? (half_edge->other->remain) : (half_edge->remain);
      int depth = MaxFlow_tree_depth(this, half_edge->dest, valid);
      
      if ((can_send>1e-12)&&(depth!=-1))
      {
       // We have a viable edge - see if its better than the existing one...
        if ((best==NULL)||(depth<bestDepth))
        {
         best = half_edge;
         bestDepth = depth;
        }
      }
     }
    
    // Move to next...
     half_edge = half_edge->next;
   }
   
   if (best!=NULL)
   {
    target->parent = best;
    continue;
   }

  // We don't have an easy solution...
   // Iterate its neighbours - some are going to need to become active, some orphans, some just watch nonchalantly...
    half_edge = target->first;
    while (half_edge)
    {
     // We only care about vertices that are in this tree...
      if (target->owner==half_edge->dest->owner)
      {
       // Check if it is a child of the target - if so we need to orphan it...
        if (half_edge->dest->parent==half_edge->other)
        {
         half_edge->dest->parent = NULL;
         half_edge->dest->next_orphan = orphans;
         orphans = half_edge->dest;
        }
     
       // Check if it needs to be made active...
        float can_send = (target->owner==-1) ? (half_edge->other->remain) : (half_edge->remain);
        if (can_send>1e-12)
        {
         MaxFlow_add_active(this, half_edge->dest);
        }
      }
      
     // Move to next...
      half_edge = half_edge->next; 
    }
   
   // Free the node...
    target->owner = 0;
    MaxFlow_rem_active(target);
 }
}



static void MaxFlow_solve(MaxFlow * this)
{
 int i;

 // Reset the parent and owner variables...
  Node * vertex = this->vertex;
  for (i=0; i<this->vertex_count; i++,vertex++)
  {
   vertex->parent = NULL;
   vertex->owner = 0;
   vertex->depth_valid = 0;
  }
 
 // Setup the source and sink...
  this->vertex[this->source].owner = -1;
  MaxFlow_add_active(this, &this->vertex[this->source]);
  
  this->vertex[this->sink].owner = 1;
  MaxFlow_add_active(this, &this->vertex[this->sink]);
 
 // Iterate sending more flow from the source to the sink until no more can be sent...
  this->max_flow = 0.0;
  int valid = 0;
  
  while (1)
  {
   // Grow the trees until a collision occurs...
    HalfLink * link = MaxFlow_grow_trees(this);
    if (link==NULL) break; // No more tree growth possible - we are done.
   
   // Use the collision to send some pureed unicorn from the source to the sink. Omnomnomnom. We get a list of orphans back from this operation...
    Node * orphans = MaxFlow_fill_route(this, link);
   
   // Adopt or free the orphans...
    valid += 1;
    MaxFlow_adopt_orphans(this, orphans, valid);
  }
}


static PyObject * MaxFlow_solve_py(MaxFlow * self, PyObject * args)
{
 // Run the algorithm...
  MaxFlow_solve(self);
  
 // Return None...
  Py_INCREF(Py_None);
  return Py_None;
}



static int MaxFlow_get_side(MaxFlow * this, int vertex)
{
 return this->vertex[vertex].owner;
}



static void MaxFlow_store_side(MaxFlow * this, int * out, int * source, int * sink, size_t elem_size, size_t out_step, size_t source_step, size_t sink_step)
{
 int i;
 Node * target = this->vertex;
 for (i=0; i<this->vertex_count; i++, target++)
 {
  // Store from the relevant array...
   if (target->owner==-1)
   {
    if (source!=NULL) memcpy(out, source, elem_size);
   }
   else
   {
    if (sink!=NULL) memcpy(out, sink, elem_size); 
   }
   
  // Move all three arrays on with their respective steps...
   out = (int*)(void*)((char*)(void*)out + out_step);
   if (source!=NULL) source = (int*)(void*)((char*)(void*)source + source_step);
   if (sink!=NULL) sink = (int*)(void*)((char*)(void*)sink + sink_step);
 }
}

void MaxFlow_store_side_range(MaxFlow * this, int start, int length, int * out, int * source, int * sink, size_t elem_size, size_t out_step, size_t source_step, size_t sink_step)
{
 int i;
 Node * target = this->vertex + start;
 for (i=0; i<length; i++, target++)
 {
  // Store from the relevant array...
   if (target->owner==-1)
   {
    if (source!=NULL) memcpy(out, source, elem_size);
   }
   else
   {
    if (sink!=NULL) memcpy(out, sink, elem_size); 
   }
   
  // Move all three arrays on with their respective steps...
   out = (int*)(void*)((char*)(void*)out + out_step);
   if (source!=NULL) source = (int*)(void*)((char*)(void*)source + source_step);
   if (sink!=NULL) sink = (int*)(void*)((char*)(void*)sink + sink_step);
 }
}


static PyObject * MaxFlow_store_side_py(MaxFlow * self, PyObject * args)
{
 // Extract the arguments - the first (output) will be a numpy array, but the other two have many acceptable options...
  PyArrayObject * out;
  PyObject * source;
  PyObject * sink;
  if (!PyArg_ParseTuple(args, "O!OO", &PyArray_Type, &out, &source, &sink)) return NULL;
  
 // Validate the output...
  if (out->nd!=1)
  {
   PyErr_SetString(PyExc_TypeError, "Output numpy array must be one dimensional");
   return NULL;
  }
  
  if (out->dimensions[0]!=self->vertex_count)
  {
   PyErr_SetString(PyExc_IndexError, "Output numpy array's length must match the number of vertices.");
   return NULL;
  }
  
  int * out_data = (int*)(void*)out->data;
  size_t out_step = out->strides[0];
  size_t elem_size = out->descr->elsize;
  
 // Verify the source, and setup its passthrough...
  int * source_data;
  size_t source_step;
  int source_buf[4];
  
  if (source==Py_None)
  {
   source_data = NULL;
   source_step = 0;
  }
  else
  {
   if (PyInt_Check(source))
   {
    memset(source_buf, 0, sizeof(int)*4);
    *(long*)(void*)source_buf = PyInt_AsLong(source);
    
    source_data = source_buf;
    source_step = 0;
   }
   else
   {
    if (PyFloat_Check(source))
    {
     if (elem_size==sizeof(float))
     {
      *(float*)(void*)source_buf = PyFloat_AsDouble(source);
     }
     else
     {
      if (elem_size==sizeof(double))
      {
       *(double*)(void*)source_buf = PyFloat_AsDouble(source);
      }
      else
      {
       PyErr_SetString(PyExc_TypeError, "Unsuported floating point type in out.");
       return NULL;
      }
     }
      
     source_data = source_buf;
     source_step = 0;
    }
    else
    {
     if (PyObject_TypeCheck(source, &PyArray_Type))
     {
      PyArrayObject * con = (PyArrayObject*)source;
      
      if (elem_size!=con->descr->elsize)
      {
       PyErr_SetString(PyExc_TypeError, "source array and out array have different element sizes");
       return NULL;
      }
      
      source_data = (int*)(void*)con->data;
      source_step = con->strides[0];
     }
     else
     {
      PyErr_SetString(PyExc_TypeError, "Type of source not supported by method.");
      return NULL;
     }
    }
   }
  }

 // Verify the sink, and setup its passthrough...
  int * sink_data;
  size_t sink_step;
  int sink_buf[4];
  
  if (sink==Py_None)
  {
   sink_data = NULL;
   sink_step = 0;
  }
  else
  {
   if (PyInt_Check(sink))
   {
    memset(sink_buf, 0, sizeof(int)*4);
    *(long*)(void*)sink_buf = PyInt_AsLong(sink);
    
    sink_data = sink_buf;
    sink_step = 0;
   }
   else
   {
    if (PyFloat_Check(sink))
    {
     if (elem_size==sizeof(float))
     {
      *(float*)(void*)sink_buf = PyFloat_AsDouble(sink);
     }
     else
     {
      if (elem_size==sizeof(double))
      {
       *(double*)(void*)sink_buf = PyFloat_AsDouble(sink);
      }
      else
      {
       PyErr_SetString(PyExc_TypeError, "Unsuported floating point type in out.");
       return NULL;
      }
     }
      
     sink_data = sink_buf;
     sink_step = 0;
    }
    else
    {
     if (PyObject_TypeCheck(sink, &PyArray_Type))
     {
      PyArrayObject * con = (PyArrayObject*)sink;
      
      if (elem_size!=con->descr->elsize)
      {
       PyErr_SetString(PyExc_TypeError, "sink array and out array have different element sizes");
       return NULL;
      }
      
      sink_data = (int*)(void*)con->data;
      sink_step = con->strides[0];
     }
     else
     {
      PyErr_SetString(PyExc_TypeError, "Type of sink not supported by method.");
      return NULL;
     }
    }
   }
  }
 
 // Call through to the store_side method...
  MaxFlow_store_side(self, out_data, source_data, sink_data, elem_size, out_step, source_step, sink_step);
  
 // Return None...
  Py_INCREF(Py_None);
  return Py_None;
}

static PyObject * MaxFlow_store_side_range_py(MaxFlow * self, PyObject * args)
{
 // Extract the arguments - the first (output) will be a numpy array, but the other two have many acceptable options...
  int start;
  PyArrayObject * out;
  PyObject * source;
  PyObject * sink;
  if (!PyArg_ParseTuple(args, "iO!OO", &start, &PyArray_Type, &out, &source, &sink)) return NULL;
  
 // Validate the output...
  if (out->nd!=1)
  {
   PyErr_SetString(PyExc_TypeError, "Output numpy array must be one dimensional");
   return NULL;
  }
  
  int * out_data = (int*)(void*)out->data;
  size_t out_step = out->strides[0];
  size_t elem_size = out->descr->elsize;
  
 // Verify the source, and setup its passthrough...
  int * source_data;
  size_t source_step;
  int source_buf[4];
  
  if (source==Py_None)
  {
   source_data = NULL;
   source_step = 0;
  }
  else
  {
   if (PyInt_Check(source))
   {
    memset(source_buf, 0, sizeof(int)*4);
    *(long*)(void*)source_buf = PyInt_AsLong(source);
    
    source_data = source_buf;
    source_step = 0;
   }
   else
   {
    if (PyFloat_Check(source))
    {
     if (elem_size==sizeof(float))
     {
      *(float*)(void*)source_buf = PyFloat_AsDouble(source);
     }
     else
     {
      if (elem_size==sizeof(double))
      {
       *(double*)(void*)source_buf = PyFloat_AsDouble(source);
      }
      else
      {
       PyErr_SetString(PyExc_TypeError, "Unsuported floating point type in out.");
       return NULL;
      }
     }
      
     source_data = source_buf;
     source_step = 0;
    }
    else
    {
     if (PyObject_TypeCheck(source, &PyArray_Type))
     {
      PyArrayObject * con = (PyArrayObject*)source;
      
      if (elem_size!=con->descr->elsize)
      {
       PyErr_SetString(PyExc_TypeError, "source array and out array have different element sizes");
       return NULL;
      }
      
      source_data = (int*)(void*)con->data;
      source_step = con->strides[0];
     }
     else
     {
      PyErr_SetString(PyExc_TypeError, "Type of source not supported by method.");
      return NULL;
     }
    }
   }
  }

 // Verify the sink, and setup its passthrough...
  int * sink_data;
  size_t sink_step;
  int sink_buf[4];
  
  if (sink==Py_None)
  {
   sink_data = NULL;
   sink_step = 0;
  }
  else
  {
   if (PyInt_Check(sink))
   {
    memset(sink_buf, 0, sizeof(int)*4);
    *(long*)(void*)sink_buf = PyInt_AsLong(sink);
    
    sink_data = sink_buf;
    sink_step = 0;
   }
   else
   {
    if (PyFloat_Check(sink))
    {
     if (elem_size==sizeof(float))
     {
      *(float*)(void*)sink_buf = PyFloat_AsDouble(sink);
     }
     else
     {
      if (elem_size==sizeof(double))
      {
       *(double*)(void*)sink_buf = PyFloat_AsDouble(sink);
      }
      else
      {
       PyErr_SetString(PyExc_TypeError, "Unsuported floating point type in out.");
       return NULL;
      }
     }
      
     sink_data = sink_buf;
     sink_step = 0;
    }
    else
    {
     if (PyObject_TypeCheck(sink, &PyArray_Type))
     {
      PyArrayObject * con = (PyArrayObject*)sink;
      
      if (elem_size!=con->descr->elsize)
      {
       PyErr_SetString(PyExc_TypeError, "sink array and out array have different element sizes");
       return NULL;
      }
      
      sink_data = (int*)(void*)con->data;
      sink_step = con->strides[0];
     }
     else
     {
      PyErr_SetString(PyExc_TypeError, "Type of sink not supported by method.");
      return NULL;
     }
    }
   }
  }
 
 // Call through to the store_side method...
  MaxFlow_store_side_range(self, start, out->dimensions[0], out_data, source_data, sink_data, elem_size, out_step, source_step, sink_step);
  
 // Return None...
  Py_INCREF(Py_None);
  return Py_None;
}



static void MaxFlow_get_unused(MaxFlow * this, int edge, float * neg, float * pos)
{
 HalfLink * target = this->half_edge + 2*edge;
 if (neg!=NULL) *neg = target[0].remain;
 if (neg!=NULL) *pos = target[1].remain;
}



static void MaxFlow_store_unused(MaxFlow * this, float * neg, float * pos, size_t neg_step, size_t pos_step)
{
 int i;
 HalfLink * target = this->half_edge;
 for (i=0; i<this->half_edge_count; i+=2, target+=2)
 {
  // Store values...
   *pos = target[0].remain; 
   *neg = target[1].remain;  
  
  // Move to next...
   neg = (float*)(void*)((char*)(void*)neg + neg_step);
   pos = (float*)(void*)((char*)(void*)pos + pos_step);
 }
}


static PyObject * MaxFlow_store_unused_py(MaxFlow * self, PyObject * args)
{
 // Extract the two numpy arrays...
  PyArrayObject * neg;
  PyArrayObject * pos;
  if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &neg, &PyArray_Type, &pos)) return NULL;
  
  if (neg->nd!=1 || pos->nd!=1)
  {
   PyErr_SetString(PyExc_TypeError, "Unused flow must be stored in one dimensional arrays");
   return NULL;
  }
  
  if (neg->dimensions[0]!=self->edge_count || pos->dimensions[0]!=self->edge_count)
  {
   PyErr_SetString(PyExc_IndexError, "Unused flow arrays are not the length of the edge count.");
   return NULL;
  }
  
  if (neg->descr->kind!='f' || neg->descr->elsize!=sizeof(float) || pos->descr->kind!='f' || pos->descr->elsize!=sizeof(float))
  {
   PyErr_SetString(PyExc_TypeError, "Unused flow must be stored in a floating point array with the same size as a 'c' float.");
   return NULL;
  }
  
 // Record the remaining flow...
  MaxFlow_store_unused(self, (float*)(void*)neg->data, (float*)(void*)pos->data, neg->strides[0], pos->strides[0]);
 
 // Return None...
  Py_INCREF(Py_None);
  return Py_None; 
}



static PyMethodDef MaxFlow_methods[] =
{
 {"resize", (PyCFunction)MaxFlow_resize_py, METH_VARARGS, "Resizes the number of vertices/edges in the object, reseting everything as it does so - simply allows you to reuse an object instead of creating a new one every time. If the new count is smaller than the previous one it keeps the old block of memory around, and can later be made larger upto the original memory blocks size without the cost of a realloc."},
 {"set_source", (PyCFunction)MaxFlow_set_source_py, METH_VARARGS, "Sets the source, using an index into the vertex array."},
 {"set_sink", (PyCFunction)MaxFlow_set_sink_py, METH_VARARGS, "Sets the sink, using an index into the vertex array."},
 {"get_source", (PyCFunction)MaxFlow_get_source_py, METH_VARARGS, "Returns the index of the source vertex."},
 {"get_sink", (PyCFunction)MaxFlow_get_sink_py, METH_VARARGS, "Returns the index of the sink vertex."},
 {"set_edges", (PyCFunction)MaxFlow_set_edges_py, METH_VARARGS, "Using two numpy vectors sets the edges, where each edge is from the vertex in the first array to the vertex in the second array."},
 {"reset_edges", (PyCFunction)MaxFlow_reset_edges_py, METH_VARARGS, "Resets the edges, so they can be overwritten using set_edges_range. Note that this is only needed if replacing the edge structure after a first run using the set_edges_range method - the set_edges method includes this implicitly as its setting all of them."},
 {"set_edges_range", (PyCFunction)MaxFlow_set_edges_range_py, METH_VARARGS, "Sets a range of edges - same as set_edges except the first parameter is a start index, and the array length indicates how many to set. You can only set each edge once - if you try and set an edge a second time nothing happens. Because of this you can't use this to partially reconfigure the edges, only to write all the edges in stages."},
 {"set_flow_cap", (PyCFunction)MaxFlow_set_flow_cap_py, METH_VARARGS, "Using two numpy floating point vectors sets the flow limit in each direction for each edge. The first array is the negative direction of flow, the second the positive direction of flow."},
 {"set_flow_cap_range", (PyCFunction)MaxFlow_set_flow_cap_range_py, METH_VARARGS, "Identical to set_flow_cap, except the first parameter is a start index in the edge array, and the length of the arrays determines how many values to write. Unlike edge construction this always writes over the current values."},
 {"solve", (PyCFunction)MaxFlow_solve_py, METH_VARARGS, "Solves to find the maximum flow, after which you can extract various results via the variables/methods. Can be called repeatedly, though note that the flow limits are lost, and need to be set each time."},
 {"store_side", (PyCFunction)MaxFlow_store_side_py, METH_VARARGS, "After solve has been called this allows you to query which side of the minimum cut each vertex is on, putting the output into an array. Actually takes an output array and then two other entities - one to be copied over when its source, one when its sink. These two entities can be (independently) None to do nothing, an integer to write an integer, a float to write a float or another 1D array, to index into"},
 {"store_side_range", (PyCFunction)MaxFlow_store_side_range_py, METH_VARARGS, "Same as store_side, except it outputs a range of values - first parameter is start, as an offset into the vertices, with the lengths of the provided array(s) indicating how many to read."},
 {"store_unused", (PyCFunction)MaxFlow_store_unused_py, METH_VARARGS, "Writes into two output arrays the remaining flow after it has been solved. These must be floating point 1D arrays of length the number of edges, the first the remaining in the negative direction, the second in the positive direction. Aligned with the original edge endpoint and flow setting."},
 {NULL}  
};



static PyTypeObject MaxFlowType =
{
 PyObject_HEAD_INIT(NULL)
 0,                              /*ob_size*/
 "maxflow_c.MaxFlow",            /*tp_name*/
 sizeof(MaxFlow),                /*tp_basicsize*/
 0,                              /*tp_itemsize*/
 (destructor)MaxFlow_dealloc_py, /*tp_dealloc*/
 0,                              /*tp_print*/
 0,                              /*tp_getattr*/
 0,                              /*tp_setattr*/
 0,                              /*tp_compare*/
 0,                              /*tp_repr*/
 0,                              /*tp_as_number*/
 0,                              /*tp_as_sequence*/
 0,                              /*tp_as_mapping*/
 0,                              /*tp_hash */
 0,                              /*tp_call*/
 0,                              /*tp_str*/
 0,                              /*tp_getattro*/
 0,                              /*tp_setattro*/
 0,                              /*tp_as_buffer*/
 Py_TPFLAGS_DEFAULT,             /*tp_flags*/
 "For solving the max-flow (min-cut) problem. You initialise with the number of vertices and the number of edges, and then set one vertex to be the source, another to be the sink. The edges are then initialised with which vertices they connect, and the maximum flow they can do in both directions. After this solve can be called to find how much flow to send over each edge to obtain the maximum flow across the graph. Once solved the total flow, which side of the minimum cut each vertex is and the remaining flow (Noting that saturated means it is on the minimum cut) of each edge. It can be run repeatedtly, though the maximum flows are lost after each run and hence need to be reset.", /* tp_doc */
 0,                              /* tp_traverse */
 0,                              /* tp_clear */
 0,                              /* tp_richcompare */
 0,                              /* tp_weaklistoffset */
 0,                              /* tp_iter */
 0,                              /* tp_iternext */
 MaxFlow_methods,                /* tp_methods */
 MaxFlow_members,                /* tp_members */
 0,                              /* tp_getset */
 0,                              /* tp_base */
 0,                              /* tp_dict */
 0,                              /* tp_descr_get */
 0,                              /* tp_descr_set */
 0,                              /* tp_dictoffset */
 0,                              /* tp_init */
 0,                              /* tp_alloc */
 MaxFlow_new_py,                 /* tp_new */
};



static PyMethodDef maxflow_c_methods[] =
{
 {NULL}
};



#ifndef PyMODINIT_FUNC
#define PyMODINIT_FUNC void
#endif

PyMODINIT_FUNC initmaxflow_c(void)
{
 PyObject * mod = Py_InitModule3("maxflow_c", maxflow_c_methods, "Provides a solver for the maximum flow/minimum cut problem.");
 import_array();

 if (PyType_Ready(&MaxFlowType) < 0) return;

 Py_INCREF(&MaxFlowType);
 PyModule_AddObject(mod, "MaxFlow", (PyObject*)&MaxFlowType);
 
 // Create the api object, that other c modules can use to access this one...
  static MaxFlowAPI api;
  api.init = MaxFlow_init;
  api.deinit = MaxFlow_deinit;
  api.resize = MaxFlow_resize;
  api.set_source = MaxFlow_set_source;
  api.set_sink = MaxFlow_set_sink;
  api.get_source = MaxFlow_get_source;
  api.get_sink = MaxFlow_get_sink;
  api.set_edges = MaxFlow_set_edges;
  api.reset_edges = MaxFlow_reset_edges;
  api.set_edge = MaxFlow_set_edge;
  api.set_edges_range = MaxFlow_set_edges_range;
  api.cap_flow = MaxFlow_cap_flow;
  api.set_flow_cap = MaxFlow_set_flow_cap;
  api.set_flow_cap_double = MaxFlow_set_flow_cap_double;
  api.set_flow_cap_range = MaxFlow_set_flow_cap_range;
  api.set_flow_cap_range_double = MaxFlow_set_flow_cap_range_double;
  api.solve = MaxFlow_solve;
  api.get_side = MaxFlow_get_side;
  api.store_side = MaxFlow_store_side;
  api.store_side_range = MaxFlow_store_side_range;
  api.get_unused = MaxFlow_get_unused;
  api.store_unused = MaxFlow_store_unused;
  
 // Register a capsule for access to api...
  PyObject * api_capsule = PyCapsule_New((void*)&api, "maxflow_c.C_API", NULL);
  if (api_capsule!=NULL) PyModule_AddObject(mod, "C_API", api_capsule);
}
