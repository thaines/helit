// Copyright 2014 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

#include <Python.h>
#include <structmember.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>



#include "gbp_c.h"



// Any precision value larger than this is assumed to be infinity, and sent through an alternate code path...
static const float infinity_and_beyond = 1e32;



// Given a node this returns its chain count, calculating it if need be...
static inline int Node_chain_count(Node * this)
{
 if (this->chain_count<0)
 {
  // Iterate the edges and count how many exist of each type - we use the pointers to define the ordering on the nodes...
   int to_past = 0;
   int to_future = 0;
   
   HalfEdge * targ = this->first;
   while (targ!=NULL)
   {
    if (targ->dest<this)
    {
     to_past += 1; 
    }
    else
    {
     to_future += 1;
    }
    
    targ = targ->next;
   }
   
  // Set the edge count to the maximum of the future/past edges counts...
   if (to_past>to_future)
   {
    this->chain_count = to_past;
   }
   else
   {
    this->chain_count = to_future;
   }
 }
 
 return this->chain_count;
}


// Given a HalfEdge returns its Edge object...
static inline Edge * HalfEdge_edge(HalfEdge * this)
{
 if (this<this->reverse)
 {
  return (Edge*)this;
 }
 else
 {
  return (Edge*)(this->reverse);
 }
}

// Returns the sign of an edge - +1 if its the forward edge, -1 if its the negative edge; happens to be the value to multiply the pmean by to get the right value for message passing...
static inline float HalfEdge_sign(HalfEdge * this)
{
 if (this<this->reverse) return 1.0;
                    else return -1.0;
}

// Returns the precision multiplied by the offset of a half edge, going from source to destination...
static inline float HalfEdge_offset_pmean(HalfEdge * this)
{
 return HalfEdge_sign(this) * HalfEdge_edge(this)->poffset;
}


// Allows you to multiply the existing offset with another one - this is equivalent to set in the first instance when its initialised to a zero precision; includes an exponent for the previous value...
static void HalfEdge_offset_mult(HalfEdge * this, float offset, float prec, float prev_weight)
{
 Edge * edge = HalfEdge_edge(this);
 edge->poffset = prev_weight*edge->poffset + HalfEdge_sign(this) * offset * prec;
 edge->diag = prev_weight*edge->diag + prec;
}

// For when you have the poffset...
static void HalfEdge_offset_mult_raw(HalfEdge * this, float poffset, float prec, float prev_weight)
{
 Edge * edge = HalfEdge_edge(this);
 edge->poffset = prev_weight*edge->poffset + HalfEdge_sign(this) * poffset;
 edge->diag = prev_weight*edge->diag + prec;
}


// Returns a new edge, as an edge with the reverse pointer set (leaves dest and next to the user to initialise)...
static HalfEdge * GBP_new_edge(GBP * this)
{
 // Get one half edge (we know we always deal with HalfEdge-s in pairs, hence why the below is safe)...
  if (this->gc==NULL)
  {
   Block * nb = (Block*)malloc(sizeof(Block) + sizeof(Edge) * this->block_size);
   nb->next = this->storage;
   this->storage = nb;
   
   int i;
   for (i=this->block_size-1; i>=0; i--)
   {
    nb->data[i].next = this->gc;
    this->gc = nb->data + i;
   }
  }

  Edge * e = this->gc;
  this->gc = e->next;
  
 // Do a basic initialisation on them...
  e->forward.reverse = &(e->backward);
  e->forward.pmean = 0.0;
  e->forward.prec = 0.0;
  
  e->backward.reverse = &(e->forward);
  e->backward.pmean = 0.0;
  e->backward.prec = 0.0;
  
  e->poffset = 0.0;
  e->diag = 0.0;
  e->co = 0.0;
  
 // Return a...
  this->edge_count += 1;
  return &(e->forward);
}


// If a requested edge exists return it, otherwise return NULL...
static inline HalfEdge * GBP_get_edge(GBP * this, int a, int b)
{
 Node * from = this->node + a;
 Node * to   = this->node + b;
  
 HalfEdge * targ = from->first;
 while (targ!=NULL)
 {
  if (targ->dest==to) return targ;
  targ = targ->next;
 }
}


// Given a pair of node indices returns the edge connecting them, noting that if it does not exist it creates it. b will be the returned half edges destination...
static HalfEdge * GBP_always_get_edge(GBP * this, int a, int b)
{
 Node * from = this->node + a;
 Node * to   = this->node + b;
  
 // Check if it already exists, and return if so...
  HalfEdge * targ = from->first;
  while (targ!=NULL)
  {
   if (targ->dest==to) return targ; 
   targ = targ->next;
  }
  
 // Does not exist, so create it...
  targ = GBP_new_edge(this);
  
  targ->dest = to;
  targ->reverse->dest = from;
  
  targ->next = from->first;
  from->first = targ;
  
  targ->reverse->next = to->first;
  to->first = targ->reverse;
  
  from->chain_count = -1;
  to->chain_count = -1;
  
 // Return the newly created edge...
  return targ;
}



void GBP_new(GBP * this, int node_count, int block_size)
{
 this->node_count = node_count;
 this->node = (Node*)malloc(this->node_count * sizeof(Node));
 
 int i;
 for (i=0; i<this->node_count; i++)
 {
  this->node[i].first = NULL;
  this->node[i].unary_pmean = 0.0;
  this->node[i].unary_prec = 0.0;
  this->node[i].pmean = 0.0;
  this->node[i].prec = 0.0;
  this->node[i].chain_count = -1;
  this->node[i].on = 1;
 }
 
 this->edge_count = 0;
 
 this->gc = NULL;
 
 this->block_size = block_size;
 this->storage = NULL;
}

void GBP_dealloc(GBP * this)
{
 free(this->node);
 
 while (this->storage!=NULL)
 {
  Block * to_die = this->storage;
  this->storage = this->storage->next;
  free(to_die);
 }
}


static PyObject * GBP_new_py(PyTypeObject * type, PyObject * args, PyObject * kwds)
{
 // Get the args...
  int node_count;
  int block_size = 512;
  if (!PyArg_ParseTuple(args, "i|i", &node_count, &block_size)) return NULL;
  
 // Allocate the object...
  GBP * self = (GBP*)type->tp_alloc(type, 0);

 // On success construct it...
  if (self!=NULL) GBP_new(self, node_count, block_size);

 // Return the new object...
  return (PyObject*)self;
}

static void GBP_dealloc_py(GBP * self)
{
 GBP_dealloc(self);
 self->ob_type->tp_free((PyObject*)self);
}



static PyTypeObject GBPType; // Preallocation for below...

static PyObject * GBP_clone_py(GBP * self, PyObject * args)
{
 // Allocate the new object...
  GBP * other = (GBP*)GBPType.tp_alloc(&GBPType, 0);
  if (other==NULL) return NULL;
  
 // Copy over all the basic details, though take care to allocate the right number of edges...
  other->node_count = self->node_count;
  other->node = (Node*)malloc(other->node_count * sizeof(Node));
 
  int i;
  for (i=0; i<other->node_count; i++)
  {
   other->node[i].first = NULL;
   other->node[i].unary_pmean = self->node[i].unary_pmean;
   other->node[i].unary_prec = self->node[i].unary_prec;
   other->node[i].pmean = self->node[i].pmean;
   other->node[i].prec = self->node[i].prec;
   other->node[i].chain_count = self->node[i].chain_count;
   other->node[i].on = self->node[i].on;
  }
 
  other->edge_count = 0;

  other->storage = (Block*)malloc(sizeof(Block) + sizeof(Edge) * self->edge_count);
  other->storage->next = NULL;

  other->gc = NULL;
  for (i=self->edge_count-1; i>=0; i--)
  {
   other->storage->data[i].next = other->gc;
   other->gc = other->storage->data + i;
  }
  
  other->block_size = self->block_size;
   
 // From here on in the object is coherant!
  
 // Now do the edges, which are extra hairy...
  for (i=0; i<self->node_count; i++)
  {
   HalfEdge * targ = self->node[i].first;
   while (targ!=NULL)
   {
    // Only want to do each edge once...
     if (targ < targ->reverse) // Check if this half edge is the forward direction - makes sure we only do each once, and allows for some assumptions below.
     {
      // Figure out the second index, create an edge...
       int j = targ->dest - self->node;
     
       Edge * e = other->gc;
       other->gc = e->next;
       other->edge_count += 1;

       e->forward.reverse = &(e->backward);
       e->backward.reverse = &(e->forward);

      // Copy over the correct values...
       e->forward.dest = other->node + j;
       e->forward.next = other->node[i].first;
       other->node[i].first = &(e->forward);
     
       e->backward.dest = other->node + i;
       e->backward.next = other->node[j].first;
       other->node[j].first = &(e->backward);
     
       e->forward.pmean = targ->pmean;
       e->forward.prec = targ->prec;
       e->backward.pmean = targ->reverse->pmean;
       e->backward.prec =  targ->reverse->prec;
       
       Edge * source = HalfEdge_edge(targ);
       e->poffset = source->poffset;
       e->diag = source->diag;
       e->co = source->co;
     }
     
    // To the next...
     targ = targ->next;
   }
  }
  
 // Return the new object...
  return (PyObject*)other;
}



// Helper function - given a numpy object to mean a range of nodes this outputs information to allow that loop to be done. Allways outputs a standard set of slice details (start, step, length), and also optionally outputs an numpy array to be checked at each position in the slice loop if its not NULL. Note that if the array is output it must be reference decrimented after use. Returns 0 on success, -1 on failure (in which case an error will have been set and arr will definitely be NULL). Note that it doesn't range check if an array is involved, Singular can be optional be passed, adn will be set nonzero if the request was a single number...
int GBP_index(GBP * this, PyObject * arg, Py_ssize_t * start, Py_ssize_t * step, Py_ssize_t * length, PyArrayObject ** arr, int * singular)
{
 if (singular!=NULL) *singular = 0;
 *arr = NULL;
 
 if (arg==NULL)
 {
  *start = 0;
  *step = 1;
  *length = this->node_count;
  
  return 0;
 }
 
 if (PyInt_Check(arg)!=0)
 {
  *start = PyNumber_AsSsize_t(arg, NULL); 
  
  if ((*start<0)||(*start>=this->node_count))
  {
   PyErr_SetString(PyExc_IndexError, "Index out of bounds.");
   return -1; 
  }
  
  *step = 1;
  *length = 1;
  if (singular!=NULL) *singular = 1;
  
  return 0;
 }
 
 if (PySlice_Check(arg)!=0)
 {
  Py_ssize_t stop;
    
  if (PySlice_GetIndicesEx((PySliceObject*)arg, this->node_count, start, &stop, step, length)!=0)
  {
   PyErr_SetString(PyExc_IndexError, "Slice doesn't play with the length.");
   return -1;
  }
  
  return 0;
 }
 
 *arr = (PyArrayObject*)PyArray_ContiguousFromAny(arg, NPY_INTP, 1, 1);
 if (*arr!=NULL)
 {
  *start = 0;
  *step = 1;
  *length = PyArray_DIMS(*arr)[0];
  
  return 0;
 }
 
 PyErr_SetString(PyExc_TypeError, "Don't know how to index with that!");
 return -1;
}



static PyObject * GBP_reset_unary_py(GBP * self, PyObject * args)
{
 // Fetch the parameter...
  PyObject * index = NULL;
  if (!PyArg_ParseTuple(args, "|O", &index)) return NULL;
  
 // Convert the input into something we can dance with...
  Py_ssize_t start;
  Py_ssize_t step;
  Py_ssize_t length;
  PyArrayObject * arr;
  
  if (GBP_index(self, index, &start, &step, &length, &arr, NULL)!=0) return NULL; 
  
 // Do the loop...
  int i, ii, iii; // I need better names for these!..
  for (i=0, ii=start; i<length; i++, ii+=step)
  {
   // Handle the array scenario...
    iii = ii;
    if (arr!=NULL)
    {
     iii = *(int*)PyArray_GETPTR1(arr, ii);
     if ((iii<0)||(iii>=self->node_count))
     {
      Py_DECREF(arr);
      PyErr_SetString(PyExc_IndexError, "Index out of bounds.");
      return NULL;
     }
    }
   
   // Store some zeroes...
    self->node[iii].unary_pmean = 0.0;
    self->node[iii].unary_prec  = 0.0;
  }
  
 // Clean up and return None...
  Py_XDECREF(arr);
  
  Py_INCREF(Py_None);
  return Py_None;
}



static PyObject * GBP_add_py(GBP * self, PyObject * args)
{
 // Fetch the parameter...
  int count = 1;
  if (!PyArg_ParseTuple(args, "|i", &count)) return NULL;
  
 // Realloc the storage, and initialise the new nodes...
  int index_first = self->node_count;
  void * old_ptr = self->node;
  
  self->node_count += count;
  void * new_ptr = realloc(old_ptr, self->node_count * sizeof(Node));
  self->node = (Node*)new_ptr;
  
  int i;
  for (i=index_first; i<self->node_count; i++)
  {
   self->node[i].first = NULL;
   self->node[i].unary_pmean = 0.0;
   self->node[i].unary_prec = 0.0;
   self->node[i].pmean = 0.0;
   self->node[i].prec = 0.0;
   self->node[i].chain_count = -1;
   self->node[i].on = 1;
  }
  
 // Loop through and correct all the pointers to nodes in the edges...
  size_t offset = new_ptr - old_ptr;
  for (i=0; i<index_first; i++)
  {
   HalfEdge * msg = self->node[i].first;
   while (msg!=NULL)
   {
    msg->dest = (Node*)((void*)msg->dest + offset);
    msg = msg->next; 
   }
  }
 
 // Return the index of the first...
  return Py_BuildValue("i", index_first);
}



static PyObject * GBP_enable_py(GBP * self, PyObject * args)
{
 // Fetch the parameter...
  PyObject * index = NULL;
  if (!PyArg_ParseTuple(args, "|O", &index)) return NULL;
  
 // Convert the input into something we can dance with...
  Py_ssize_t start;
  Py_ssize_t step;
  Py_ssize_t length;
  PyArrayObject * arr;
  
  if (GBP_index(self, index, &start, &step, &length, &arr, NULL)!=0) return NULL; 
  
 // Do the loop...
  int i, ii, iii; // I need better names for these!..
  for (i=0, ii=start; i<length; i++, ii+=step)
  {
   // Handle the array scenario...
    iii = ii;
    if (arr!=NULL)
    {
     iii = *(int*)PyArray_GETPTR1(arr, ii);
     if ((iii<0)||(iii>=self->node_count))
     {
      Py_DECREF(arr);
      PyErr_SetString(PyExc_IndexError, "Index out of bounds.");
      return NULL;
     }
    }
   
   // Switch node on...
    self->node[iii].on = 1;
  }
  
 // Clean up and return None...
  Py_XDECREF(arr);
  
  Py_INCREF(Py_None);
  return Py_None;
}


static PyObject * GBP_disable_py(GBP * self, PyObject * args)
{
 // Fetch the parameter...
  PyObject * index = NULL;
  if (!PyArg_ParseTuple(args, "|O", &index)) return NULL;
  
 // Convert the input into something we can dance with...
  Py_ssize_t start;
  Py_ssize_t step;
  Py_ssize_t length;
  PyArrayObject * arr;
  
  if (GBP_index(self, index, &start, &step, &length, &arr, NULL)!=0) return NULL; 
  
 // Do the loop...
  int i, ii, iii; // I need better names for these!..
  for (i=0, ii=start; i<length; i++, ii+=step)
  {
   // Handle the array scenario...
    iii = ii;
    if (arr!=NULL)
    {
     iii = *(int*)PyArray_GETPTR1(arr, ii);
     if ((iii<0)||(iii>=self->node_count))
     {
      Py_DECREF(arr);
      PyErr_SetString(PyExc_IndexError, "Index out of bounds.");
      return NULL;
     }
    }
   
   // Switch node off...
    self->node[iii].on = 0;
    
   // Zero out all messages exiting it, so it has no future influence...
    HalfEdge * msg = self->node[iii].first;
    while (msg!=NULL)
    {
     msg->pmean = 0.0;
     msg->prec = 0.0;
    
     msg = msg->next; 
    }
  }
  
 // Clean up and return None...
  Py_XDECREF(arr);
  
  Py_INCREF(Py_None);
  return Py_None;
}



static PyObject * GBP_reset_pairwise_py(GBP * self, PyObject * args)
{
 // Fetch the parameters...
  PyObject * index_a = NULL;
  PyObject * index_b = NULL;
  if (!PyArg_ParseTuple(args, "|OO", &index_a, &index_b)) return NULL;

 // Special case two NULLs - i.e. delete everything...
  if (index_a==NULL)
  {
   int i;
   for (i=0; i<self->node_count; i++)
   {
    while (self->node[i].first!=NULL)
    {
     HalfEdge * he = self->node[i].first;
     Edge * victim = HalfEdge_edge(he);
     self->node[i].first = he->next;
     
     if (he < he->reverse) // Only delete each pair of half edges once by only doing pointers in the forward direction and ignoring pointers in the backwards direction.
     {
      victim->next = self->gc;
      self->gc = victim;
     }
    }
   }
   
   self->edge_count = 0;
   
   Py_INCREF(Py_None);
   return Py_None;
  }
  
 // Interpret the index_a indices...
  Py_ssize_t start_a;
  Py_ssize_t step_a;
  Py_ssize_t length_a;
  PyArrayObject * arr_a;
  
  if (GBP_index(self, index_a, &start_a, &step_a, &length_a, &arr_a, NULL)!=0) return NULL; 

 // Special case one null, i.e. terminate all edges leaving a given node...
  if (index_b==NULL)
  {
   int i, ii;
   for (i=0, ii=start_a; i<length_a; i++,ii+=step_a)
   {
    // Handle if array indexing is occuring...
     int iii = ii;
     if (arr_a!=NULL)
     {
      iii = *(int*)PyArray_GETPTR1(arr_a, ii);
      if ((iii<0)||(iii>=self->node_count))
      {
       Py_DECREF(arr_a);
       PyErr_SetString(PyExc_IndexError, "Index out of bounds.");
       return NULL;
      }
     }
     
    // Loop and terminate each half in turn, taking care to throw its partner off a cliff at the same time...
     Node * targ = self->node + iii;
     while (targ->first!=NULL)
     {
      // Remove its partner...
       Node * partner = targ->first->dest;
       if (targ->first->reverse==partner->first)
       {
        partner->first = partner->first->next;
       }
       else
       {
        HalfEdge * curr = partner->first;
        while (curr->next!=targ->first->reverse)
        {
         curr = curr->next; 
        }
        
        curr->next = curr->next->next;
       }
       
      // Remove it...
       HalfEdge * he = targ->first;
       targ->first = he->next;
       
      // Dump the edge into the rubish, update the edge count...
       Edge * victim = HalfEdge_edge(he);
       victim->next = self->gc;
       self->gc = victim;
       
       self->edge_count -= 1;
     }
   }
   
   Py_XDECREF(arr_a);
 
   Py_INCREF(Py_None);
   return Py_None;
  }
 
 // Interpret the index_b indices...
  Py_ssize_t start_b;
  Py_ssize_t step_b;
  Py_ssize_t length_b;
  PyArrayObject * arr_b;
  
  if (GBP_index(self, index_b, &start_b, &step_b, &length_b, &arr_b, NULL)!=0)
  {
   Py_XDECREF(arr_a);
   return NULL; 
  }
  
  if (length_a!=length_b)
  {
   Py_XDECREF(arr_a);
   Py_XDECREF(arr_b);
   
   PyErr_SetString(PyExc_IndexError, "Pairwise reset when given two lists expects them to be the same length.");
   return NULL; 
  }
 
 // Do the loop - above means we only have to code one double loop...
  int i, ii, iii;  
  for (i=0, ii=start_a; i<length_a; i++,ii+=step_a)
  {
   iii = ii;
   if (arr_a!=NULL)
   {
    iii = *(int*)PyArray_GETPTR1(arr_a, ii);
    if ((iii<0)||(iii>=self->node_count))
    {
     Py_DECREF(arr_a);
     Py_XDECREF(arr_b);
     PyErr_SetString(PyExc_IndexError, "Index out of bounds.");
     return NULL;
    }
   }
   
   Node * targ_a = self->node + iii;
   
   int jj = start_b + i * step_b;
   int jjj = jj;
   if (arr_b!=NULL)
   {
    jjj = *(int*)PyArray_GETPTR1(arr_b, jj);
    if ((jjj<0)||(jjj>=self->node_count))
    {
     Py_XDECREF(arr_a);
     Py_DECREF(arr_b);
     PyErr_SetString(PyExc_IndexError, "Index out of bounds.");
     return NULL;
    }
   }
    
   if (jjj==iii) continue; // Don't error out as that would be inconveniant - just skip.
   
   Node * targ_b = self->node + jjj;
    
   // Ok - we have two nodes - targ_a and targ_b - see if we can find a pair of half edges that make up an edge between them, and terminate if found...
    // Find and remove from targ_a - if it does not exist then we have nothing to do and can stop...
     HalfEdge * atob = NULL;
     if (targ_a->first==NULL) continue;
     if (targ_a->first->dest==targ_b)
     {
      atob = targ_a->first;
      targ_a->first = atob->next;
     }
     else
     {
      HalfEdge * t = targ_a->first;
      while ((t->next!=NULL)&&(t->next->dest!=targ_b))
      {
       t = t->next;  
      }
       
      if (t->next!=NULL)
      {
       atob = t->next;
       t->next = atob->next;
      }
     }
     if (atob==NULL) continue;
      
    // We know there must be one in targ_b - simpler find and remove...
     HalfEdge * btoa = NULL;
     if (targ_b->first->dest==targ_a)
     {
      btoa = targ_b->first;
      targ_b->first = btoa->next;
     }
     else
     {
      HalfEdge * t = targ_b->first;
      while (t->next->dest!=targ_a)
      {
       t = t->next;  
      }
       
      btoa = t->next;
      t->next = btoa->next;
     }
      
    // Deposite them both down the garbage chute; decriment the edge count...
     Edge * victim = HalfEdge_edge(atob);
     victim->next = self->gc;
     self->gc = victim;
       
     self->edge_count -= 1;
  }
 
 // Clean up and return None...
  Py_XDECREF(arr_a);
  Py_XDECREF(arr_b);
 
  Py_INCREF(Py_None);
  return Py_None;
}



static PyObject * GBP_unary_py(GBP * self, PyObject * args, PyObject * kw)
{
 // We expect three parameters...
  PyObject * index;
  PyObject * mean_obj;
  PyObject * prec_obj;
  float prev_weight = 1.0;
  
  static char * kw_list[] = {"index", "mean", "prec", "prev_exp", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOO|f", kw_list, &index, &mean_obj, &prec_obj, &prev_weight)) return NULL;
  
 // Interprete all of the inputs...
  Py_ssize_t start;
  Py_ssize_t step;
  Py_ssize_t length;
  PyArrayObject * arr;
  
  if (GBP_index(self, index, &start, &step, &length, &arr, NULL)!=0) return NULL;
  
  PyArrayObject * mean = NULL;
  if ((PyInt_Check(mean_obj)==0)&&(PyFloat_Check(mean_obj)==0))
  {
   mean = (PyArrayObject*)PyArray_ContiguousFromAny(mean_obj, NPY_DOUBLE, 1, 1);
   if (mean==NULL)
   {
    Py_XDECREF(arr);
    return NULL;
   }
  }
  
  PyArrayObject * prec = NULL;
  if ((PyInt_Check(prec_obj)==0)&&(PyFloat_Check(prec_obj)==0))
  {
   prec = (PyArrayObject*)PyArray_ContiguousFromAny(prec_obj, NPY_DOUBLE, 1, 1); 
   if (prec==NULL)
   {
    Py_XDECREF(arr);
    Py_XDECREF(mean);
    return NULL;
   }
  }
  
 // Loop through and set the required values...
  int i, ii, iii;
  float m = (mean==NULL) ? PyFloat_AsDouble(mean_obj) : 0.0;
  float p = (prec==NULL) ? PyFloat_AsDouble(prec_obj) : 0.0;
  
  for (i=0,ii=start; i<length; i++,ii+=step)
  {
   // Handle array indexing...
    iii = ii;
    if (arr!=NULL)
    {
     iii = *(int*)PyArray_GETPTR1(arr, ii);
     if ((iii<0)||(iii>=self->node_count))
     {
      Py_DECREF(arr);
      Py_XDECREF(mean);
      Py_XDECREF(prec);
      PyErr_SetString(PyExc_IndexError, "Index out of bounds.");
      return NULL;
     }
    }
    
   // Extract the mean and precision...
    if (mean!=NULL)
    {
     m = *(double*)PyArray_GETPTR1(mean, i % PyArray_DIMS(mean)[0]);
    }
    
    if (prec!=NULL)
    {
     p = *(double*)PyArray_GETPTR1(prec, i % PyArray_DIMS(prec)[0]);
    }
   
   // Store the extracted values...
    if (p>infinity_and_beyond) // For infinity its a straight replace, using mean rather than pmean.
    {
     self->node[iii].unary_pmean = m;
     self->node[iii].unary_prec = p;
    }
    else
    {
     self->node[iii].unary_pmean = prev_weight*self->node[iii].unary_pmean + m * p;
     self->node[iii].unary_prec  = prev_weight*self->node[iii].unary_prec + p;
    }
  }

 // Clean up and return None...
  Py_XDECREF(mean);
  Py_XDECREF(prec);
 
  Py_INCREF(Py_None);
  return Py_None;
}


static PyObject * GBP_unary_raw_py(GBP * self, PyObject * args, PyObject * kw)
{
 // We expect three parameters...
  PyObject * index;
  PyObject * pmean_obj;
  PyObject * prec_obj;
  float prev_weight = 1.0;
  
  static char * kw_list[] = {"index", "pmean", "prec", "prev_exp", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOO|f", kw_list, &index, &pmean_obj, &prec_obj, &prev_weight)) return NULL;
  
 // Interprete all of the inputs...
  Py_ssize_t start;
  Py_ssize_t step;
  Py_ssize_t length;
  PyArrayObject * arr;
  
  if (GBP_index(self, index, &start, &step, &length, &arr, NULL)!=0) return NULL;
  
  PyArrayObject * pmean = NULL;
  if ((PyInt_Check(pmean_obj)==0)&&(PyFloat_Check(pmean_obj)==0))
  {
   pmean = (PyArrayObject*)PyArray_ContiguousFromAny(pmean_obj, NPY_DOUBLE, 1, 1);
   if (pmean==NULL)
   {
    Py_XDECREF(arr);
    return NULL;
   }
  }
  
  PyArrayObject * prec = NULL;
  if ((PyInt_Check(prec_obj)==0)&&(PyFloat_Check(prec_obj)==0))
  {
   prec = (PyArrayObject*)PyArray_ContiguousFromAny(prec_obj, NPY_DOUBLE, 1, 1); 
   if (prec==NULL)
   {
    Py_XDECREF(arr);
    Py_XDECREF(pmean);
    return NULL;
   }
  }
  
 // Loop through and set the required values...
  int i, ii, iii;
  float pm = (pmean==NULL) ? PyFloat_AsDouble(pmean_obj) : 0.0;
  float p = (prec==NULL) ? PyFloat_AsDouble(prec_obj) : 0.0;
  
  for (i=0,ii=start; i<length; i++,ii+=step)
  {
   // Handle array indexing...
    iii = ii;
    if (arr!=NULL)
    {
     iii = *(int*)PyArray_GETPTR1(arr, ii);
     if ((iii<0)||(iii>=self->node_count))
     {
      Py_DECREF(arr);
      Py_XDECREF(pmean);
      Py_XDECREF(prec);
      PyErr_SetString(PyExc_IndexError, "Index out of bounds.");
      return NULL;
     }
    }
    
   // Extract the mean and precision...
    if (pmean!=NULL)
    {
     pm = *(double*)PyArray_GETPTR1(pmean, i % PyArray_DIMS(pmean)[0]);
    }
    
    if (prec!=NULL)
    {
     p = *(double*)PyArray_GETPTR1(prec, i % PyArray_DIMS(prec)[0]);
    }
   
   // Store the extracted values...
    if (p>infinity_and_beyond)
    {
     self->node[iii].unary_pmean = pm;
     self->node[iii].unary_prec = p;
    }
    else
    {
     self->node[iii].unary_pmean = prev_weight*self->node[iii].unary_pmean + pm;
     self->node[iii].unary_prec  = prev_weight*self->node[iii].unary_prec + p;
    }
  }

 // Clean up and return None...
  Py_XDECREF(pmean);
  Py_XDECREF(prec);
 
  Py_INCREF(Py_None);
  return Py_None;
}


static PyObject * GBP_unary_sd_py(GBP * self, PyObject * args, PyObject * kw)
{
 // We expect three parameters...
  PyObject * index;
  PyObject * mean_obj;
  PyObject * sd_obj;
  float prev_weight = 1.0;
  
  static char * kw_list[] = {"index", "mean", "sd", "prev_exp", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOO|f", kw_list, &index, &mean_obj, &sd_obj, &prev_weight)) return NULL;
  
 // Interprete all of the inputs...
  Py_ssize_t start;
  Py_ssize_t step;
  Py_ssize_t length;
  PyArrayObject * arr;
  
  if (GBP_index(self, index, &start, &step, &length, &arr, NULL)!=0) return NULL;
  
  PyArrayObject * mean = NULL;
  if ((PyInt_Check(mean_obj)==0)&&(PyFloat_Check(mean_obj)==0))
  {
   mean = (PyArrayObject*)PyArray_ContiguousFromAny(mean_obj, NPY_DOUBLE, 1, 1);
   if (mean==NULL)
   {
    Py_XDECREF(arr);
    return NULL;
   }
  }
  
  PyArrayObject * sd = NULL;
  if ((PyInt_Check(sd_obj)==0)&&(PyFloat_Check(sd_obj)==0))
  {
   sd = (PyArrayObject*)PyArray_ContiguousFromAny(sd_obj, NPY_DOUBLE, 1, 1); 
   if (sd==NULL)
   {
    Py_XDECREF(arr);
    Py_XDECREF(mean);
    return NULL;
   }
  }
  
 // Loop through and set the required values...
  int i, ii, iii;
  float m = (mean==NULL) ? PyFloat_AsDouble(mean_obj) : 0.0;
  float d = (sd==NULL) ? PyFloat_AsDouble(sd_obj) : 0.0;
  
  for (i=0,ii=start; i<length; i++,ii+=step)
  {
   // Handle array indexing...
    iii = ii;
    if (arr!=NULL)
    {
     iii = *(int*)PyArray_GETPTR1(arr, ii);
     if ((iii<0)||(iii>=self->node_count))
     {
      Py_DECREF(arr);
      Py_XDECREF(mean);
      Py_XDECREF(sd);
      PyErr_SetString(PyExc_IndexError, "Index out of bounds.");
      return NULL;
     }
    }
    
   // Extract the mean and precision...
    if (mean!=NULL)
    {
     m = *(double*)PyArray_GETPTR1(mean, i % PyArray_DIMS(mean)[0]);
    }
    
    if (sd!=NULL)
    {
     d = *(double*)PyArray_GETPTR1(sd, i % PyArray_DIMS(sd)[0]);
    }
   
   // Store the extracted values...
    float var = d*d;
    float prec = 1.0/var;
    if (prec>infinity_and_beyond)
    {
     self->node[iii].unary_pmean = m;
     self->node[iii].unary_prec = prec;
    }
    else
    {
     self->node[iii].unary_pmean = prev_weight*self->node[iii].unary_pmean + m/var;
     self->node[iii].unary_prec  = prev_weight*self->node[iii].unary_prec + prec;
    }
  }

 // Clean up and return None...
  Py_XDECREF(mean);
  Py_XDECREF(sd);
 
  Py_INCREF(Py_None);
  return Py_None;
}



static PyObject * GBP_pairwise_py(GBP * self, PyObject * args, PyObject * kw)
{
 // Three or four parameters please, with keywords as optional...
  PyObject * index_from;
  PyObject * index_to;
  PyObject * offset_obj;
  PyObject * prec_obj = NULL;
  float prev_weight = 1.0;
  
  static char * kw_list[] = {"from", "to", "offset", "prec", "prev_exp", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOO|Of", kw_list, &index_from, &index_to, &offset_obj, &prec_obj, &prev_weight)) return NULL;
  
  if (prec_obj==Py_None) prec_obj = NULL;

 // Analyse the two node indices...
  Py_ssize_t start_from;
  Py_ssize_t step_from;
  Py_ssize_t length_from;
  PyArrayObject * arr_from;
  if (GBP_index(self, index_from, &start_from, &step_from, &length_from, &arr_from, NULL)!=0)
  {
   return NULL;
  }
  
  Py_ssize_t start_to;
  Py_ssize_t step_to;
  Py_ssize_t length_to;
  PyArrayObject * arr_to;
  if (GBP_index(self, index_to, &start_to, &step_to, &length_to, &arr_to, NULL)!=0)
  {
   Py_XDECREF(arr_from);
   return NULL;
  }
  
  if (length_from!=length_to)
  {
   Py_XDECREF(arr_from);
   Py_XDECREF(arr_to);
   
   PyErr_SetString(PyExc_IndexError, "from and to must have the same number of node indices");
   return NULL;
  }

 // Now offset and precision...
  PyArrayObject * offset = NULL;
  if ((PyInt_Check(offset_obj)==0)&&(PyFloat_Check(offset_obj)==0))
  {
   offset = (PyArrayObject*)PyArray_ContiguousFromAny(offset_obj, NPY_DOUBLE, 1, 1);
   if (offset==NULL)
   {
    Py_XDECREF(arr_from);
    Py_XDECREF(arr_to);
    return NULL;
   }
  }
   
  PyArrayObject * prec = NULL;
  if ((prec_obj!=NULL) && (PyInt_Check(prec_obj)==0) && (PyFloat_Check(prec_obj)==0))
  {
   prec = (PyArrayObject*)PyArray_ContiguousFromAny(prec_obj, NPY_DOUBLE, 1, 1);
   if (prec==NULL)
   {
    Py_XDECREF(arr_from);
    Py_XDECREF(arr_to);
    Py_XDECREF(offset);
    return NULL;
   }
  }
  
 // Loop through and set the required values...
  int i, from_ii, to_ii, from_iii, to_iii;
  float os = (offset==NULL) ? PyFloat_AsDouble(offset_obj) : 0.0;
  float pr = ((prec==NULL)&&(prec_obj!=NULL))   ? PyFloat_AsDouble(prec_obj)   : 0.0;
  
  for (i=0,from_ii=start_from,to_ii=start_to; i<length_to; i++,from_ii+=step_from,to_ii+=step_to)
  {
   // Handle array indexing for both from and to...
    from_iii = from_ii;
    if (arr_from!=NULL)
    {
     from_iii = *(int*)PyArray_GETPTR1(arr_from, from_ii);
     if ((from_iii<0)||(from_iii>=self->node_count))
     {
      Py_DECREF(arr_from);
      Py_XDECREF(arr_to);
      Py_XDECREF(offset);
      Py_XDECREF(prec);
      PyErr_SetString(PyExc_IndexError, "Index out of bounds.");
      return NULL;
     }
    }
    
    to_iii = to_ii;
    if (arr_to!=NULL)
    {
     to_iii = *(int*)PyArray_GETPTR1(arr_to, to_ii);
     if ((to_iii<0)||(to_iii>=self->node_count))
     {
      Py_XDECREF(arr_from);
      Py_DECREF(arr_to);
      Py_XDECREF(offset);
      Py_XDECREF(prec);
      PyErr_SetString(PyExc_IndexError, "Index out of bounds.");
      return NULL;
     }
    }
    
   // Fetch the offset and precision values as needed...
    if (offset!=NULL)
    {
     os = *(double*)PyArray_GETPTR1(offset, i % PyArray_DIMS(offset)[0]);
    }
    
    if (prec!=NULL)
    {
     pr = *(double*)PyArray_GETPTR1(prec, i % PyArray_DIMS(prec)[0]);
    }
    
   // Test the user is not stupid...
    if (from_iii==to_iii)
    {
     Py_XDECREF(arr_from);
     Py_XDECREF(arr_to);
     Py_XDECREF(offset);
     Py_XDECREF(prec);
     PyErr_SetString(PyExc_IndexError, "Edges connecting a node to itself are not permitted");
     return NULL;
    }
    
   // Fetch the relevant edge, creating it if need be...
    HalfEdge * targ = GBP_always_get_edge(self, from_iii, to_iii);
    
   // Apply the update...
    if (prec_obj!=NULL)
    {
     // Offset with precision case...
      HalfEdge_offset_mult(targ, os, pr, prev_weight);
    }
    else
    {
     // Precision only case...
     float * co = &(HalfEdge_edge(targ)->co);
     *co = (*co) * prev_weight + os;
    }
  }
  
 // Cleanup and return None...
  Py_XDECREF(arr_from);
  Py_XDECREF(arr_to);
  Py_XDECREF(offset);
  Py_XDECREF(prec);
 
  Py_INCREF(Py_None);
  return Py_None;
}


static PyObject * GBP_pairwise_raw_py(GBP * self, PyObject * args, PyObject * kw)
{
 // Three or four parameters please, with keywords as optional...
  PyObject * index_from;
  PyObject * index_to;
  PyObject * poffset_obj;
  PyObject * prec_obj = NULL;
  float prev_weight = 1.0;
  
  static char * kw_list[] = {"from", "to", "poffset", "prec", "prev_exp", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOO|Of", kw_list, &index_from, &index_to, &poffset_obj, &prec_obj, &prev_weight)) return NULL;
  
  if (prec_obj==Py_None) prec_obj = NULL;

 // Analyse the two node indices...
  Py_ssize_t start_from;
  Py_ssize_t step_from;
  Py_ssize_t length_from;
  PyArrayObject * arr_from;
  if (GBP_index(self, index_from, &start_from, &step_from, &length_from, &arr_from, NULL)!=0)
  {
   return NULL;
  }
  
  Py_ssize_t start_to;
  Py_ssize_t step_to;
  Py_ssize_t length_to;
  PyArrayObject * arr_to;
  if (GBP_index(self, index_to, &start_to, &step_to, &length_to, &arr_to, NULL)!=0)
  {
   Py_XDECREF(arr_from);
   return NULL;
  }
  
  if (length_from!=length_to)
  {
   Py_XDECREF(arr_from);
   Py_XDECREF(arr_to);
   
   PyErr_SetString(PyExc_IndexError, "from and to must have the same number of node indices");
   return NULL;
  }

 // Now offset and precision...
  PyArrayObject * poffset = NULL;
  if ((PyInt_Check(poffset_obj)==0)&&(PyFloat_Check(poffset_obj)==0))
  {
   poffset = (PyArrayObject*)PyArray_ContiguousFromAny(poffset_obj, NPY_DOUBLE, 1, 1);
   if (poffset==NULL)
   {
    Py_XDECREF(arr_from);
    Py_XDECREF(arr_to);
    return NULL;
   }
  }
   
  PyArrayObject * prec = NULL;
  if ((prec_obj!=NULL) && (PyInt_Check(prec_obj)==0) && (PyFloat_Check(prec_obj)==0))
  {
   prec = (PyArrayObject*)PyArray_ContiguousFromAny(prec_obj, NPY_DOUBLE, 1, 1);
   if (prec==NULL)
   {
    Py_XDECREF(arr_from);
    Py_XDECREF(arr_to);
    Py_XDECREF(poffset);
    return NULL;
   }
  }
  
 // Loop through and set the required values...
  int i, from_ii, to_ii, from_iii, to_iii;
  float pos = (poffset==NULL) ? PyFloat_AsDouble(poffset_obj) : 0.0;
  float pr  = ((prec==NULL)&&(prec_obj!=NULL))   ? PyFloat_AsDouble(prec_obj)   : 0.0;
  
  for (i=0,from_ii=start_from,to_ii=start_to; i<length_to; i++,from_ii+=step_from,to_ii+=step_to)
  {
   // Handle array indexing for both from and to...
    from_iii = from_ii;
    if (arr_from!=NULL)
    {
     from_iii = *(int*)PyArray_GETPTR1(arr_from, from_ii);
     if ((from_iii<0)||(from_iii>=self->node_count))
     {
      Py_DECREF(arr_from);
      Py_XDECREF(arr_to);
      Py_XDECREF(poffset);
      Py_XDECREF(prec);
      PyErr_SetString(PyExc_IndexError, "Index out of bounds.");
      return NULL;
     }
    }
    
    to_iii = to_ii;
    if (arr_to!=NULL)
    {
     to_iii = *(int*)PyArray_GETPTR1(arr_to, to_ii);
     if ((to_iii<0)||(to_iii>=self->node_count))
     {
      Py_XDECREF(arr_from);
      Py_DECREF(arr_to);
      Py_XDECREF(poffset);
      Py_XDECREF(prec);
      PyErr_SetString(PyExc_IndexError, "Index out of bounds.");
      return NULL;
     }
    }
    
   // Fetch the offset and precision values as needed...
    if (poffset!=NULL)
    {
     pos = *(double*)PyArray_GETPTR1(poffset, i % PyArray_DIMS(poffset)[0]);
    }
    
    if (prec!=NULL)
    {
     pr = *(double*)PyArray_GETPTR1(prec, i % PyArray_DIMS(prec)[0]);
    }
    
   // Test the user is not stupid...
    if (from_iii==to_iii)
    {
     Py_XDECREF(arr_from);
     Py_XDECREF(arr_to);
     Py_XDECREF(poffset);
     Py_XDECREF(prec);
     PyErr_SetString(PyExc_IndexError, "Edges connecting a node to itself are not permitted");
     return NULL;
    }
    
   // Fetch the relevant edge, creating it if need be...
    HalfEdge * targ = GBP_always_get_edge(self, from_iii, to_iii);
    
   // Apply the update...
    if (prec_obj!=NULL)
    {
     // Offset with precision case...
      HalfEdge_offset_mult_raw(targ, pos, pr, prev_weight);
    }
    else
    {
     // Precision only case...
     float * co = &(HalfEdge_edge(targ)->co);
     *co = (*co) * prev_weight + pos;
    }
  }
  
 // Cleanup and return None...
  Py_XDECREF(arr_from);
  Py_XDECREF(arr_to);
  Py_XDECREF(poffset);
  Py_XDECREF(prec);
 
  Py_INCREF(Py_None);
  return Py_None;
}


static PyObject * GBP_pairwise_sd_py(GBP * self, PyObject * args, PyObject * kw)
{
 // Three or four parameters please, with keywords as optional...
  PyObject * index_from;
  PyObject * index_to;
  PyObject * offset_obj;
  PyObject * sd_obj = NULL;
  float prev_weight = 1.0;
  
  static char * kw_list[] = {"from", "to", "offset", "sd", "prev_exp", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOO|Of", kw_list, &index_from, &index_to, &offset_obj, &sd_obj, &prev_weight)) return NULL;
  
  if (sd_obj==Py_None) sd_obj = NULL;

 // Analyse the two node indices...
  Py_ssize_t start_from;
  Py_ssize_t step_from;
  Py_ssize_t length_from;
  PyArrayObject * arr_from;
  if (GBP_index(self, index_from, &start_from, &step_from, &length_from, &arr_from, NULL)!=0)
  {
   return NULL;
  }
  
  Py_ssize_t start_to;
  Py_ssize_t step_to;
  Py_ssize_t length_to;
  PyArrayObject * arr_to;
  if (GBP_index(self, index_to, &start_to, &step_to, &length_to, &arr_to, NULL)!=0)
  {
   Py_XDECREF(arr_from);
   return NULL;
  }
  
  if (length_from!=length_to)
  {
   Py_XDECREF(arr_from);
   Py_XDECREF(arr_to);
   
   PyErr_SetString(PyExc_IndexError, "from and to must have the same number of node indices");
   return NULL;
  }

 // Now offset and precision...
  PyArrayObject * offset = NULL;
  if ((PyInt_Check(offset_obj)==0)&&(PyFloat_Check(offset_obj)==0))
  {
   offset = (PyArrayObject*)PyArray_ContiguousFromAny(offset_obj, NPY_DOUBLE, 1, 1);
   if (offset==NULL)
   {
    Py_XDECREF(arr_from);
    Py_XDECREF(arr_to);
    return NULL;
   }
  }
   
  PyArrayObject * sd = NULL;
  if ((sd_obj!=NULL) && (PyInt_Check(sd_obj)==0) && (PyFloat_Check(sd_obj)==0))
  {
   sd = (PyArrayObject*)PyArray_ContiguousFromAny(sd_obj, NPY_DOUBLE, 1, 1);
   if (sd==NULL)
   {
    Py_XDECREF(arr_from);
    Py_XDECREF(arr_to);
    Py_XDECREF(offset);
    return NULL;
   }
  }
  
 // Loop through and set the required values...
  int i, from_ii, to_ii, from_iii, to_iii;
  float os = (offset==NULL) ? PyFloat_AsDouble(offset_obj) : 0.0;
  float d = ((sd==NULL)&&(sd_obj!=NULL))   ? PyFloat_AsDouble(sd_obj)   : 0.0;
  
  for (i=0,from_ii=start_from,to_ii=start_to; i<length_to; i++,from_ii+=step_from,to_ii+=step_to)
  {
   // Handle array indexing for both from and to...
    from_iii = from_ii;
    if (arr_from!=NULL)
    {
     from_iii = *(int*)PyArray_GETPTR1(arr_from, from_ii);
     if ((from_iii<0)||(from_iii>=self->node_count))
     {
      Py_DECREF(arr_from);
      Py_XDECREF(arr_to);
      Py_XDECREF(offset);
      Py_XDECREF(sd);
      PyErr_SetString(PyExc_IndexError, "Index out of bounds.");
      return NULL;
     }
    }
    
    to_iii = to_ii;
    if (arr_to!=NULL)
    {
     to_iii = *(int*)PyArray_GETPTR1(arr_to, to_ii);
     if ((to_iii<0)||(to_iii>=self->node_count))
     {
      Py_XDECREF(arr_from);
      Py_DECREF(arr_to);
      Py_XDECREF(offset);
      Py_XDECREF(sd);
      PyErr_SetString(PyExc_IndexError, "Index out of bounds.");
      return NULL;
     }
    }
    
   // Fetch the offset and precision values as needed...
    if (offset!=NULL)
    {
     os = *(double*)PyArray_GETPTR1(offset, i % PyArray_DIMS(offset)[0]);
    }
    
    if (sd!=NULL)
    {
     d = *(double*)PyArray_GETPTR1(sd, i % PyArray_DIMS(sd)[0]);
    }
    
   // Test the user is not stupid...
    if (from_iii==to_iii)
    {
     Py_XDECREF(arr_from);
     Py_XDECREF(arr_to);
     Py_XDECREF(offset);
     Py_XDECREF(sd);
     PyErr_SetString(PyExc_IndexError, "Edges connecting a node to itself are not permitted");
     return NULL;
    }
    
   // Fetch the relevant edge, creating it if need be...
    HalfEdge * targ = GBP_always_get_edge(self, from_iii, to_iii);
    
   // Apply the update...
    if (sd_obj!=NULL)
    {
     // Offset with precision case...
      HalfEdge_offset_mult(targ, os, 1.0/(d*d), prev_weight);
    }
    else
    {
     // Precision only case...
     float * co = &(HalfEdge_edge(targ)->co);
     *co = (*co) * prev_weight + 1.0/(os*os);
    }
  }
  
 // Cleanup and return None...
  Py_XDECREF(arr_from);
  Py_XDECREF(arr_to);
  Py_XDECREF(offset);
  Py_XDECREF(sd);
 
  Py_INCREF(Py_None);
  return Py_None;
}



static PyObject * GBP_solve_bp_py(GBP * self, PyObject * args)
{
 // Fetch the maximum iterations, desired epsilon and momentum...
  int max_iters = 1024;
  float epsilon = 1e-6;
  float momentum = 0.1;
  if (!PyArg_ParseTuple(args, "|iff", &max_iters, &epsilon, &momentum)) return NULL;
  float rev_momentum = 1.0 - momentum;
  
 // Loop through passing, alternating between forwards and backwards throught he node order...
  int dir = 1;
  int iters = 0;
  int i;
  
  while (1)
  {
   float delta = 0.0;
   
   // Loop and parse each node inturn...
    for (i=((dir>0)?(0):(self->node_count-1)); (i>=0)&&(i<self->node_count); i+=dir)
    {
     Node * targ = self->node + i;
     if (targ->on==0) continue; // Skip nodes that have been switched off.
     
     if (targ->unary_prec>infinity_and_beyond)
     {
      // Only process infinite nodes once - no information flows through them so this works...
       if (iters==0)
       {
        // Pass the messages, which are constant...
         HalfEdge * msg = targ->first;
         while (msg!=NULL)
         {
          float oset_pmean = HalfEdge_offset_pmean(msg);
          float oset_prec = HalfEdge_edge(msg)->diag;
         
          msg->prec = oset_prec;
          msg->pmean = oset_pmean + targ->unary_pmean * oset_prec;
          
          msg = msg->next;
         }
       }
     }
     else
     {
      // Sumarise the incomming messages for the node, as the total sum thus far...
       targ->pmean = targ->unary_pmean;
       targ->prec = targ->unary_prec;
      
       HalfEdge * msg = targ->first;
       while (msg!=NULL)
       {
        targ->pmean += msg->reverse->pmean;
        targ->prec  += msg->reverse->prec;
    
        msg = msg->next; 
       }
      
      // Go through and calculate the output of each message by subtracting from the summary this one message and then calculating the message to send...
       msg = targ->first;
       while (msg!=NULL)
       {
        float oset_pmean = HalfEdge_offset_pmean(msg);
        float oset_prec = HalfEdge_edge(msg)->diag;
        float gauss_prec = HalfEdge_edge(msg)->co;
       
        float msg_prec = targ->prec - msg->reverse->prec;
        float msg_pmean = targ->pmean - msg->reverse->pmean;
       
        float div = oset_prec + msg_prec;
        if (fabs(div)<1e-6) div = copysign(1e-6, div);
        float diag = gauss_prec - oset_prec;
       
        float new_prec  = oset_prec - diag * diag / div;
        float new_pmean = oset_pmean - (msg_pmean - oset_pmean) * diag / div;
       
        new_prec = momentum*msg->prec + rev_momentum*new_prec;
        new_pmean = momentum*msg->pmean + rev_momentum*new_pmean;
       
        float dp = fabs(new_prec - msg->prec);
        if (dp>delta) delta = dp;
       
        float dm = fabs(new_pmean - msg->pmean);
        if (dm>delta) delta = dm;
       
        msg->prec = new_prec;
        msg->pmean = new_pmean;
    
        msg = msg->next;
       }
     }
    }
    
   // Check epsilon, update iteration count, break if done and swap the direction...
    ++iters;
    if (delta<epsilon) break;
    if (iters>=max_iters) break;
    dir *= -1;
  }
  
 // Sumarrise the incomming messages one last time - we want to use the last iterations messages!..
  for (i=0; i<self->node_count; i++)
  {
   Node * targ = self->node + i;
   targ->pmean = targ->unary_pmean; 
   targ->prec = targ->unary_prec;
   
   if (targ->prec<=infinity_and_beyond)
   {
    HalfEdge * msg = targ->first;
    while (msg!=NULL)
    {
     targ->pmean += msg->reverse->pmean;
     targ->prec  += msg->reverse->prec;
    
     msg = msg->next; 
    }
   }
  }
  
 // Return the total number of iterations...
  return Py_BuildValue("i", iters);
}


static PyObject * GBP_solve_trws_py(GBP * self, PyObject * args)
{
 // Fetch the maximum iterations and desired epsilon...
  int max_iters = 1024;
  float epsilon = 1e-6;
  if (!PyArg_ParseTuple(args, "|iff", &max_iters, &epsilon)) return NULL;
  
 // Loop through passing, alternating between forwards and backwards throught he node order...
  int dir = 1;
  int iters = 0;
  int i;
  float delta = 0.0;
  
  while (1)
  {
   // Loop and parse each node inturn...
    for (i=((dir>0)?(0):(self->node_count-1)); (i>=0)&&(i<self->node_count); i+=dir)
    {
     Node * targ = self->node + i;
     if (targ->on==0) continue; // Skip nodes that have been switched off.

     if (targ->unary_prec>infinity_and_beyond)
     {
      // Only process infinite nodes once - no information flows through them so this works...
       if (iters==0)
       {
        // Pass the messages, which are constant...
         HalfEdge * msg = targ->first;
         while (msg!=NULL)
         {
          float oset_pmean = HalfEdge_offset_pmean(msg);
          float oset_prec = HalfEdge_edge(msg)->diag;
         
          msg->prec = oset_prec;
          msg->pmean = oset_pmean + targ->unary_pmean * oset_prec;
          
          msg = msg->next;
         }
       }
     }
     else
     {
      // Summarise the incomming messages for the node, as the total sum thus far... 
       targ->pmean = targ->unary_pmean;
       targ->prec = targ->unary_prec;
      
       HalfEdge * msg = targ->first;
       while (msg!=NULL)
       {
        targ->pmean += msg->reverse->pmean;
        targ->prec  += msg->reverse->prec;
    
        msg = msg->next; 
       }
      
      // Go through and calculate the output of each message by subtracting from the summary this one message and then calculating the message to send...
       msg = targ->first;
       while (msg!=NULL)
       {
        // Only do the edge if its going in the correct direction for this pass (dir is 1 for positive direction, -1 for negative direction, values of pointers to nodes define the ordering)...
         if (((msg->dest - targ) * dir)>0)
         {
          float oset_pmean = HalfEdge_offset_pmean(msg);
          float oset_prec = HalfEdge_edge(msg)->diag;
          float gauss_prec = HalfEdge_edge(msg)->co;
         
          int chain_count = Node_chain_count(targ);
          float msg_prec = (targ->prec / chain_count) - msg->reverse->prec;
          float msg_pmean = (targ->pmean / chain_count) - msg->reverse->pmean;
       
          float div = oset_prec + msg_prec;
          if (fabs(div)<1e-6) div = copysign(1e-6, div);
          float diag = gauss_prec - oset_prec;
       
          float new_prec  = oset_prec - diag * diag / div;
          float new_pmean = oset_pmean - (msg_pmean - oset_pmean) * diag / div;
       
          float dp = fabs(new_prec - msg->prec);
          if (dp>delta) delta = dp;
       
          float dm = fabs(new_pmean - msg->pmean);
          if (dm>delta) delta = dm;
       
          msg->prec = new_prec;
          msg->pmean = new_pmean;
         }
        
        msg = msg->next;
       }
     }
    }
    
   // Check epsilon, update iteration count, break if done and swap the direction...
    ++iters;
    if (iters>=max_iters) break;
    
    if ((iters%2)==0)
    {
     if (delta<epsilon) break;
     delta = 0.0;
    }
    
    dir *= -1;
  }
  
 // Sumarise the incomming messages one last time - we want to use the last iterations messages!..
  for (i=0; i<self->node_count; i++)
  {
   Node * targ = self->node + i;
   targ->pmean = targ->unary_pmean; 
   targ->prec = targ->unary_prec;
      
   HalfEdge * msg = targ->first;
   while (msg!=NULL)
   {
    targ->pmean += msg->reverse->pmean;
    targ->prec  += msg->reverse->prec;
    
    msg = msg->next; 
   }
  }
  
 // Return the total number of iterations...
  return Py_BuildValue("i", iters);
}



static PyObject * GBP_result_py(GBP * self, PyObject * args)
{
 // Convert the parameter to something we can dance with...
  PyObject * index = NULL;
  PyArrayObject * mean = NULL;
  PyArrayObject * prec = NULL;
  if (!PyArg_ParseTuple(args, "|OO!O!", &index, &PyArray_Type, &mean, &PyArray_Type, &prec)) return NULL;
 
  Py_ssize_t start;
  Py_ssize_t step;
  Py_ssize_t length;
  PyArrayObject * arr;
  int singular;
  
  if (GBP_index(self, index, &start, &step, &length, &arr, &singular)!=0) return NULL;
  
 // Special case a singular scenario...
  if ((singular!=0)&&(mean==NULL)&&(prec==NULL))
  {
   float p = self->node[start].prec;
   float m = self->node[start].pmean;
   
   if (p<=infinity_and_beyond)
   {
    float div = p;
    if (fabs(div)<1e-6) div = copysign(1e-6, div);
    
    m /= div;
   }
   
   Py_XDECREF(arr);
   return Py_BuildValue("(f,f)", m, p); 
  }
  
 // Create the return arrays, or validate the existing ones... 
  if (mean==NULL)
  {
   npy_intp len = length;
   mean = (PyArrayObject*)PyArray_SimpleNew(1, &len, NPY_FLOAT32);
  }
  else
  {
   if ((PyArray_NDIM(mean)!=1)||(PyArray_DIMS(mean)[0]!=length)||(PyArray_DESCR(mean)->kind!='f')||(PyArray_DESCR(mean)->elsize!=sizeof(float)))
   {
    PyErr_SetString(PyExc_TypeError, "Provided mean array did not satisfy the requirements");
    Py_XDECREF(arr);
    return NULL; 
   }
    
   Py_INCREF(mean); 
  }
  
  if (prec==NULL)
  {
   npy_intp len = length;
   prec = (PyArrayObject*)PyArray_SimpleNew(1, &len, NPY_FLOAT32);
  }
  else
  {
   if ((PyArray_NDIM(prec)!=1)||(PyArray_DIMS(prec)[0]!=length)||(PyArray_DESCR(prec)->kind!='f')||(PyArray_DESCR(prec)->elsize!=sizeof(float)))
   {
    PyErr_SetString(PyExc_TypeError, "Provided prec array did not satisfy the requirements");
    Py_XDECREF(arr);
    Py_DECREF(mean);
    return NULL; 
   }
    
   Py_INCREF(prec); 
  }
  
 // Loop and store each value in turn...
  int i, ii, iii;
  for (i=0,ii=start; i<length; i++,ii+=step)
  {
   // Handle if an array has been passed...
    iii = ii;
    if (arr!=NULL)
    {
     iii = *(int*)PyArray_GETPTR1(arr, ii);
     if ((iii<0)||(iii>=self->node_count))
     {
      Py_DECREF(arr);
      Py_DECREF(mean);
      Py_DECREF(prec);
      PyErr_SetString(PyExc_IndexError, "Index out of bounds.");
      return NULL;
     }
    }
    
   // Store the relevant values for this entry...
    float p = self->node[iii].prec;
    float m = self->node[iii].pmean;
    
    if (p<=infinity_and_beyond)
    {
     float div = p;
     if (fabs(div)<1e-6) div = copysign(1e-6, div);
    
     m /= div;
    }
    
    *(float*)PyArray_GETPTR1(prec, i) = p;
    *(float*)PyArray_GETPTR1(mean, i) = m;
  }
 
 // Clean up and return... 
  Py_XDECREF(arr);
  return Py_BuildValue("(N,N)", mean, prec); 
}



static PyObject * GBP_result_raw_py(GBP * self, PyObject * args)
{
 // Convert the parameter to something we can dance with...
  PyObject * index = NULL;
  PyArrayObject * pmean = NULL;
  PyArrayObject * prec = NULL;
  if (!PyArg_ParseTuple(args, "|OO!O!", &index, &PyArray_Type, &pmean, &PyArray_Type, &prec)) return NULL;
 
  Py_ssize_t start;
  Py_ssize_t step;
  Py_ssize_t length;
  PyArrayObject * arr;
  int singular;
  
  if (GBP_index(self, index, &start, &step, &length, &arr, &singular)!=0) return NULL;
  
 // Special case a singular scenario...
  if ((singular!=0)&&(pmean==NULL)&&(prec==NULL))
  {
   float p = self->node[start].prec;
   float m = self->node[start].pmean;
   
   Py_XDECREF(arr);
   return Py_BuildValue("(f,f)", m, p); 
  }
  
 // Create the return arrays, or validate the existing ones... 
  if (pmean==NULL)
  {
   npy_intp len = length;
   pmean = (PyArrayObject*)PyArray_SimpleNew(1, &len, NPY_FLOAT32);
  }
  else
  {
   if ((PyArray_NDIM(pmean)!=1)||(PyArray_DIMS(pmean)[0]!=length)||(PyArray_DESCR(pmean)->kind!='f')||(PyArray_DESCR(pmean)->elsize!=sizeof(float)))
   {
    PyErr_SetString(PyExc_TypeError, "Provided p-mean array did not satisfy the requirements");
    Py_XDECREF(arr);
    return NULL; 
   }
    
   Py_INCREF(pmean); 
  }
  
  if (prec==NULL)
  {
   npy_intp len = length;
   prec = (PyArrayObject*)PyArray_SimpleNew(1, &len, NPY_FLOAT32);
  }
  else
  {
   if ((PyArray_NDIM(prec)!=1)||(PyArray_DIMS(prec)[0]!=length)||(PyArray_DESCR(prec)->kind!='f')||(PyArray_DESCR(prec)->elsize!=sizeof(float)))
   {
    PyErr_SetString(PyExc_TypeError, "Provided prec array did not satisfy the requirements");
    Py_XDECREF(arr);
    Py_DECREF(pmean);
    return NULL; 
   }
    
   Py_INCREF(prec); 
  }
  
 // Loop and store each value in turn...
  int i, ii, iii;
  for (i=0,ii=start; i<length; i++,ii+=step)
  {
   // Handle if an array has been passed...
    iii = ii;
    if (arr!=NULL)
    {
     iii = *(int*)PyArray_GETPTR1(arr, ii);
     if ((iii<0)||(iii>=self->node_count))
     {
      Py_DECREF(arr);
      Py_DECREF(pmean);
      Py_DECREF(prec);
      PyErr_SetString(PyExc_IndexError, "Index out of bounds.");
      return NULL;
     }
    }
    
   // Store the relevant values for this entry...
    float p = self->node[iii].prec;
    *(float*)PyArray_GETPTR1(prec, i) = p;

    float m = self->node[iii].pmean;
    *(float*)PyArray_GETPTR1(pmean, i) = m;
  }
 
 // Clean up and return... 
  Py_XDECREF(arr);
  return Py_BuildValue("(N,N)", pmean, prec); 
}


static PyObject * GBP_result_sd_py(GBP * self, PyObject * args)
{
 // Convert the parameter to something we can dance with...
  PyObject * index = NULL;
  PyArrayObject * mean = NULL;
  PyArrayObject * sd = NULL;
  if (!PyArg_ParseTuple(args, "|OO!O!", &index, &PyArray_Type, &mean, &PyArray_Type, &sd)) return NULL;
 
  Py_ssize_t start;
  Py_ssize_t step;
  Py_ssize_t length;
  PyArrayObject * arr;
  int singular;
  
  if (GBP_index(self, index, &start, &step, &length, &arr, &singular)!=0) return NULL;
  
 // Special case a singular scenario...
  if ((singular!=0)&&(mean==NULL)&&(sd==NULL))
  {
   float p = self->node[start].prec;
   float m = self->node[start].pmean;
   
   if (p<=infinity_and_beyond)
   {
    float div = p;
    if (fabs(div)<1e-6) div = copysign(1e-6, div);
    m /= div;
   }
   float d = 1.0 / sqrt((p>=0.0)?p:0.0);
   
   Py_XDECREF(arr);
   return Py_BuildValue("(f,f)", m, d); 
  }
  
 // Create the return arrays, or validate the existing ones... 
  if (mean==NULL)
  {
   npy_intp len = length;
   mean = (PyArrayObject*)PyArray_SimpleNew(1, &len, NPY_FLOAT32);
  }
  else
  {
   if ((PyArray_NDIM(mean)!=1)||(PyArray_DIMS(mean)[0]!=length)||(PyArray_DESCR(mean)->kind!='f')||(PyArray_DESCR(mean)->elsize!=sizeof(float)))
   {
    PyErr_SetString(PyExc_TypeError, "Provided mean array did not satisfy the requirements");
    Py_XDECREF(arr);
    return NULL; 
   }
    
   Py_INCREF(mean); 
  }
  
  if (sd==NULL)
  {
   npy_intp len = length;
   sd = (PyArrayObject*)PyArray_SimpleNew(1, &len, NPY_FLOAT32);
  }
  else
  {
   if ((PyArray_NDIM(sd)!=1)||(PyArray_DIMS(sd)[0]!=length)||(PyArray_DESCR(sd)->kind!='f')||(PyArray_DESCR(sd)->elsize!=sizeof(float)))
   {
    PyErr_SetString(PyExc_TypeError, "Provided sd array did not satisfy the requirements");
    Py_XDECREF(arr);
    Py_DECREF(mean);
    return NULL; 
   }
    
   Py_INCREF(sd); 
  }
  
 // Loop and store each value in turn...
  int i, ii, iii;
  for (i=0,ii=start; i<length; i++,ii+=step)
  {
   // Handle if an array has been passed...
    iii = ii;
    if (arr!=NULL)
    {
     iii = *(int*)PyArray_GETPTR1(arr, ii);
     if ((iii<0)||(iii>=self->node_count))
     {
      Py_DECREF(arr);
      Py_DECREF(mean);
      Py_DECREF(sd);
      PyErr_SetString(PyExc_IndexError, "Index out of bounds.");
      return NULL;
     }
    }
    
   // Store the relevant values for this entry...
    float p = self->node[iii].prec;
    float m = self->node[iii].pmean;
    
    if (p<=infinity_and_beyond)
    {
     float div = p;
     if (fabs(div)<1e-6) div = copysign(1e-6, div);
     m /= div;
    }
    float d = 1.0 / sqrt((p>=0.0)?p:0.0);
    
    *(float*)PyArray_GETPTR1(mean, i) = m;
    *(float*)PyArray_GETPTR1(sd, i)   = d;
  }
 
 // Clean up and return... 
  Py_XDECREF(arr);
  return Py_BuildValue("(N,N)", mean, sd); 
}



static PyMemberDef GBP_members[] =
{
 {"node_count", T_INT, offsetof(GBP, node_count), READONLY, "Number of nodes in the graph"},
 {"edge_count", T_INT, offsetof(GBP, edge_count), READONLY, "Number of edges in the graph"},
 {"block_size", T_INT, offsetof(GBP, block_size), 0, "Number of edges worth of memory to allocate each time it runs out of space for more. Can be editted whenever you want, but will only affect future allocations."},
 {NULL}
};



static PyMethodDef GBP_methods[] =
{
 {"clone", (PyCFunction)GBP_clone_py, METH_NOARGS, "Returns a clone of this object - everything is exactly the same, though all memory allocations will have been adjusted to be tight, rather than having spare space for future edges."},
 
 {"reset_unary", (PyCFunction)GBP_reset_unary_py, METH_VARARGS, "Given array-like indexing (integer, slice, numpy array or something that can be interpreted as an array) this resets the unary term for all of the given node indices, back to 'no information'."},
 {"reset_pairwise", (PyCFunction)GBP_reset_pairwise_py, METH_VARARGS, "Given two inputs, as indices of nodes between an edge this resets that edge to provide 'no relationship' between the nodes. Can do one edge with two integers or a set of edges with two numpy arrays. Can give one input as an integer and another as a numpy array to do a list of edges that all include the same node. Can omit the second parameter to wipe out all edges for a given node or set of nodes, or both to wipe them all up."},
 
 {"add", (PyCFunction)GBP_add_py, METH_VARARGS, "Adds new random variables to the graph - you optionally provide the number to add, which defaults to 1. It returns the index of the first random variable, with them being contiguous so you can infer the rest. Note that this is slow, as it involves a realloc and a lot of pointer twiddlng - best avoided, and best done with large blocks at once - repeated calls with the default parameter would be stupid."},
 
 {"enable", (PyCFunction)GBP_enable_py, METH_VARARGS, "Given array-like indexing (integer, slice, numpy array or something that can be interpreted as an array) this enables the given nodes, reverting their status from being disabled if that has been done. (Nodes are enabled by default)"},
 {"disable", (PyCFunction)GBP_disable_py, METH_VARARGS, "Given array-like indexing (integer, slice, numpy array or something that can be interpreted as an array) this disables the given nodes. A disabled node effectivly does not exist for the purpose of inference, but can be easily reenabled as required. Note that disabling a node only removes its influence - it remains in the system and does generate some inefficiency (messages are still sent to it, just not sent from it) compared to not having the node at all."},
 
 {"unary", (PyCFunction)GBP_unary_py, METH_KEYWORDS | METH_VARARGS, "Given three parameters - the node indices, then the means and then the precision values (inverse variance/inverse squared standard deviation). It then updates the nodes by multiplying the unary term already in the node with the given, noting that this is a set for a node that has not yet had unary called on it/been reset. For the node indicies it accepts an integer, a slice, a numpy array or something that can be converted into a numpy array. For the mean and precision it accepts either floats or a numpy array, noting that if an array is too small it will be accessed modulus the number of nodes provided. Note that it accepts negative precision, which allows you to divide through rather than multiply - can be used to remove previously provided information for instance. Also accepts a fourth parameter, an exponent of the previous pdf before multiplication - it defaults to 1, which is multiplying the preexisting distribution with the new distribution provided by this call; a typical alternate value is to set it to 0 to replace the old information entirly. Also support keyword arguments: {index, mean, prec, prev_exp}. Supports infinity as the precision value, for when you want a delta distribution."},
 {"unary_raw", (PyCFunction)GBP_unary_raw_py, METH_KEYWORDS | METH_VARARGS, "Identical to unary, except you pass in the precision multiplied by the mean instead of just the mean (other arguments remain the same) - this is the actual internal representation, and so saves a multiplication (maybe a division) if you already have that. The keyword arguments are: {index, pmean, prec, prev_exp}. Support for infinite precision is a bit weird - you can pass it in, but if precision is infinite then you provide the mean, not the p-mean."},
 {"unary_sd", (PyCFunction)GBP_unary_sd_py, METH_KEYWORDS | METH_VARARGS, "Identical to unary, except you pass in the normal mean and standard deviation (other arguments remain the same). A conveniance interface. The keyword arguments are: {index, mean, sd, prev_exp}. A sd of zero means infinite precision, which will be handled correctly."},
 
 {"pairwise", (PyCFunction)GBP_pairwise_py, METH_KEYWORDS | METH_VARARGS, "Given three or four parameters - the first is the from node index, the second the too node index. If there are three parameters then the third is the precision between the two unary terms (in this use case you are defining a sparse Gaussian distributon to marginalise), if four parameters then it is the expected offset in the implied direction followed by the precision of that offset (in this use case your solving a linear equation of differences between random variables). Has the same amount of flexability as the unary method, except it insists that the two node index objects have the same length. This supports the negative precision insanity (well, perfectly reasonable in the three parameter case), the multiplication of pre-existing stuff etc. Also accepts a fifth parameter (can set the fourth to None if required), an exponent of the previous pdf before multiplication - it defaults to 1, which is multiplying the preexisting distribution with the new distribution provided by this call; a typical alternate value is to set it to 0 to replace the old information entirly. Also support keyword arguments: {from, to, offset, prec, prev_exp}."},
 {"pairwise_raw", (PyCFunction)GBP_pairwise_raw_py, METH_KEYWORDS | METH_VARARGS, "Identical to pairwise except you provide the offset multiplied by the mean instead of just the offset - this is the internal representation and so saves a little time. In the three parameter case of providing a precision between variables it makes no difference if you call this or pairwise. The keyword arguments are: {from, to, poffset, prec, prev_exp}."},
 {"pairwise_sd", (PyCFunction)GBP_pairwise_sd_py, METH_KEYWORDS | METH_VARARGS, "Identical to pairwise except it takes the standard deviation instead of the precision - a conveniance method. In the three parameter case you are again providing the standard deviation, though this is a bit weird and I can't think of an actual use case. The keyword arguments are: {from, to, offset, sd, prev_exp}."},
 
 {"solve_bp", (PyCFunction)GBP_solve_bp_py, METH_VARARGS, "Solves the model using BP. Optionally given three parameters - the iteration cap, the epsilon and the momentum, which default to 1024, 1e-6 and 0.1 respectivly. Returns how many iterations have been performed."},
 {"solve_trws", (PyCFunction)GBP_solve_trws_py, METH_VARARGS, "Solves the model, using TRW-S. Optionally given two parameters - the iteration cap and the epsilon, which default to 1024 and 1e-6 respectivly. Returns how many iterations have been performed."},
 {"solve", (PyCFunction)GBP_solve_bp_py, METH_VARARGS, "Synonym for a default solver, specifically the solve_bp method."},
 
 {"result", (PyCFunction)GBP_result_py, METH_VARARGS, "Given a standard array index (integer, slice, numpy array, equiv. to numpy array) this returns the marginal of the indexed nodes, as a tuple (mean, precision), noting that as precision approaches zero the mean will arbitrarily veer towards zero, to avoid instability (Equivalent to being regularised with a really wide distribution when below an epsilon). The output can be either a tuple of floats or arrays, depending on the request. There are two optional parameters where you can provide the return arrays, to avoid it doing memory allocation - they must be the correct size and floaty, and must be arrays even if you are requesting a single variable."},
 {"result_raw", (PyCFunction)GBP_result_raw_py, METH_VARARGS, "Identical to result(...), except it outputs the p-mean instead of the mean. The p-mean is the precision multiplied by the mean, and is the internal representation used - this allows you to avoid the regularisation that result(...) applies to low precision values (to avoid divide by zeros) and get at the raw result. Note that if the precision is sufficiently high (greater than 1e32) to be considered as infinity then the pmean is replaced by the mean - its an internal hack to support infinite precision values."},
 {"result_sd", (PyCFunction)GBP_result_sd_py, METH_VARARGS, "Identical to result(...), except it outputs standard deviation instead of precision - a conveniance method. Note that it can output a standard deviation of infinity, indicating a precision of zero."},
 
 {NULL}
};



static PyTypeObject GBPType =
{
 PyObject_HEAD_INIT(NULL)
 0,                                /*ob_size*/
 "gbp_c.GBP",                      /*tp_name*/
 sizeof(GBP),                      /*tp_basicsize*/
 0,                                /*tp_itemsize*/
 (destructor)GBP_dealloc_py,       /*tp_dealloc*/
 0,                                /*tp_print*/
 0,                                /*tp_getattr*/
 0,                                /*tp_setattr*/
 0,                                /*tp_compare*/
 0,                                /*tp_repr*/
 0,                                /*tp_as_number*/
 0,                                /*tp_as_sequence*/
 0,                                /*tp_as_mapping*/
 0,                                /*tp_hash */
 0,                                /*tp_call*/
 0,                                /*tp_str*/
 0,                                /*tp_getattro*/
 0,                                /*tp_setattro*/
 0,                                /*tp_as_buffer*/
 Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
 "A fairly straight forward implimentation of Gaussian belief propagation, with univariate random variables for each node, with unary and pairwise terms, where pairwise terms are specified in terms of offsets with noise or precisions. Works with inverse precision throughout, so it can represent 'no information' with a value of zero and avoid the awkward variance of zero, which it does not support. Note that for GBP the MAP and the marginal are the same, so this is calculating both. It also has a TRW-S solver. Constructed with the number of random variables, which remains constant; number of random edges can change at will, and the constructor takes an optional second parameter for the block size, as in the number of new edges to allocate each time it runs out of memory.", /* tp_doc */
 0,                                /* tp_traverse */
 0,                                /* tp_clear */
 0,                                /* tp_richcompare */
 0,                                /* tp_weaklistoffset */
 0,                                /* tp_iter */
 0,                                /* tp_iternext */
 GBP_methods,                      /* tp_methods */
 GBP_members,                      /* tp_members */
 0,                                /* tp_getset */
 0,                                /* tp_base */
 0,                                /* tp_dict */
 0,                                /* tp_descr_get */
 0,                                /* tp_descr_set */
 0,                                /* tp_dictoffset */
 0,                                /* tp_init */
 0,                                /* tp_alloc */
 GBP_new_py,                       /* tp_new */
};



static PyMethodDef gbp_c_methods[] =
{
 {NULL}
};



#ifndef PyMODINIT_FUNC
#define PyMODINIT_FUNC void
#endif

PyMODINIT_FUNC initgbp_c(void)
{
 PyObject * mod = Py_InitModule3("gbp_c", gbp_c_methods, "Provides a simple Gaussian belief propagation implimentation - basically a nice way of specifying certain kinds of linear problems.");
 
 import_array();
 
 if (PyType_Ready(&GBPType) < 0) return;
 
 Py_INCREF(&GBPType);
 PyModule_AddObject(mod, "GBP", (PyObject*)&GBPType);
}
