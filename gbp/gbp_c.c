// Copyright 2014 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

#include <Python.h>
#include <structmember.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>



#include "gbp_c.h"



// Returns the offset of a half edge, going from source to destination...
static inline float HalfEdge_offset_pmean(HalfEdge * he)
{
 if (he < he->reverse) return  he->pairwise;
                  else return -he->reverse->pairwise;
}

// Returns the precision of the half edge...
static inline float HalfEdge_offset_prec(HalfEdge * he)
{
 if (he > he->reverse) return he->pairwise;
                  else return he->reverse->pairwise;
}

// Allows you to multiply the existing offset with another one - this is equivalent to set in the first instance when its initialised to a zero precision...
static void HalfEdge_offset_mult(HalfEdge * he, float offset, float prec)
{
 if (he < he->reverse)
 {
  he->pairwise += offset * prec;
  he->reverse->pairwise += prec;
 }
 else
 {
  he->pairwise += prec;
  he->reverse->pairwise -= offset * prec; 
 }
}


// Returns a new edge, as an edge with the reverse pointer set (leaves dest and next to the user to initialise)...
static HalfEdge * GBP_new_edge(GBP * this)
{
 // Get one half edge (we know we always deal with HalfEdge-s in pairs, hence why the below is safe)...
  if (this->gc==NULL)
  {
   Block * nb = (Block*)malloc(sizeof(Block) + sizeof(HalfEdge) * 2 * this->block_size);
   nb->next = this->storage;
   this->storage = nb;
   
   int i;
   for (i=this->block_size*2-1; i>=0; i--)
   {
    nb->data[i].next = this->gc;
    this->gc = nb->data + i;
   }
  }

  HalfEdge * a = this->gc;
  this->gc = a->next;
  
  HalfEdge * b = this->gc;
  this->gc = b->next;
  
 // Do a basic initialisation on them...
  a->reverse = b;
  a->pairwise = 0.0;
  a->pmean = 0.0;
  a->prec = 0.0;
  
  b->reverse = a;
  b->pairwise = 0.0;
  b->pmean = 0.0;
  b->prec = 0.0;
  
 // Return a...
  this->edge_count += 1;
  return a;
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
  
 // Return the newly created edge...
  return targ;
}



void GBP_new(GBP * this, int node_count)
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
 }
 
 this->edge_count = 0;
 
 this->gc = NULL;
 
 this->block_size = 512;
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
  if (!PyArg_ParseTuple(args, "i", &node_count)) return NULL;
  
 // Allocate the object...
  GBP * self = (GBP*)type->tp_alloc(type, 0);

 // On success construct it...
  if (self!=NULL) GBP_new(self, node_count);

 // Return the new object...
  return (PyObject*)self;
}

static void GBP_dealloc_py(GBP * self)
{
 GBP_dealloc(self);
 self->ob_type->tp_free((PyObject*)self);
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
    self->node[iii].unary_prec   = 0.0;
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
    if (self->node[i].first!=NULL)
    {
     HalfEdge * targ = self->node[i].first;
     while (targ->next!=NULL)
     {
      targ = targ->next; 
     }
     
     targ->next = self->gc;
     self->gc = self->node[i].first;
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
       HalfEdge * to_die = targ->first;
       targ->first = to_die->next;
       
      // Dump both into the rubish, update the edge count...
       to_die->reverse->next = self->gc;
       to_die->next = to_die->reverse;
       self->gc = to_die;
       
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
   
   int j, jj, jjj;
   for (j=0, jj=start_b; j<length_b; j++,jj+=step_b)
   {
    jjj = jj;
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
    
    if (jjj==iii) continue; // Don't error out as that would be inconveniant.
   
    Node * targ_b = self->node + jjj;
    
    // Ok - we have two nodes - targ_a and targ_b - see if we can find a pair of half edges that make up an edge between them, and terminate if found...
     // Find and remove from targ_a - if it does not exist then we have nothing to do and can stop...
      HalfEdge * atob = NULL;
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
       HalfEdge * t = targ_a->first;
       while (t->next->dest!=targ_a)
       {
        t = t->next;  
       }
       
       btoa = t->next;
       t->next = btoa->next;
      }
      
     // Deposite them both down the garbage chute; decriment the edge count...
      atob->next = self->gc;
      btoa->next = atob;
      self->gc = btoa;
       
      self->edge_count -= 1;
   }
  }
 
 // Clean up and return None...
  Py_XDECREF(arr_a);
  Py_XDECREF(arr_b);
 
  Py_INCREF(Py_None);
  return Py_None;
}



static PyObject * GBP_unary_py(GBP * self, PyObject * args)
{
 // We expect three parameters...
  PyObject * index;
  PyObject * mean_obj;
  PyObject * prec_obj;
  if (!PyArg_ParseTuple(args, "OOO", &index, &mean_obj, &prec_obj)) return NULL;
  
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
    self->node[iii].unary_pmean += m * p;
    self->node[iii].unary_prec  += p;
  }

 // Clean up and return None...
  Py_XDECREF(mean);
  Py_XDECREF(prec);
 
  Py_INCREF(Py_None);
  return Py_None;
}


static PyObject * GBP_pairwise_py(GBP * self, PyObject * args)
{
 // Four parameters please...
  PyObject * index_from;
  PyObject * index_to;
  PyObject * offset_obj;
  PyObject * prec_obj;
  if (!PyArg_ParseTuple(args, "OOOO", &index_from, &index_to, &offset_obj, &prec_obj)) return NULL;

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
  if ((PyInt_Check(prec_obj)==0)&&(PyFloat_Check(prec_obj)==0))
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
  float pr = (prec==NULL)   ? PyFloat_AsDouble(prec_obj)   : 0.0;
  
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
    HalfEdge_offset_mult(targ, os, pr);
  }
  
 // Cleanup and return None...
  Py_XDECREF(arr_from);
  Py_XDECREF(arr_to);
  Py_XDECREF(offset);
  Py_XDECREF(prec);
 
  Py_INCREF(Py_None);
  return Py_None;
}



static PyObject * GBP_solve_py(GBP * self, PyObject * args)
{
 // Fetch the maximum iterations and desired epsilon...
  int max_iters = 1024;
  float epsilon = 1e-4;
  if (!PyArg_ParseTuple(args, "|if", &max_iters, &epsilon)) return NULL;
  
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
     // Sumarise the incomming messages for the node, as the total sum thus far...
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
      
     // Go through and calculate the output of each message by subtracting from the summary this one message and then calculating the message to send...
      msg = targ->first;
      while (msg!=NULL)
      {
       float rel_oset = HalfEdge_offset_pmean(msg);
       float rel_prec = HalfEdge_offset_prec(msg);
       
       float div = rel_prec + targ->prec - msg->reverse->prec;
       if (div<1e-6) div = 1e-6;
       
       float new_prec  = rel_prec - rel_prec * rel_prec / div;
       float new_pmean = rel_oset + rel_prec * (targ->pmean - msg->reverse->pmean - rel_oset) / div;
       
       float dm = fabs(new_pmean - msg->pmean);
       if (dm>delta) delta = dm;
       
       float dp = fabs(new_prec - msg->prec);
       if (dp>delta) delta = dp;
       
       msg->pmean = new_pmean;
       msg->prec = new_prec;
    
       msg = msg->next; 
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
   float m = self->node[start].pmean / ((p>1e-6)?(p):(1e-6));
   
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
    *(float*)PyArray_GETPTR1(prec, i) = p;

    float m = self->node[iii].pmean / ((p>1e-6)?(p):(1e-6));
    *(float*)PyArray_GETPTR1(mean, i) = m;
  }
 
 // Clean up and return... 
  Py_XDECREF(arr);
  return Py_BuildValue("(N,N)", mean, prec); 
}



static PyMemberDef GBP_members[] =
{
 {"node_count", T_INT, offsetof(GBP, node_count), READONLY, "Number of nodes in the graph"},
 {"edge_count", T_INT, offsetof(GBP, edge_count), READONLY, "Number of edges in the graph"},
 {"block_size", T_INT, offsetof(GBP, block_size), 0, "Number of edges worth of memory to allocate each time it runs out of space for more."},
 {NULL}
};



static PyMethodDef GBP_methods[] =
{
 {"reset_unary", (PyCFunction)GBP_reset_unary_py, METH_VARARGS, "Given array indexing (integer, slice, numpy array or something that can be interpreted as an array) this resets the unary term for all of the given node indices, back to 'no information'."},
 {"reset_pairwise", (PyCFunction)GBP_reset_pairwise_py, METH_VARARGS, "Given two inputs, as indices of nodes between an edge this resets that edge to provide 'no relationship' between the nodes. Can do one edge with two integers or a set of edges with two numpy arrays. Can give one input as an integer and another as a numpy array to do a list of edges that all include the same node. Can omit the second parameter to wipe out all edges for a given node or set of nodes, or both to wipe them all up."},
 
 {"unary", (PyCFunction)GBP_unary_py, METH_VARARGS, "Given three parameters - the node indices, then the means and then the precision values (inverse variance/inverse squared standard deviation). It then updates the nodes by multiplying the unary term already in the node with the given, noting that this is a set for a node that has not yet had unary called on it/been reset. For the node indicies it accepts an integer, a slice, a numpy array or something that can be converted into a numpy array. For the mean and precision it accepts either floats or a numpy array, noting that if an array is too small it will be accessed modulus the number of nodes provided. Note that it accepts negative precision, which allows you to divide through rather than multiply - can be used to remove previously provided information for instance."},
 {"pairwise", (PyCFunction)GBP_pairwise_py, METH_VARARGS, "Given four parameters - the from node index, the too node index, the expected offset in the implied direction and the precision. Has the same amount of flexability as the unary method, except it insists that the two node index objects have the same length. This includes the negative precision insanity, the multiplication of pre-existing stuff etc."},
 
 {"solve", (PyCFunction)GBP_solve_py, METH_VARARGS, "Solves the model. Optionally given two parameters - the iteration cap and the epsilon, which default to 1024 and 1e-4 respectivly. Returns how many iterations have been performed."},
 
 {"result", (PyCFunction)GBP_result_py, METH_VARARGS, "Given a standard array index (integer, slice, numpy array, equiv. to numpy array) this returns the marginal of the indexed nodes, as a tuple (mean, precision), noting that as precision approaches zero the mean will arbitrarily veer towards zero, to avoid instability (Equivalent to being regularised with a really wide distribution when below an epsilon). The output can be either a tuple of floats or arrays, depending on the request. There are two optional parameters where you can provide the return arrays, to avoid it doing memory allocation - they must be the correct size and floaty, and must be arrays even if you are requesting a single variable."},
 
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
 "A fairly straight forward implimentation of Gaussian belief propagation, with univariate random variables for each node, with unary and pairwise terms, where pairwise terms are specified in terms of offsets. Works with inverse precision throughout, so it can represent 'no information' with a value of zero and avoid the awkward variance of zero, which it does not support. Note that for GBP the MAP and the marginal are the same, so this is calculating both.", /* tp_doc */
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
