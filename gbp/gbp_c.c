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
  this->node[i].unary_p_mean = 0.0;
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



// Helper function - given a numpy object to mean a range of nodes this outputs information to allow that loop to be done. Allways outputs a standard set of slice details (start, step, length), and also optionally outputs an numpy array to be checked at each position in the slice loop if its not NULL. Note that if the array is output it must be reference decrimented after use. Returns 0 on success, -1 on failure (in which case an error will have been set and arr will definitely be NULL). Note that it doesn't range check if an array is involved...
int GBP_index(GBP * this, PyObject * arg, Py_ssize_t * start, Py_ssize_t * step, Py_ssize_t * length, PyArrayObject ** arr)
{
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
  
  if (GBP_index(self, index, &start, &step, &length, &arr)!=0) return NULL; 
  
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
    self->node[iii].unary_p_mean = 0.0;
    self->node[iii].unary_prec   = 0.0;
  }
  
 // Return None...
  Py_INCREF(Py_None);
  return Py_None;
}


static PyObject * GBP_reset_pairwise_py(GBP * self, PyObject * args)
{
 // Fetch the parameters...
  PyObject * index_a = NULL;
  PyObject * index_b = NULL;
  if (!PyArg_ParseTuple(args, "|OO", &index_a)) return NULL;

 // Special case two NULLs - i.e. delete everything...
  if ((index_a==NULL)&&(index_b==NULL))
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
  

 // Special case one null, i.e. terminate all edges leaving given nodes...
 
 
 // Interpret the index_b indices...
 
 
 // Do the loop - above means we only have to code one double loop...

 
 // Return None...
  Py_INCREF(Py_None);
  return Py_None;
}




static PyMemberDef GBP_members[] =
{
 {"node_count", T_INT, offsetof(GBP, node_count), 0, "Number of nodes in the graph"},
 {"edge_count", T_INT, offsetof(GBP, edge_count), 0, "Number of edges in the graph"},
 {NULL}
};



static PyMethodDef GBP_methods[] =
{
 {"reset_unary", (PyCFunction)GBP_reset_unary_py, METH_VARARGS, "Given array indexing (integer, slice, numpy array or something that can be interpreted as an array) this resets the unary term for all of the given node indices, back to 'no information'."},
 {"reset_pairwise", (PyCFunction)GBP_reset_pairwise_py, METH_VARARGS, "Given two inputs, as indices of nodes between an edge this resets that edge to provide 'no relationship' between the nodes. Can do one edge with two integers or a set of edges with two numpy arrays. Can give one input as an integer and another as a numpy array to do a list of edges that all include the same node. Can omit the second parameter to wipe out all edges for a given node or set of nodes, or both to wipe them all up."},
 
 //{"unary", (PyCFunction)GBP_unary_py, METH_VARARGS, "Given three parameters - the node indices, then the means and then the precision values (inverse variance/inverse squared standard deviation). It then updates the nodes by multiplying the unary term already in the node with the given, noting that this is a set for a node that has not yet had unary called on it/been reset. For the node indicies it accepts an integer, a slice, a numpy array or something that can be converted into a numpy array. For the mean and precision it accepts either floats or a numpy array, noting that if an array is too small it will be accessed modulus the number of nodes provided."},
 //{"pairwise", (PyCFunction)GBP_pairwise_py, METH_VARARGS, "Given four parameters - the from node index, the too node index, the expected offset in the implied direction and the precision. Has the same amount of flexability as the unary method, except it insists that the two node index objects have the same length."},
 
 //{"solve", (PyCFunction)GBP_solve_py, METH_VARARGS, "Solves the model - optionally given two parameters - the iteration cap and the epsilon, which default to 1024 and 1e-4 respectivly. Returns how many iterations have been performed."},
 
 //{"result", (PyCFunction)GBP_result_py, METH_VARARGS, "Given a standard array index (integer, slice, numpy array) this returns the marginal of the indexed nodes, as a tuple (mean, precision), noting that as precision approaches zero the mean will arbitrarily veer towards zero, to avoid instability (Equivalent to being regularised with a really wide distribution when below an epsilon). The output can be either a tuple of floats or arrays, depending on the request. There are two optional parameters where you can provide the return arrays, to avoid it doing memeory allocation - they must be the correct size and floaty"},
 
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
