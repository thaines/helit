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
static float HalfEdge_offset_mult(HalfEdge * he, float offset, float prec)
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
 
 this->block_size = 1024;
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



static PyMemberDef GBP_members[] =
{
 {"node_count", T_INT, offsetof(GBP, node_count), 0, "Number of nodes in the graph"},
 {"edge_count", T_INT, offsetof(GBP, edge_count), 0, "Number of edges in the graph"},
 {NULL}
};



static PyMethodDef GBP_methods[] =
{
 //{"reset_unary", (PyCFunction)GBP_reset_unary_py, METH_VARARGS, "Given array indexing (integer, slice or numpy array) this resets the unary term for all of the given node indices, back to 'no information'."},
 //{"reset_pairwise", (PyCFunction)GBP_reset_pairwise_py, METH_VARARGS, "Given two inputs, as indices of nodes between an edge this resets that edge to provide 'no relationship' between the nodes. Can do one edge with two integers or a set of edges with two numpy arrays. Can give one input as an integer and another as a numpy array to do a list of edges that all include the same node. Can omit the second parameter to wipe out all edges for a given node or set of nodes."},
 
 //{"unary", (PyCFunction)GBP_unary_py, METH_VARARGS, "Given three parameters - the node indices, then the means and then the precision values (inverse variance/inverse squared standard deviation). It then updates the nodes by multiplying the unary term already in the node with the given, noting that this is a set for a node that has not yet had unary called on it/been reset. For the node indicies it accepts an integer, a slice or a numpy array. For the mean and precision it accepts either floats or a numpy array, noting that if an array is too small it will be accessed modulus the number of nodes provided."},
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
