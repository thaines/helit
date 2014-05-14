// Copyright 2014 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



#include <Python.h>
#include <structmember.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>


#include "summary.h"
#include "information.h"
#include "learner.h"

#include "frf_c.h"



void Forest_new(Forest * this)
{
 this->trees = 0;
}

void Forest_dealloc(Forest * this)
{

}


static PyObject * Forest_new_py(PyTypeObject * type, PyObject * args, PyObject * kwds)
{
 // Allocate the object...
  Forest * self = (Forest*)type->tp_alloc(type, 0);

 // On success construct it...
  if (self!=NULL) Forest_new(self);

 // Return the new object...
  return (PyObject*)self;
}

static void Forest_dealloc_py(Forest * self)
{
 Forest_dealloc(self);
 self->ob_type->tp_free((PyObject*)self);
}



static PyObject * Forest_summary_list_py(Forest * self, PyObject * args)
{
 // Create the return list...
  PyObject * ret = PyList_New(0);
  
 // Add each random variable type in turn...
  int i = 0;
  while (ListSummary[i]!=NULL)
  {
   char code[2];
   code[0] = ListSummary[i]->code;
   code[1] = 0;
   
   PyObject * dic = Py_BuildValue("{ssssss}", "code", code, "name", ListSummary[i]->name, "description", ListSummary[i]->description);
   
   PyList_Append(ret, dic);
   Py_DECREF(dic);
   
   ++i; 
  }
 
 // Return...
  return ret;
}


static PyObject * Forest_info_list_py(Forest * self, PyObject * args)
{
 // Create the return list...
  PyObject * ret = PyList_New(0);
  
 // Add each random variable type in turn...
  int i = 0;
  while (ListInfo[i]!=NULL)
  {
   char code[2];
   code[0] = ListInfo[i]->code;
   code[1] = 0;
   
   PyObject * dic = Py_BuildValue("{ssssss}", "code", code, "name", ListInfo[i]->name, "description", ListInfo[i]->description);
   
   PyList_Append(ret, dic);
   Py_DECREF(dic);
   
   ++i; 
  }
 
 // Return...
  return ret;
}


static PyObject * Forest_learner_list_py(Forest * self, PyObject * args)
{
 // Create the return list...
  PyObject * ret = PyList_New(0);
  
 // Add each random variable type in turn...
  int i = 0;
  while (ListLearner[i]!=NULL)
  {
   char code[2];
   code[0] = ListLearner[i]->code;
   code[1] = 0;
   
   char test[2];
   test[0] = ListLearner[i]->code_test;
   test[1] = 0;
   
   PyObject * dic = Py_BuildValue("{ssssssss}", "code", code, "name", ListLearner[i]->name, "description", ListLearner[i]->description, "test", test);
   
   PyList_Append(ret, dic);
   Py_DECREF(dic);
   
   ++i; 
  }
 
 // Return...
  return ret;
}



static PyMemberDef Forest_members[] =
{
 {"trees", T_INT, offsetof(Forest, trees), READONLY, "Number of trees in the forest."},
 {NULL}
};



static PyMethodDef Forest_methods[] =
{
 {"summary_list", (PyCFunction)Forest_summary_list_py, METH_NOARGS | METH_STATIC, "A static method that returns a list of summary types, as dictionaries. The summary types define how a particular target variable leaf summarises the exemplars that land in it, per output feature, and in what form that is returned to the user. Each dictionary contains 'code', one character string, for requesting it, a long form 'name' and a 'description'."},
 {"info_list", (PyCFunction)Forest_info_list_py, METH_NOARGS | METH_STATIC, "A static method that returns a list of information (as in entropy) types, as dictionaries. The information types define the goal of a split optimisation procedure - for any split they give a number for performing that split, typically entropy, that is to be minimised. This is on a per output feature basis. Each dictionary contains 'code', one character string, for requesting it, a long form 'name' and a 'description'."},
 {"learner_list", (PyCFunction)Forest_learner_list_py, METH_NOARGS | METH_STATIC, "A static method that returns a list of split learner types, as dictionaries. The learner types optimise and select a split for an input feature. Each dictionary contains 'code', one character string, for requesting it, a long form 'name' and a 'description'. Also contains 'test', a one character string, of the kind of test it generates, not that this is of any use as its internal use only."},
 {NULL}
};



static PyTypeObject ForestType =
{
 PyObject_HEAD_INIT(NULL)
 0,                                /*ob_size*/
 "frf_c.Forest",                   /*tp_name*/
 sizeof(Forest),                   /*tp_basicsize*/
 0,                                /*tp_itemsize*/
 (destructor)Forest_dealloc_py,    /*tp_dealloc*/
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
 "A random forest implimentation, designed with speed in mind, as well as I/O that doesn't suck so it can actually be saved/loaded to disk. Remains fairly modular so it can be customised to specific use cases. Supports both classification and regression, as well as multivariate output (Including mixed classification/regression!). Input feature vectors can contain both discrete and continuous variables; supports unknown values for discrete features.", /* tp_doc */
 0,                                /* tp_traverse */
 0,                                /* tp_clear */
 0,                                /* tp_richcompare */
 0,                                /* tp_weaklistoffset */
 0,                                /* tp_iter */
 0,                                /* tp_iternext */
 Forest_methods,                   /* tp_methods */
 Forest_members,                   /* tp_members */
 0,                                /* tp_getset */
 0,                                /* tp_base */
 0,                                /* tp_dict */
 0,                                /* tp_descr_get */
 0,                                /* tp_descr_set */
 0,                                /* tp_dictoffset */
 0,                                /* tp_init */
 0,                                /* tp_alloc */
 Forest_new_py,                    /* tp_new */
};



static PyMethodDef frf_c_methods[] =
{
 {NULL}
};



#ifndef PyMODINIT_FUNC
#define PyMODINIT_FUNC void
#endif

PyMODINIT_FUNC initfrf_c(void)
{
 // Create the module...
  PyObject * mod = Py_InitModule3("frf_c", frf_c_methods, "Provides a straight forward random forest implimentation that is designed to be fast and have good loading/saving capabilities, unlike all the other Python ones.");
 
 // Call some initialisation code...
  import_array();
  SetupCodeToTest();
 
 // Fill in the summary lookup table...
  int i;
  for (i=0; i<256; i++) CodeSummary[i] = NULL;
  
  i = 0;
  while (ListSummary[i]!=NULL)
  {
   CodeSummary[(unsigned char)ListSummary[i]->code] = ListSummary[i];
   i += 1;
  }
 
 // Register the Forest object...
  if (PyType_Ready(&ForestType) < 0) return;
 
  Py_INCREF(&ForestType);
  PyModule_AddObject(mod, "Forest", (PyObject*)&ForestType);
}
