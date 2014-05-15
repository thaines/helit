// Copyright 2014 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



#include <Python.h>
#include <structmember.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <limits.h>


#include "summary.h"
#include "information.h"
#include "learner.h"

#include "frf_c.h"



// Predeclerations...
static PyTypeObject TreeBufferType;
static PyTypeObject ForestType;



//
// The Tree type...
//

void TreeBuffer_new(TreeBuffer * this, size_t size)
{
 this->size = size;
 this->tree = malloc(size);
 
 this->ready = 0;
}

void TreeBuffer_dealloc(TreeBuffer * this)
{
 if (this->ready!=0) Tree_deinit(this->tree);
 free(this->tree);
}


static PyObject * TreeBuffer_new_py(PyTypeObject * type, PyObject * args, PyObject * kwds)
{
 // Extract the parameters...
  size_t size;
  if (!PyArg_ParseTuple(args, "n", &size)) return NULL;
  
 // Allocate the object...
  TreeBuffer * self = (TreeBuffer*)type->tp_alloc(type, 0);

 // On success construct it...
  if (self!=NULL) TreeBuffer_new(self, size);

 // Return the new object...
  return (PyObject*)self;
}

static void TreeBuffer_dealloc_py(TreeBuffer * self)
{
 TreeBuffer_dealloc(self);
 self->ob_type->tp_free((PyObject*)self);
}



static PyObject * TreeBuffer_head_size_py(Forest * self, PyObject * args)
{
 return Py_BuildValue("n", Tree_head_size());
}

static PyObject * TreeBuffer_size_from_head_py(Forest * self, PyObject * args)
{
 // Read in the header...
  const char * data;
  int data_size;
  if (!PyArg_ParseTuple(args, "s#", &data, &data_size)) return NULL;
  
 // Verify its safe...
  if (data_size<Tree_head_size())
  {
   PyErr_SetString(PyExc_RuntimeError, "Data block too small to be a Tree header");
   return NULL; 
  }
  
  if (Tree_safe((Tree*)data)==0)
  {
   return NULL; 
  }
 
 // Extract the value and return it...
  size_t size = Tree_size((Tree*)(data));
  return Py_BuildValue("n", size);
}



static Py_ssize_t TreeBuffer_buffer_get(TreeBuffer * self, Py_ssize_t index, const void **ptr)
{
 if (index!=0)
 {
  PyErr_SetString(PyExc_SystemError, "Byte segment does not exist");
  return -1;
 }
 
 *ptr = (void*)self->tree;
 return self->size;
}

static Py_ssize_t TreeBuffer_buffer_segs(TreeBuffer * self, Py_ssize_t * lenp)
{
 if (lenp!=NULL) *lenp = self->size;
 return 1;
}

static int TreeBuffer_buffer_acquire(TreeBuffer * self, Py_buffer * view, int flags)
{
 if (view == NULL) return 0;
 return PyBuffer_FillInfo(view, (PyObject*)self, (void*)self->tree, self->size, 0, flags);
}

static void TreeBuffer_buffer_release(TreeBuffer * self, Py_buffer * view)
{
 // No-op
}


static PyBufferProcs TreeBuffer_as_buffer =
{
 (readbufferproc)TreeBuffer_buffer_get,
 (writebufferproc)TreeBuffer_buffer_get,
 (segcountproc)TreeBuffer_buffer_segs,
 NULL,
 (getbufferproc)TreeBuffer_buffer_acquire,
 (releasebufferproc)TreeBuffer_buffer_release,
};



static PyMemberDef TreeBuffer_members[] =
{
 {"size", T_INT, offsetof(TreeBuffer, size), READONLY, "How many bytes are contained within."},
 {NULL}
};



static PyMethodDef TreeBuffer_methods[] =
{
 {"head_size", (PyCFunction)TreeBuffer_head_size_py, METH_NOARGS | METH_STATIC, "Returns how many bytes are in the header of a Tree, so you can read the entire header from a stream."},
 {"size_from_head", (PyCFunction)TreeBuffer_size_from_head_py, METH_VARARGS | METH_STATIC, "Given the head, as a read-only buffer compatible object (string, return value of read(), numpy array.) this returns the size of the associated tree, or throws an error if there is something wrong."},
 {NULL}
};



static PyTypeObject TreeBufferType =
{
 PyObject_HEAD_INIT(NULL)
 0,                                /*ob_size*/
 "frf_c.Tree",                     /*tp_name*/
 sizeof(TreeBuffer),               /*tp_basicsize*/
 0,                                /*tp_itemsize*/
 (destructor)TreeBuffer_dealloc_py,/*tp_dealloc*/
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
 &TreeBuffer_as_buffer,            /*tp_as_buffer*/
 Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_NEWBUFFER, /*tp_flags*/
 "A tree within the Forest, but it provides no functionality - its only useful when attached to a Forest. Exists for loading Trees from a seperate source, such as a file or another Forest object (with a compatible configuration). Constructed with a size, in bytes, and impliments the memoryview(tree) interface, so you can extract/set the data within. The contents is entirly streamable, and therefore safe to be dumped to disk/socket etc. - file.write(my_tree) works, as does file.readinto(my_tree). It does have some static methods that provide useful information for loading a tree from a stream - size of the header, and header to total size.", /* tp_doc */
 0,                                /* tp_traverse */
 0,                                /* tp_clear */
 0,                                /* tp_richcompare */
 0,                                /* tp_weaklistoffset */
 0,                                /* tp_iter */
 0,                                /* tp_iternext */
 TreeBuffer_methods,               /* tp_methods */
 TreeBuffer_members,               /* tp_members */
 0,                                /* tp_getset */
 0,                                /* tp_base */
 0,                                /* tp_dict */
 0,                                /* tp_descr_get */
 0,                                /* tp_descr_set */
 0,                                /* tp_dictoffset */
 0,                                /* tp_init */
 0,                                /* tp_alloc */
 TreeBuffer_new_py,                /* tp_new */
};



//
// The Forest type...
//

void Forest_new(Forest * this)
{
 this->x_feat = 0; 
 this->y_feat = 0;
 
 this->summary_codes = NULL;
 this->info_codes = NULL;
 this->learn_codes = NULL;
 
 this->ready = 0;
 
 this->bootstrap = 1;
 this->opt_features = INT_MAX;
 this->min_exemplars = 1;
 this->max_splits = INT_MAX;
 
 int i;
 for (i=0; i<4; i++) this->key[i] = 0;
 
 this->info_ratios = NULL;
 
 this->trees = 0;
 this->tree = NULL;
}

void Forest_dealloc(Forest * this)
{
 free(this->summary_codes);
 free(this->info_codes);
 free(this->learn_codes);
 
 Py_XDECREF(this->info_ratios);
 
 int i;
 for (i=0; i<this->trees; i++)
 {
  Py_DECREF((PyObject*)this->tree[i]);
 }
 free(this->tree);
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



static PyObject * Forest_initial_size_py(Forest * self, PyObject * args)
{
 return Py_BuildValue("n", sizeof(ForestHeader));
}


static PyObject * Forest_size_from_initial_py(Forest * self, PyObject * args)
{
 // Read in the header...
  const char * data;
  int data_size;
  if (!PyArg_ParseTuple(args, "s#", &data, &data_size)) return NULL;
  
 // Verify its safe...
  if (data_size<sizeof(ForestHeader))
  {
   PyErr_SetString(PyExc_RuntimeError, "Data block too small to be a Forest initial header.");
   return NULL; 
  }
  ForestHeader * fh = (ForestHeader*)data;
  
  if ((fh->magic[0]!='F')||(fh->magic[1]!='R')||(fh->magic[2]!='F')||(fh->magic[3]!='F')||(fh->revision!=1))
  {
   PyErr_SetString(PyExc_ValueError, "Forest initial header appears corrupted.");
   return NULL; 
  }
 
 // Calculate the value and return it...
  size_t size = fh->size;
  return Py_BuildValue("n", size);  
}


static PyObject * Forest_load_py(Forest * self, PyObject * args)
{
 // Read in the header...
  const char * data;
  int data_size;
  if (!PyArg_ParseTuple(args, "s#", &data, &data_size)) return NULL;
  
 // Verify its safe...
  if (data_size<sizeof(ForestHeader))
  {
   PyErr_SetString(PyExc_RuntimeError, "Data block too small to be a Forest initial header.");
   return NULL; 
  }
  ForestHeader * fh = (ForestHeader*)data;
  
  if ((fh->magic[0]!='F')||(fh->magic[1]!='R')||(fh->magic[2]!='F')||(fh->magic[3]!='F')||(fh->revision!=1))
  {
   PyErr_SetString(PyExc_ValueError, "Forest initial header appears corrupted.");
   return NULL; 
  }
  
  if (fh->size<data_size)
  {
   PyErr_SetString(PyExc_RuntimeError, "Data block too small to be a Forest complete header.");
   return NULL; 
  }
  
 // Terminate any trees in the object at present...
  int i;
  for (i=0; i<self->trees; i++)
  {
   Py_DECREF((PyObject*)self->tree[i]);
  }
  
  free(self->tree);
  self->tree = NULL;
  self->trees = 0;
  
 // Extract and record all the values...
  self->x_feat = fh->x_feat;
  self->y_feat = fh->y_feat;
  
  free(self->summary_codes);
  self->summary_codes = (char*)malloc(self->y_feat+1);
  memcpy(self->summary_codes, fh->codes, self->y_feat);
  self->summary_codes[self->y_feat] = 0;
  
  free(self->info_codes);
  self->info_codes = (char*)malloc(self->y_feat+1);
  memcpy(self->info_codes, fh->codes + self->y_feat, self->y_feat);
  self->info_codes[self->y_feat] = 0;
  
  free(self->learn_codes);
  self->learn_codes = (char*)malloc(self->x_feat+1);
  memcpy(self->learn_codes, fh->codes + self->y_feat*2, self->x_feat);
  self->learn_codes[self->x_feat] = 0;
  
  self->ready = 1;
 
  self->bootstrap = fh->bootstrap;
  self->opt_features = fh->opt_features;
  self->min_exemplars = fh->min_exemplars;
  self->max_splits = fh->max_splits;
  
  for (i=0; i<4; i++)
  {
   self->key[i] = fh->key[i];
  }
  
  Py_XDECREF(self->info_ratios);
  self->info_ratios = NULL;
  if (fh->ratios!=0)
  {
   npy_intp dims[2] = {fh->ratios, fh->y_feat};
   self->info_ratios = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_FLOAT32);
   
   float * base = (float*)(fh->codes + self->y_feat *2 + self->x_feat);
   int j;
   for (j=0; j<fh->ratios; j++)
   {
    for (i=0; i<fh->y_feat; i++)
    {
     *(float*)PyArray_GETPTR2(self->info_ratios, j, i) = base[j * fh->y_feat + i];
    }
   }
  }
  
 // Return the number of trees...
  return Py_BuildValue("i", fh->trees);  
}



static PyObject * Forest_configure_py(Forest * self, PyObject * args)
{
 // Read in the header...
  const char * summary;
  const char * info;
  const char * learn;
  if (!PyArg_ParseTuple(args, "sss", &summary, &info, &learn)) return NULL;
  
 // Error check the strings...
  int summary_len = strlen(summary);
  int info_len = strlen(info);
  int learn_len = strlen(learn);
  
  if (summary_len!=info_len)
  {
   PyErr_SetString(PyExc_ValueError, "Summary type string and info type string must have the same length.");
   return NULL; 
  }
  
  int i;
  for (i=0; i<summary_len; i++)
  {
   if (CodeSummary[(unsigned char)summary[i]]==NULL)
   {
    PyErr_SetString(PyExc_ValueError, "Unrecognised summary code.");
    return NULL;
   }
  }
  
  for (i=0; i<info_len; i++)
  {
   int j = 0;
   while (ListInfo[j]!=NULL)
   {
    if (info[i]==ListInfo[j]->code) break;
    j += 1;  
   }
   
   if (info[i]!=ListInfo[j]->code)
   {
    PyErr_SetString(PyExc_ValueError, "Unrecognised info code.");
    return NULL;
   }
  }
  
  for (i=0; i<learn_len; i++)
  {
   int j = 0;
   while (ListLearner[j]!=NULL)
   {
    if (learn[i]==ListLearner[j]->code) break;
    j += 1;  
   }
   
   if (learn[i]!=ListLearner[j]->code)
   {
    PyErr_SetString(PyExc_ValueError, "Unrecognised learner code.");
    return NULL;
   }
  }
  
 // Remove all trees...
  for (i=0; i<self->trees; i++)
  {
   Py_DECREF((PyObject*)self->tree[i]);
  }
  
  free(self->tree);
  self->tree = NULL;
  self->trees = 0;
  
 // Record the feat sizes, codes and indicate ready...
  self->x_feat = learn_len;
  self->y_feat = summary_len;
  
  free(self->summary_codes); 
  self->summary_codes = malloc(self->y_feat+1);
  memcpy(self->summary_codes, summary, self->y_feat);
  self->summary_codes[self->y_feat] = 0;
  
  free(self->info_codes); 
  self->info_codes = malloc(self->y_feat+1);
  memcpy(self->info_codes, info, self->y_feat);
  self->info_codes[self->y_feat] = 0;
  
  free(self->learn_codes);
  self->learn_codes = malloc(self->x_feat+1);
  memcpy(self->learn_codes, learn, self->x_feat);
  self->learn_codes[self->x_feat] = 0;
    
  self->ready = 1;
  
 // Can't keep ratios...
  Py_XDECREF(self->info_ratios);
  self->info_ratios = NULL;
  
 // Return None...
  Py_INCREF(Py_None);
  return Py_None;
}


static PyObject * Forest_set_ratios_py(Forest * self, PyObject * args)
{
 // Get the passed in array...
  PyArrayObject * ratios;
  if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &ratios)) return NULL;
 
 // Check its compliant...
  if (PyArray_NDIM(ratios)!=2)
  {
   PyErr_SetString(PyExc_ValueError, "Ratios array required to be 2D.");
   return NULL;
  }
  
  if (PyArray_DIMS(ratios)[1]!=self->y_feat)
  {
   PyErr_SetString(PyExc_ValueError, "Second dimension of ratios array must match up with the number of output features.");
   return NULL; 
  }
 
 // Store it...
  Py_XDECREF(self->info_ratios);
  self->info_ratios = ratios;
  Py_INCREF(self->info_ratios);
 
 // Return None...
  Py_INCREF(Py_None);
  return Py_None;
}



static PyObject * Forest_save_py(Forest * self, PyObject * args)
{
 // Create a temporary object for the head...
 size_t size = sizeof(ForestHeader) + self->x_feat + 2*self->y_feat;
 if (self->info_ratios!=NULL)
 {
  size += PyArray_DIMS(self->info_ratios)[0] * PyArray_DIMS(self->info_ratios)[1] * sizeof(float); 
 }
 
 ForestHeader * fh = (ForestHeader*)malloc(size);
 
 // Fill it in...
  fh->magic[0] = 'F';
  fh->magic[1] = 'R';
  fh->magic[2] = 'F';
  fh->magic[3] = 'F';
  fh->revision = 1;
  fh->size = size;
  
  fh->trees = self->trees;
  
  fh->bootstrap = self->bootstrap;
  fh->opt_features = self->opt_features;
  fh->min_exemplars = self->min_exemplars;
  fh->max_splits = self->max_splits;
  
  int i;
  for (i=0; i<4; i++) fh->key[i] = self->key[i];
 
  fh->x_feat = self->x_feat;
  fh->y_feat = self->y_feat;
  fh->ratios = (self->info_ratios==NULL) ? 0 : PyArray_DIMS(self->info_ratios)[0];
  
  memcpy(fh->codes, self->summary_codes, self->y_feat);
  memcpy(fh->codes + self->y_feat, self->info_codes, self->y_feat);
  memcpy(fh->codes + 2*self->y_feat, self->learn_codes, self->x_feat);
  
  if (self->info_ratios!=NULL)
  {
   float * base = (float*)(fh->codes + 2*self->y_feat + self->x_feat);
   
   int j;
   for (j=0; j<fh->ratios; j++)
   {
    for (i=0; i<self->y_feat; i++)
    {
     base[j*self->y_feat+i] = *(float*)PyArray_GETPTR2(self->info_ratios, j, i);
    }
   }
  }
 
 // Create a ByteArray...
  PyObject * ret = PyByteArray_FromStringAndSize((char*)fh, size);
 
 // Clean up and return...
  free(fh);
  return ret;
}


static PyObject * Forest_clone_py(Forest * self, PyObject * args)
{
 // Can't clone objects that are not ready - thats silly...
  if (self->ready==0)
  {
   PyErr_SetString(PyExc_RuntimeError, "Cloning a Forest that is not ready is silly.");
   return NULL; 
  }
 
 // Create a new forest object...
  Forest * ret = (Forest*)ForestType.tp_alloc(&ForestType, 0);
  if (ret==NULL) return NULL;
 
 // Fill it in as an almost complete copy of this one...
  ret->x_feat = self->x_feat;
  ret->y_feat = self->y_feat;
  
  ret->summary_codes = strdup(self->summary_codes);
  ret->info_codes = strdup(self->info_codes);
  ret->learn_codes = strdup(self->learn_codes);
  
  ret->ready = 1;
  
  ret->bootstrap = self->bootstrap;
  ret->opt_features = self->opt_features;
  ret->min_exemplars = self->min_exemplars;
  ret->max_splits = self->max_splits;
  
  int i;
  for (i=0; i<4; i++) ret->key[i] = self->key[i];
  
  ret->info_ratios = self->info_ratios;
  Py_XINCREF(ret->info_ratios);
  
  ret->trees = 0;
  ret->tree = NULL;
  
 // Return...
  return (PyObject*)ret;
}



// ******************************** train to go here!


static PyObject * Forest_append_py(Forest * self, PyObject * args)
{
 // Parse the parameters...
  TreeBuffer * tree;
  if (!PyArg_ParseTuple(args, "O!", &TreeBufferType, &tree)) return NULL;
  
 // Dump it on the end...
  self->tree = (TreeBuffer**)realloc(self->tree, (self->trees + 1) * sizeof(TreeBuffer*));
  self->tree[self->trees] = tree;
  Py_INCREF(tree);
  self->trees += 1;
  
 // Return None...
  Py_INCREF(Py_None);
  return Py_None;
}



static Py_ssize_t Forest_Length(Forest * self)
{
 return self->trees; 
}

static PyObject * Forest_GetItem(Forest * self, Py_ssize_t i)
{
 if (i<0) i += self->trees;
 if ((i<0)||(i>=self->trees))
 {
  PyErr_SetString(PyExc_IndexError, "Forest index out of range.");
  return NULL; 
 }
 
 PyObject * ret = (PyObject*)self->tree[i];
 Py_INCREF(ret);
 return ret;
}

static int Forest_SetItem(Forest * self, Py_ssize_t i, PyObject * tree)
{
 if (i<0) i += self->trees;
 if ((i<0)||(i>=self->trees))
 {
  PyErr_SetString(PyExc_IndexError, "Forest index out of range.");
  return -1; 
 }
 
 if (PyObject_TypeCheck(tree, &TreeBufferType)==0)
 {
  PyErr_SetString(PyExc_ValueError, "Forest can only contain trees.");
  return -1;
 }
  
 Py_DECREF((PyObject*)self->tree[i]);
 self->tree[i] = (TreeBuffer*)tree;
 Py_INCREF((PyObject*)tree);
 
 return 0;
}

static PyObject * Forest_InPlaceConcat(Forest * self, PyObject * other)
{
 int i;
 // Verify the other is in fact a Forest...
  if (PyObject_TypeCheck(other, &ForestType)==0)
  {
   PyErr_SetString(PyExc_ValueError, "Only a forest can be appended to another forest.");
   return NULL;
  }
  Forest * rhs = (Forest*)other;
  
  if (self==rhs)
  {
   PyErr_SetString(PyExc_ValueError, "Appending a forest to itself makes no sense");
   return NULL;
  }
  
 // Grow the array, store the new trees, including incrimenting their ref counts...
  self->tree = (TreeBuffer**)realloc(self->tree, (self->trees + rhs->trees) * sizeof(TreeBuffer*));
  
  for (i=0; i<rhs->trees; i++)
  {
   self->tree[self->trees+i] = rhs->tree[i];
   Py_INCREF(self->tree[self->trees+i]);
  }
  
  self->trees += rhs->trees;
  
 // Return self...
  Py_INCREF(self);
  return (PyObject*)self;
}


static PySequenceMethods Forest_as_sequence =
{
 (lenfunc)Forest_Length,             /* sq_length */
 NULL,                               /* sq_concat */
 NULL,                               /* sq_repeat */
 (ssizeargfunc)Forest_GetItem,       /* sq_item */
 NULL,                               /* sq_slice */
 (ssizeobjargproc)Forest_SetItem,    /* sq_ass_item */
 NULL,                               /* sq_ass_slice */
 NULL,                               /* sq_contains */
 (binaryfunc)Forest_InPlaceConcat,   /* sq_inplace_concat */
 NULL,                               /* sq_inplace_repeat */
};



static PyMemberDef Forest_members[] =
{
 {"x_feat", T_INT, offsetof(Forest, x_feat), READONLY, "Number of features it expect of the input x data matrix, which is the data matrix that is known."},
 {"y_feat", T_INT, offsetof(Forest, y_feat), READONLY, "Number of features it expect of the output y data matrix, which is the data matrix that it is learning to predeict from the x matrix."},
 
 {"summary_codes", T_STRING, offsetof(Forest, summary_codes), READONLY, "Returns the codes used for creating summary objects, a string indexed by y feature that gives the code of the summary to be used for that features."},
 {"info_codes", T_STRING, offsetof(Forest, info_codes), READONLY, "Returns the codes used for creating information gain objects, a string indexed by y feature that gives the code of the info calculator to be used for that features."},
 {"learn_codes", T_STRING, offsetof(Forest, learn_codes), READONLY, "Returns the codes used for creating learners, a string indexed by x feature that gives the code of the leaner to sugest splits for that features."}, 
 
 {"ready", T_BOOL, offsetof(Forest, ready), READONLY, "True if its safe to train trees, False if the forest has not been setup - i.e. neither a header has been loaded not a header has been configured. Note that safe to train is not the same as safe to predict - it could contain 0 trees (check with len(forest))."},
 
 {"bootstrap", T_BOOL, offsetof(Forest, bootstrap), 0, "True to train trees on bootsrap draws of the training data (The default), False to just train on everything."},
 {"opt_features", T_INT, offsetof(Forest, opt_features), 0, "Number of features to randomly select to try optimising for each split in the forest. Defaults so high as to be irrelevant."},
 {"min_exemplars", T_INT, offsetof(Forest, min_exemplars), 0, "Minimum number of exemplars to allow in a node - no node should ever have less than this count in it. Defaults to 1, making it irrelevant."},
 {"max_splits", T_INT, offsetof(Forest, max_splits), 0, "Maximum number of splits when building a new tree. Defaults so high you will run out of memeory first."},
 
 {"seed0", T_UINT, offsetof(Forest, key[0]), 0, "One of the 4 seeds that drives the random number generator used during tree construction. Will change as its moved along by the need for more pseudo-random data."},
 {"seed1", T_UINT, offsetof(Forest, key[1]), 0, "One of the 4 seeds that drives the random number generator used during tree construction. Will change as its moved along by the need for more pseudo-random data."},
 {"seed2", T_UINT, offsetof(Forest, key[2]), 0, "One of the 4 seeds that drives the random number generator used during tree construction. Will change as its moved along by the need for more pseudo-random data."},
 {"seed3", T_UINT, offsetof(Forest, key[3]), 0, "One of the 4 seeds that drives the random number generator used during tree construction. Will change as its moved along by the need for more pseudo-random data."},
 
 {"info_ratios", T_OBJECT, offsetof(Forest, info_ratios), READONLY, "Returns the information ratios numpy array, if it has been set. A 2D array, indexed by depth in the first dimension, by y-feature in the second (First dimension accessed modulus). Returns the weight of the entropy from that feature when summing them together, so you can control the objective of the tree."},
 
 {"trees", T_INT, offsetof(Forest, trees), READONLY, "Number of trees in the forest."},
 {NULL}
};



static PyMethodDef Forest_methods[] =
{
 {"summary_list", (PyCFunction)Forest_summary_list_py, METH_NOARGS | METH_STATIC, "A static method that returns a list of summary types, as dictionaries. The summary types define how a particular target variable leaf summarises the exemplars that land in it, per output feature, and in what form that is returned to the user. Each dictionary contains 'code', one character string, for requesting it, a long form 'name' and a 'description'."},
 {"info_list", (PyCFunction)Forest_info_list_py, METH_NOARGS | METH_STATIC, "A static method that returns a list of information (as in entropy) types, as dictionaries. The information types define the goal of a split optimisation procedure - for any split they give a number for performing that split, typically entropy, that is to be minimised. This is on a per output feature basis. Each dictionary contains 'code', one character string, for requesting it, a long form 'name' and a 'description'."},
 {"learner_list", (PyCFunction)Forest_learner_list_py, METH_NOARGS | METH_STATIC, "A static method that returns a list of split learner types, as dictionaries. The learner types optimise and select a split for an input feature. Each dictionary contains 'code', one character string, for requesting it, a long form 'name' and a 'description'. Also contains 'test', a one character string, of the kind of test it generates, not that this is of any use as its internal use only."},
 
 {"initial_size", (PyCFunction)Forest_initial_size_py, METH_NOARGS | METH_STATIC, "Returns the size of a forests initial header, so you can load that to get basic information about the forest, then load the rest of the header then the trees."},
 {"size_from_initial", (PyCFunction)Forest_size_from_initial_py, METH_VARARGS | METH_STATIC, "Given the inital header, as a read-only buffer compatible object (string, return value of read(), numpy array.) this returns the size of the entire header, or throws an error if there is something wrong."},
 {"load", (PyCFunction)Forest_load_py, METH_VARARGS, "Given an entire header (See initial_size and size_from_initial for how to do this) as a read-only buffer compatible object this initialises this object to those settings. If there are any trees they will be terminated. Can raise a whole litany of errors. Returns how many trees follow the header - it is upto the user to then load them from whatever stream is providing the information."},
 
 {"configure", (PyCFunction)Forest_configure_py, METH_VARARGS, "Configures the object - must be called before any learning, unless load is used instead. Takes three tag strings - first for summary, next for info, final for learn. Summary and info must have the same length, being the number of output features in length, whilst learn is the length of the number of input features. See the various _list static methods for lists of possible codes. Will throw an error if any are wrong."},
 {"set_ratios", (PyCFunction)Forest_set_ratios_py, METH_VARARGS, "Sets the ratios to use when balancing the priority of learning each output feature - must be a 2D numpy array with the first dimension indexed by depth, the second indexed by output feature. The depth is indexed modulus its size, so you can have a repeating structure. Can only be called on a ready Forest, and keeps a pointer to the array, so you can change its values afterwards if you want."},
 
 {"save", (PyCFunction)Forest_save_py, METH_NOARGS, "Returns the header for this Forest, such that it can be saved to disk/stream etc. and later loaded. Return value is a bytearray"},
 {"clone", (PyCFunction)Forest_clone_py, METH_NOARGS, "Returns a new Forest with the exact same setup as this one, but no trees. Be warned that it also duplicates the key for the random number generator, so any new trees trained with the same data in both will be identical - may want to change the key after cloning."},
 
 //{"train", (PyCFunction)Forest_train_py, METH_VARARGS, "Trains and appends more trees to this Forest - first parameter is the x/input data matrix, second is the y/output data matrix, third is the number of trees, which defaults to 1. Data matrices can be either a numpy array (exemplars X features) or a list of numpy arrays that are implicity joined to make the final data matrix - good when you want both continuous and discrete types. When a list 1D array are assumed to be indexed by exemplar. If boostrap is true this returns the out of bag error - a 2D array indexed by new tree then feature of how much error exists in that channel on average."},
 {"append", (PyCFunction)Forest_append_py, METH_VARARGS, "Appends a Tree to the Forest, that has presumably been trained in another Forest and is now to be merged. Note that the Forest must be compatible (identical type codes given to configure), and this is not checked. Break this and expect the fan to get very brown."},
 
 //{"predict", (PyCFunction)Forest_predict_py, METH_VARARGS, "Given an x/input data matrix returns what it knows about the output data matrix. Return will be a list indexed by feature, with the contents defined by the summary codes (Typically a dictionary of arrays, often of things like 'prob' or 'mean')."},
 //{"score", (PyCFunction)Forest_score_py, METH_VARARGS, "Given a x/input data matrix and a y/output data matrix of true answers this returns an array, indexed by tree then output features, of how much error each tree has on that feature, divided by the number of exemplars. Same as the oob calculation, but for a hold out set etc. Array tree index will align with the list of trees this can be accessed as."},
 
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
 &Forest_as_sequence,              /*tp_as_sequence*/
 0,                                /*tp_as_mapping*/
 0,                                /*tp_hash */
 0,                                /*tp_call*/
 0,                                /*tp_str*/
 0,                                /*tp_getattro*/
 0,                                /*tp_setattro*/
 0,                                /*tp_as_buffer*/
 Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
 "A random forest implimentation, designed with speed in mind, as well as I/O that doesn't suck (almost - should be 32/64 bit safe, but not endian change safe.) so it can actually be saved/loaded to disk. Remains fairly modular so it can be customised to specific use cases. Supports both classification and regression, as well as multivariate output (Including mixed classification/regression!). Input feature vectors can contain both discrete and continuous variables; kinda supports unknown values for discrete features. Provides the sequence interface, to access the individual Tree objects contained within.", /* tp_doc */
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
 
 // Register the Tree object...
  if (PyType_Ready(&TreeBufferType) < 0) return;
    
  Py_INCREF(&TreeBufferType);
  PyModule_AddObject(mod, "Tree", (PyObject*)&TreeBufferType);
 
 // Register the Forest object...
  if (PyType_Ready(&ForestType) < 0) return;
 
  Py_INCREF(&ForestType);
  PyModule_AddObject(mod, "Forest", (PyObject*)&ForestType);
}
