# -*- coding: utf-8 -*-

# Copyright (c) 2011, Tom SF Haines
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#  * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



from start_cpp import start_cpp
from numpy_help_cpp import numpy_util_code



# Provides various functions to assist with manipulating python objects from c++ code.
python_obj_code = numpy_util_code + start_cpp() + """
#ifndef PYTHON_OBJ_CODE
#define PYTHON_OBJ_CODE

// Extracts a boolean from an object...
bool GetObjectBoolean(PyObject * obj, const char * name)
{
 PyObject * b = PyObject_GetAttrString(obj, name);
 bool ret = b!=Py_False;
 Py_DECREF(b);
 return ret;
}

// Extracts an int from an object...
int GetObjectInt(PyObject * obj, const char * name)
{
 PyObject * i = PyObject_GetAttrString(obj, name);
 int ret = PyInt_AsLong(i);
 Py_DECREF(i);
 return ret;
}

// Extracts a float from an object...
float GetObjectFloat(PyObject * obj, const char * name)
{
 PyObject * f = PyObject_GetAttrString(obj, name);
 float ret = PyFloat_AsDouble(f);
 Py_DECREF(f);
 return ret;
}

// Extracts an array from an object, returning it as a new[] unsigned char array. You can also pass in a pointer to an int to have the size of the array stored...
unsigned char * GetObjectByte1D(PyObject * obj, const char * name, int * size = 0)
{
 PyArrayObject * nao = (PyArrayObject*)PyObject_GetAttrString(obj, name);
 unsigned char * ret = new unsigned char[nao->dimensions[0]];
 if (size) *size = nao->dimensions[0];

 for (int i=0;i<nao->dimensions[0];i++) ret[i] = Byte1D(nao,i);

 Py_DECREF(nao);
 return ret;
}


// Extracts an array from an object, returning it as a new[] float array. You can also pass in a pointer to an int to have the size of the array stored...
float * GetObjectFloat1D(PyObject * obj, const char * name, int * size = 0)
{
 PyArrayObject * nao = (PyArrayObject*)PyObject_GetAttrString(obj, name);
 float * ret = new float[nao->dimensions[0]];
 if (size) *size = nao->dimensions[0];
 
 for (int i=0;i<nao->dimensions[0];i++) ret[i] = Float1D(nao,i);

 Py_DECREF(nao);
 return ret;
}

#endif
"""
