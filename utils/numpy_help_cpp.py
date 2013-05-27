# -*- coding: utf-8 -*-

# Copyright (c) 2011, Tom SF Haines
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#  * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



from start_cpp import start_cpp



# Defines helper functions for accessing numpy arrays...
numpy_util_code = start_cpp() + """
#ifndef NUMPY_UTIL_CODE
#define NUMPY_UTIL_CODE

float & Float1D(PyArrayObject * arr, int index = 0)
{
 return *(float*)(arr->data + index*arr->strides[0]);
}

float & Float2D(PyArrayObject * arr, int index1 = 0, int index2 = 0)
{
 return *(float*)(arr->data + index1*arr->strides[0] + index2*arr->strides[1]);
}

float & Float3D(PyArrayObject * arr, int index1 = 0, int index2 = 0, int index3 = 0)
{
 return *(float*)(arr->data + index1*arr->strides[0] + index2*arr->strides[1] + index3*arr->strides[2]);
}


unsigned char & Byte1D(PyArrayObject * arr, int index = 0)
{
 //assert(arr->strides[0]==sizeof(unsigned char));
 return *(unsigned char*)(arr->data + index*arr->strides[0]);
}

unsigned char & Byte2D(PyArrayObject * arr, int index1 = 0, int index2 = 0)
{
 //assert(arr->strides[0]==sizeof(unsigned char));
 return *(unsigned char*)(arr->data + index1*arr->strides[0] + index2*arr->strides[1]);
}

unsigned char & Byte3D(PyArrayObject * arr, int index1 = 0, int index2 = 0, int index3 = 0)
{
 //assert(arr->strides[0]==sizeof(unsigned char));
 return *(unsigned char*)(arr->data + index1*arr->strides[0] + index2*arr->strides[1] + index3*arr->strides[2]);
}


int & Int1D(PyArrayObject * arr, int index = 0)
{
 //assert(arr->strides[0]==sizeof(int));
 return *(int*)(arr->data + index*arr->strides[0]);
}

int & Int2D(PyArrayObject * arr, int index1 = 0, int index2 = 0)
{
 //assert(arr->strides[0]==sizeof(int));
 return *(int*)(arr->data + index1*arr->strides[0] + index2*arr->strides[1]);
}

int & Int3D(PyArrayObject * arr, int index1 = 0, int index2 = 0, int index3 = 0)
{
 //assert(arr->strides[0]==sizeof(int));
 return *(int*)(arr->data + index1*arr->strides[0] + index2*arr->strides[1] + index3*arr->strides[2]);
}

#endif
"""
