// Copyright 2012 Tom SF Haines

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.



#include <Python.h>
#include <structmember.h>
#include <numpy/arrayobject.h>

#include <fcntl.h>
#include <string.h>
#include <sys/time.h>

#include <CL/cl.h>

#include "manager_cl.h"
#include "open_cl_help.h"



typedef struct
{
 PyObject_HEAD

 cl_context context;
 cl_command_queue queue;
 cl_program program;


 int width;
 int height;

 cl_mem in;
 cl_mem out;

 cl_float * temp;

 cl_kernel from_rgb; size_t from_rgb_size;
 cl_kernel to_rgb;   size_t to_rgb_size;

 float chromaScale;
 float lumScale;
 float noiseFloor;

 float rot[3][3]; // [row][column]
 float inv_rot[3][3];
 float dud[7]; // We hand the above in as buffers to openCL for float16 - this just makes sure it doesn't read past the end of the structure.
} ColourBiasCL;



static PyObject * ColourBiasCL_new(PyTypeObject * type, PyObject * args, PyObject * kwds)
{
 ColourBiasCL * self = (ColourBiasCL*)type->tp_alloc(type, 0);

 if (self!=NULL)
 {
  self->context = NULL;
  self->queue = NULL;
  self->program = NULL;

  self->width = 0;
  self->height = 0;

  self->in = NULL;
  self->out = NULL;

  self->temp = NULL;

  self->from_rgb = NULL; self->from_rgb_size = 32;
  self->to_rgb = NULL; self->to_rgb_size = 32;

  self->chromaScale = 0.7176;
  self->lumScale = 0.7176;
  self->noiseFloor = 0.2;

  const float sr2 = 0.70710678;
  const float sr3 = 0.57735027;
  const float sr6 = 0.40824829;

  self->rot[0][0] = sr3;      self->rot[0][1] = sr3; self->rot[0][2] = sr3;
  self->rot[1][0] = 0.0;      self->rot[1][1] = sr2; self->rot[1][2] = -sr2;
  self->rot[2][0] = -2.0*sr6; self->rot[2][1] = sr6; self->rot[2][2] = sr6;

  self->inv_rot[0][0] = sr3; self->inv_rot[0][1] = 0.0;  self->inv_rot[0][2] = -2.0*sr6;
  self->inv_rot[1][0] = sr3; self->inv_rot[1][1] = sr2;  self->inv_rot[1][2] = sr6;
  self->inv_rot[2][0] = sr3; self->inv_rot[2][1] = -sr2; self->inv_rot[2][2] = sr6;
 }

 return (PyObject*)self;
}

static void ColourBiasCL_dealloc(ColourBiasCL * self)
{
 clFinish(self->queue);


 clReleaseKernel(self->from_rgb);
 clReleaseKernel(self->to_rgb);

 free(self->temp);

 clReleaseMemObject(self->out);
 clReleaseMemObject(self->in);

 clReleaseProgram(self->program);
 clReleaseCommandQueue(self->queue);
 clReleaseContext(self->context);

 self->ob_type->tp_free((PyObject*)self);
}



static PyMemberDef ColourBiasCL_members[] =
{
 {"width", T_INT, offsetof(ColourBiasCL, width), READONLY, "width of each frame"},
 {"height", T_INT, offsetof(ColourBiasCL, height), READONLY, "height of each frame"},
 {"chromaScale", T_FLOAT, offsetof(ColourBiasCL, chromaScale), 0, "Scaler to apply to the chromacity component."},
 {"lumScale", T_FLOAT, offsetof(ColourBiasCL, lumScale), 0, "Scaler to apply to the luminance component."},
 {"noiseFloor", T_FLOAT, offsetof(ColourBiasCL, noiseFloor), 0, "Point where it prioritises noise reduction over chromacity-luminance seperation."},
 {NULL}
};



static int ColourBiasCL_init(ColourBiasCL * self, PyObject * args, PyObject * kwds)
{
 // Extract the parameters...
  ManagerCL * managerCL;
  int width, height;
  char * path;
  if (!PyArg_ParseTuple(args, "Oiis", &managerCL, &width, &height, &path)) return -1;

  self->width = width;
  self->height = height;

  cl_int status;

 // Get the context and queue from the manager...
  self->context = managerCL->context;
  clRetainContext(self->context);

  self->queue = managerCL->queue;
  clRetainCommandQueue(self->queue);

 // Load and compile the program...
  const char * file = "colour_bias_cl.cl";
  self->program = prep_program(path, file, self->context, managerCL->device);
  if (self->program==NULL) return -1;

 // Create the required memory blocks...
  self->in = clCreateBuffer(self->context, CL_MEM_READ_ONLY, (self->height+16)*self->width*4*sizeof(cl_float), NULL, &status); // Extra storage seems to fix a bug I can't find:-( Been working on it most of a day so I'm giving up, at least for now - probably blind to what my code is actually doing.
  if (status!=CL_SUCCESS) return -1;

  self->out = clCreateBuffer(self->context, CL_MEM_WRITE_ONLY, self->height*self->width*4*sizeof(cl_float), NULL, &status);
  if (status!=CL_SUCCESS) return -1;

 // Temporary, for moving stuff into and out of the graphics card...
  self->temp = (cl_float*)malloc(width*height*4*sizeof(cl_float));

 // Create the kernel to convert from rgb...
  self->from_rgb = clCreateKernel(self->program, "from_rgb", &status);
  if (status!=CL_SUCCESS) return -1;

  status |= clSetKernelArg(self->from_rgb, 0, sizeof(cl_int), &self->width);
  status |= clSetKernelArg(self->from_rgb, 1, sizeof(cl_float), &self->chromaScale);
  status |= clSetKernelArg(self->from_rgb, 2, sizeof(cl_float), &self->lumScale);
  status |= clSetKernelArg(self->from_rgb, 3, sizeof(cl_float), &self->noiseFloor);
  status |= clSetKernelArg(self->from_rgb, 4, sizeof(cl_float16), self->rot);
  status |= clSetKernelArg(self->from_rgb, 5, sizeof(cl_mem), &self->in);
  status |= clSetKernelArg(self->from_rgb, 6, sizeof(cl_mem), &self->out);

  status |= clGetKernelWorkGroupInfo(self->from_rgb, managerCL->device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &self->from_rgb_size, NULL);

  if (status!=CL_SUCCESS) return -1;

 // Create the kernel to convert to rgb...
  self->to_rgb = clCreateKernel(self->program, "to_rgb", &status);
  if (status!=CL_SUCCESS) return -1;

  status |= clSetKernelArg(self->to_rgb, 0, sizeof(cl_int), &self->width);
  status |= clSetKernelArg(self->to_rgb, 1, sizeof(cl_float), &self->chromaScale);
  status |= clSetKernelArg(self->to_rgb, 2, sizeof(cl_float), &self->lumScale);
  status |= clSetKernelArg(self->to_rgb, 3, sizeof(cl_float), &self->noiseFloor);
  status |= clSetKernelArg(self->to_rgb, 4, sizeof(cl_float16), self->inv_rot);
  status |= clSetKernelArg(self->to_rgb, 5, sizeof(cl_mem), &self->in);
  status |= clSetKernelArg(self->to_rgb, 6, sizeof(cl_mem), &self->out);

  status |= clGetKernelWorkGroupInfo(self->to_rgb, managerCL->device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &self->to_rgb_size, NULL);

  if (status!=CL_SUCCESS) return -1;

 // Return...
  return 0;
}



static PyObject * ColourBiasCL_post_change(ColourBiasCL * self, PyObject * args)
{
 cl_int status = CL_SUCCESS;

 status |= clSetKernelArg(self->from_rgb, 1, sizeof(cl_float), &self->chromaScale);
 status |= clSetKernelArg(self->from_rgb, 2, sizeof(cl_float), &self->lumScale);
 status |= clSetKernelArg(self->from_rgb, 3, sizeof(cl_float), &self->noiseFloor);

 status |= clSetKernelArg(self->to_rgb, 1, sizeof(cl_float), &self->chromaScale);
 status |= clSetKernelArg(self->to_rgb, 2, sizeof(cl_float), &self->lumScale);
 status |= clSetKernelArg(self->to_rgb, 3, sizeof(cl_float), &self->noiseFloor);

 if (status!=CL_SUCCESS) return NULL;

 Py_INCREF(Py_None);
 return Py_None;
}



static PyObject * ColourBiasCL_from_rgb(ColourBiasCL * self, PyObject * args)
{
 // Get the parameters...
  PyArrayObject * in;
  if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &in)) return NULL;

 // Copy the image to the device...
  int y, x, i;
  for (y=0; y<self->height; y++)
  {
   for (x=0; x<self->width; x++)
   {
    float * from = (float*)(in->data + y*in->strides[0] + x*in->strides[1]);
    cl_float * to = self->temp + (y*self->width + x)*4;

    for (i=0; i<3; i++) to[i] = from[i];
    to[3] = 0.0;
   }
  }

  cl_int status = clEnqueueWriteBuffer(self->queue, self->in, CL_FALSE, 0, self->height*self->width*4*sizeof(cl_float), self->temp, 0, NULL, NULL);
  if (status!=CL_SUCCESS) return NULL;

  status = clEnqueueBarrier(self->queue);
  if (status!=CL_SUCCESS) return NULL;

 // Run the kernel on it...
  size_t work_size[2];
  size_t block_size[2];

  work_size[0] = self->width;
  work_size[1] = self->height;
  calc_block_size(self->from_rgb_size, 2, work_size, block_size, 0);

  status = clEnqueueNDRangeKernel(self->queue, self->from_rgb, 2, NULL, work_size, block_size, 0, NULL, NULL);
  if (status!=CL_SUCCESS) return NULL;

 // Stick a barrier in so its definatly done ready for the next thing...
  status = clEnqueueBarrier(self->queue);
  if (status!=CL_SUCCESS) return NULL;

 // Return...
  Py_INCREF(Py_None);
  return Py_None;
}



static PyObject * ColourBiasCL_to_rgb(ColourBiasCL * self, PyObject * args)
{
 // Get the parameters...
  PyArrayObject * in;
  if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &in)) return NULL;

 // Copy the image to the device...
  int y, x, i;
  for (y=0;y<self->height;y++)
  {
   for (x=0;x<self->width;x++)
   {
    float * from = (float*)(in->data + y*in->strides[0] + x*in->strides[1]);
    cl_float * to = self->temp + (y*self->width + x)*4;

    for (i=0;i<3;i++) to[i] = from[i];
    to[3] = 0.0;
   }
  }

  cl_int status = clEnqueueWriteBuffer(self->queue, self->in, CL_FALSE, 0, self->height*self->width*4*sizeof(cl_float), self->temp, 0, NULL, NULL);
  if (status!=CL_SUCCESS) return NULL;

  status = clEnqueueBarrier(self->queue);
  if (status!=CL_SUCCESS) return NULL;

 // Run the kernel on it...
  size_t work_size[2];
  size_t block_size[2];

  work_size[0] = self->width;
  work_size[1] = self->height;
  calc_block_size(self->to_rgb_size, 2, work_size, block_size, 0);

  status = clEnqueueNDRangeKernel(self->queue, self->to_rgb, 2, NULL, work_size, block_size, 0, NULL, NULL);
  if (status!=CL_SUCCESS) return NULL;

 // Stick a barrier in so its definatly done ready for the next thing...
  status = clEnqueueBarrier(self->queue);
  if (status!=CL_SUCCESS) return NULL;

 // Return...
  Py_INCREF(Py_None);
  return Py_None;
}



static PyObject * ColourBiasCL_fetch(ColourBiasCL * self, PyObject * args)
{
 // Get the parameters...
  PyArrayObject * out;
  if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &out)) return NULL;

  // Enqueue extracting the result...
  cl_int status = clEnqueueReadBuffer(self->queue, self->out, CL_FALSE, 0, self->height*self->width*4*sizeof(cl_float), self->temp, 0, NULL, NULL);
  if (status!=CL_SUCCESS) return NULL;

 // Wait till the queue is done...
  status = clFinish(self->queue);
  if (status!=CL_SUCCESS) return NULL;

 // Write the result into the output image...
  int y, x, i;
  for (y=0;y<self->height;y++)
  {
   for (x=0;x<self->width;x++)
   {
    cl_float * from = self->temp + (y*self->width + x)*4;
    float * to = (float*)(out->data + y*out->strides[0] + x*out->strides[1]);

    for (i=0;i<3;i++) to[i] = from[i];
   }
  }

 // Return...
  Py_INCREF(Py_None);
  return Py_None;
}



static PyMethodDef ColourBiasCL_methods[] =
{
 {"post_change", (PyCFunction)ColourBiasCL_post_change, METH_VARARGS, "If you edit any of the editable -aprameters you must call this afterwards so it can send those changes to the graphics card."},
 {"from_rgb", (PyCFunction)ColourBiasCL_from_rgb, METH_VARARGS, "Given a numpy array as an input image does the conversion from rgb to the adjusted colour space."},
 {"to_rgb", (PyCFunction)ColourBiasCL_to_rgb, METH_VARARGS, "Given a numpy array as an input image does the conversion from the adjusted colour space to rgb."},
 {"fetch", (PyCFunction)ColourBiasCL_fetch, METH_VARARGS, "Given a numpy array this dumps the output of the last call into it."},
 {NULL}
};



static PyTypeObject ColourBiasCLType =
{
 PyObject_HEAD_INIT(NULL)
 0,                                 /*ob_size*/
 "colour_bias_cl.ColourBiasCL",     /*tp_name*/
 sizeof(ColourBiasCL),              /*tp_basicsize*/
 0,                                 /*tp_itemsize*/
 (destructor)ColourBiasCL_dealloc,  /*tp_dealloc*/
 0,                                 /*tp_print*/
 0,                                 /*tp_getattr*/
 0,                                 /*tp_setattr*/
 0,                                 /*tp_compare*/
 0,                                 /*tp_repr*/
 0,                                 /*tp_as_number*/
 0,                                 /*tp_as_sequence*/
 0,                                 /*tp_as_mapping*/
 0,                                 /*tp_hash */
 0,                                 /*tp_call*/
 0,                                 /*tp_str*/
 0,                                 /*tp_getattro*/
 0,                                 /*tp_setattro*/
 0,                                 /*tp_as_buffer*/
 Py_TPFLAGS_DEFAULT,                /*tp_flags*/
 "Does conversion to and from a colour space where chromacity and luminance are separated, including a bias term that allows you to control the influence of chromacity vs luminance when taking distances in the space. Volume of the space is always one.", /* tp_doc */
 0,                                 /* tp_traverse */
 0,                                 /* tp_clear */
 0,                                 /* tp_richcompare */
 0,                                 /* tp_weaklistoffset */
 0,                                 /* tp_iter */
 0,                                 /* tp_iternext */
 ColourBiasCL_methods,              /* tp_methods */
 ColourBiasCL_members,              /* tp_members */
 0,                                 /* tp_getset */
 0,                                 /* tp_base */
 0,                                 /* tp_dict */
 0,                                 /* tp_descr_get */
 0,                                 /* tp_descr_set */
 0,                                 /* tp_dictoffset */
 (initproc)ColourBiasCL_init,       /* tp_init */
 0,                                 /* tp_alloc */
 ColourBiasCL_new,                  /* tp_new */
};



static PyMethodDef colour_bias_cl_methods[] =
{
 {NULL}
};



#ifndef PyMODINIT_FUNC
#define PyMODINIT_FUNC void
#endif

PyMODINIT_FUNC initcolour_bias_cl(void)
{
 PyObject * mod = Py_InitModule3("colour_bias_cl", colour_bias_cl_methods, "Provides a conversion to and from a useful colour model.");
 import_array();

 if (PyType_Ready(&ColourBiasCLType) < 0) return;

 Py_INCREF(&ColourBiasCLType);
 PyModule_AddObject(mod, "ColourBiasCL", (PyObject*)&ColourBiasCLType);
}
