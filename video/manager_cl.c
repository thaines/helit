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



#include "manager_cl.h"



static PyObject * ManagerCL_new(PyTypeObject * type, PyObject * args, PyObject * kwds)
{
 ManagerCL * self = (ManagerCL*)type->tp_alloc(type, 0);

 if (self!=NULL)
 {
  self->context = NULL;
  self->queue = NULL;
  self->device = NULL;
 }

 return (PyObject*)self;
}



static void ManagerCL_dealloc(ManagerCL * self)
{
 clFinish(self->queue);
 
 clReleaseCommandQueue(self->queue);
 clReleaseContext(self->context);

 self->ob_type->tp_free((PyObject*)self);
}



static PyMemberDef ManagerCL_members[] =
{
 {NULL}
};



static int ManagerCL_init(ManagerCL * self, PyObject * args, PyObject * kwds)
{
 // We need to find out how many openCL platforms are avaliable on the current system...
  cl_uint platformCount;
  cl_int status = clGetPlatformIDs(0, NULL, &platformCount);
  if (status!=CL_SUCCESS||platformCount==0) return -1;


 // Iterate platforms, and the devices on those platforms, scoring each in turn to find the 'fastest' device...
  float bestScore = -1.0;
  cl_device_id bestDevice;
  cl_device_type bestType = -1;

  // Get an array of platform id-s...
   cl_platform_id * platform = (cl_platform_id*)malloc(sizeof(cl_platform_id)*platformCount);
   status = clGetPlatformIDs(platformCount, platform, NULL);
   if (status!=CL_SUCCESS) return -1;

  // Loop and analyse each platform in turn...
   int p;
   for (p=0;p<platformCount;p++)
   {
    // Get the number of devices provided by the platform...
     cl_uint deviceCount;
     status = clGetDeviceIDs(platform[p], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
     if (status!=CL_SUCCESS)
     {
      free(platform);
      return -1;
     }

    // Get an array of device id's...
     cl_device_id * device = (cl_device_id*)malloc(sizeof(cl_device_id)*deviceCount);
     status = clGetDeviceIDs(platform[p], CL_DEVICE_TYPE_ALL, deviceCount, device, NULL);
     if (status!=CL_SUCCESS)
     {
      free(device);
      return -1;
     }

    // Loop and have a close look at each device in sequence...
     int d;
     for (d=0;d<deviceCount;d++)
     {
      // Verify it has enough memory...
       cl_ulong mem;
       status = clGetDeviceInfo(device[d], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem), &mem, NULL);
       if (status!=CL_SUCCESS) continue;

       cl_ulong memReq = 8*1920*1080*3*sizeof(cl_float); // Set a memory requirement of at least full HD 16 images - realistically speaking this is a requirement that the device have at least quater of a gig of memory, as that is what it rounds up to.
       if (memReq>mem) continue;

      // Get some measure of how fast it is...
       cl_uint freq;
       status = clGetDeviceInfo(device[d], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(freq), &freq, NULL);

       cl_uint cores;
       status |= clGetDeviceInfo(device[d], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cores), &cores, NULL);

       cl_uint samplers;
       status |= clGetDeviceInfo(device[d], CL_DEVICE_MAX_SAMPLERS, sizeof(samplers), &samplers, NULL);

       cl_device_type type;
       status |= clGetDeviceInfo(device[d], CL_DEVICE_TYPE, sizeof(type), &type, NULL);

       if (status!=CL_SUCCESS) continue;

       float score = cores * samplers * freq;
       if (type==CL_DEVICE_TYPE_GPU) score *= 64.0; // Yup, a fudge factor.

      // If its faster than what we have already found then make it our top choice...
       if (score>bestScore)
       {
        bestScore = score;
        bestDevice = device[d];
        bestType = type;
       }
     }

    // Terminate device array...
     free(device);
   }
   self->device = bestDevice;


 // Clean up the memory that was used to find a good device, fail if a good device was not found...
  free(platform);

  if (bestScore<0.0) return -1;
  if (bestType!=CL_DEVICE_TYPE_GPU)
  {
   printf("Warning: Did not select a GPU for acceleration of video processing.\n");
  }


 // The device is selected - create the assorted objects needed - a context and a work queue...
  self->context = clCreateContext(NULL, 1, &bestDevice, NULL, NULL, &status);
  if (status!=CL_SUCCESS) return -1;

  self->queue = clCreateCommandQueue(self->context, bestDevice, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &status);
  if (status!=CL_SUCCESS) return -1;


 return 0;
}



static PyMethodDef ManagerCL_methods[] =
{
 {NULL}
};



static PyTypeObject ManagerCLType =
{
 PyObject_HEAD_INIT(NULL)
 0,                                 /*ob_size*/
 "manager_cl.ManagerCL",            /*tp_name*/
 sizeof(ManagerCL),                 /*tp_basicsize*/
 0,                                 /*tp_itemsize*/
 (destructor)ManagerCL_dealloc,     /*tp_dealloc*/
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
 "Selects an OpenCL device and initialises it, so it can be shared by multiple video processing nodes. Trys to choose the best one.", /* tp_doc */
 0,                                 /* tp_traverse */
 0,                                 /* tp_clear */
 0,                                 /* tp_richcompare */
 0,                                 /* tp_weaklistoffset */
 0,                                 /* tp_iter */
 0,                                 /* tp_iternext */
 ManagerCL_methods,                 /* tp_methods */
 ManagerCL_members,                 /* tp_members */
 0,                                 /* tp_getset */
 0,                                 /* tp_base */
 0,                                 /* tp_dict */
 0,                                 /* tp_descr_get */
 0,                                 /* tp_descr_set */
 0,                                 /* tp_dictoffset */
 (initproc)ManagerCL_init,          /* tp_init */
 0,                                 /* tp_alloc */
 ManagerCL_new,                     /* tp_new */
};



static PyMethodDef manager_cl_methods[] =
{
 {NULL}
};



#ifndef PyMODINIT_FUNC
#define PyMODINIT_FUNC void
#endif

PyMODINIT_FUNC initmanager_cl(void)
{
 PyObject * mod = Py_InitModule3("manager_cl", manager_cl_methods, "Provides a simple object that select and OpenCL device, so it can be shared between all video processing nodes.");

 if (PyType_Ready(&ManagerCLType) < 0) return;

 Py_INCREF(&ManagerCLType);
 PyModule_AddObject(mod, "ManagerCL", (PyObject*)&ManagerCLType);
}
