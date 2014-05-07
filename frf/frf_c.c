// Copyright 2014 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



#include <Python.h>
#include <structmember.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>


#include "learner.h"

#include "frf_c.h"



static PyMethodDef frf_c_methods[] =
{
 {NULL}
};



#ifndef PyMODINIT_FUNC
#define PyMODINIT_FUNC void
#endif

PyMODINIT_FUNC initfrf_c(void)
{
 PyObject * mod = Py_InitModule3("frf_c", frf_c_methods, "Provides a straight forward random forest implimentation that is designed to be fast and have good loading/saving capabilities, unlike all the other Pythion ones.");
 
 import_array();
 SetupCodeToTest();
 
 //if (PyType_Ready(&FRFType) < 0) return;
 
 //Py_INCREF(&FRFType);
 //PyModule_AddObject(mod, "FRF", (PyObject*)&FRFType);
}
