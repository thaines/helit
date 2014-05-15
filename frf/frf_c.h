#ifndef FRF_H
#define FRF_H

// Copyright 2014 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.


#include <Python.h>
#include <structmember.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>



// Declare the forest object...
typedef struct Forest Forest;

struct Forest
{
 PyObject_HEAD
 
 // The fixed configuration data, that can't be changed without getting new trees...
  int x_feat; // Number of features for input.
  int y_feat; // Number of features for output.
  
  char * summary_codes; // length==y_feat - codes for creating summary objects.
  char * info_codes; // length==y_feat - codes for creating info objects.
  char * learn_codes; // length==x_feat - codes for creating learning objects.
 
 // The variable configuration data, which can be adjusted on a per-tree basis...
  char bootstrap; // Zero to train trees on everything, non-zero to do a bootstrap draw.
  int opt_features;
  int min_exemplars;
  int max_splits;
  unsigned int key[4];
  
  PyArrayObject * info_ratios; // 2D array indexed by depth (modulus) then feature, of weight to assign to information of feature when optimising at that depth.
  
 // Store the trees as a linked list...
  int trees;
  
};



#endif
