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


#include "tree.h"



// Define struct used for Forest I/O...
typedef struct ForestHeader ForestHeader;

struct ForestHeader
{
 char magic[4]; // 'FRFF'
 int revision; // 1 in current versions.
 long long size; // Total size of just the header, ignoring trees.
 
 int trees; // Number of trees that follow this header. Can be ignored if appropriate.
 
 char bootstrap;
 int opt_features;
 int min_exemplars;
 int max_splits;
 unsigned int key[4];
 
 int x_feat;
 int y_feat;
 int ratios; // Number of ratio rows.
 
 char codes[0]; // Codes for summary, then info then learn; y_feat * 2 + x_feat.
 // Ratios follow, as float; ratios * y_feat * sizeof(float).
 // x_max follows, as array of int. Always included, even if all -1; x_feat * sizeof(int).
 // Then y_max - as above; y_feat * sizeof(int).
};



// Declare the python version of the tree object...
typedef struct TreeBuffer TreeBuffer;

struct TreeBuffer
{
 PyObject_HEAD
 
 size_t size;
 Tree * tree;
 
 char ready; // 1 if its ready to be used (init has been called), 0 if not.
};



// Declare the forest object...
typedef struct Forest Forest;

struct Forest
{
 PyObject_HEAD
 
 // The fixed configuration data, that can't be changed without getting new trees...
  int x_feat; // Number of features for input.
  int y_feat; // Number of features for output.
  
  int * x_max; // Maximum, values, if provided.
  int * y_max; // "
  
  char * summary_codes; // length==y_feat - codes for creating summary objects.
  char * info_codes; // length==y_feat - codes for creating info objects.
  char * learn_codes; // length==x_feat - codes for creating learning objects.
  
  char ready; // Non-zero if the above are all configured correctly.
 
 // The variable configuration data, which can be adjusted on a per-tree basis...
  char bootstrap; // Zero to train trees on everything, non-zero to do a bootstrap draw.
  int opt_features;
  int min_exemplars;
  int max_splits;
  unsigned int key[4];
  
  PyArrayObject * info_ratios; // 2D array indexed by depth (modulus) then feature, of weight to assign to information of feature when optimising at that depth.
  
 // Store the trees as a straight array of pointers (The cost of loading a tree, let alone learning one, compared to a realloc means doing anything more dcomplicated is pointless.)...
  int trees;
  TreeBuffer ** tree;
  
 // Cached stuff...
  int ss_size;
  SummarySet ** ss;
};



#endif
