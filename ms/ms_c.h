#ifndef MS_C_H
#define MS_C_H

// Copyright 2013 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



#include <Python.h>
#include <structmember.h>
#include <numpy/arrayobject.h>

#include "mean_shift.h"



// Pre-decleration...
typedef struct MeanShift MeanShift;



// The actual structure...
struct MeanShift
{
 PyObject_HEAD
  
 // The kernel to use, with an alpha parameter...
  const Kernel * kernel;
  float alpha;
  
 // The spatial indexing structure to use...
  const SpatialType * spatial_type;
  
 // The hyper-sphere indexing used for clustering...
  const BallsType * balls_type;
 
 // The data matrix, plus the spatial indexing structure if it has been created...
  DataMatrix dm;
  float weight; // Total weight in the system if calculated, negative if not calculated.
  float norm; // The normalising constant - the kernel normaliser divided by the weight, or negative if not calculated. Also include the affect of any feature scaling.
  Spatial spatial;
  
 // The current ball object - after clustering this can be used to find out which cluster novel features belong to...
  Balls balls;
  
 // Parameters to control the quality/runtime of the output...
  float quality;
  float epsilon;
  int iter_cap;
  
  float ident_dist;
  float merge_range;
  int merge_check_step;
};



#endif
