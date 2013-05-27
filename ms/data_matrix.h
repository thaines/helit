#ifndef DATA_MATRIX_H
#define DATA_MATRIX_H

// Copyright 2013 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



#include <Python.h>
#include <structmember.h>
#include <numpy/arrayobject.h>

// Provides a wrapper around a numpy matrix that converts it into a data matrix of floats - allows each index of the dat matrix to be assigned to one of three classes - data, feature or spatial. It does linearisation to allow indexing of the exemplars/features, always in row major order...



// Define the types for an index of a matrix...
typedef enum {DIM_DATA, DIM_DUAL, DIM_FEATURE} DimType;

// A function that converts a pointer to a float...
typedef float (*ToFloat)(void * data);

// Helper function, that is also used elsewhere in the system...
ToFloat KindToFunc(const PyArray_Descr * descr);



// Define the structure that wraps a matrix...
typedef struct DataMatrix DataMatrix;

struct DataMatrix
{
 // Actual data matrix...
  PyArrayObject * array;
  
 // Assignment of type to each dimension of them dm...
  DimType * dt;
  
 // The index of the feature that provides the weight, negative if none; also the multiplier of the weight to output...
  int weight_index;
  float weight_scale;
  
 // Number of data items represented by the data matrix - calculated when you set the array...
  int exemplars;
  
 // Number of features for each exemplar - calculated when you set the array as its quite complex (Takes into account feature dimensions and the weight feature)...
  int feats;
  
 // Number of dual features - avoids having to calculate it at every request...
  int dual_feats;
  
 // Entries in the feature vector are multiplied by these values before extraction...
  float * mult;
  
 // Temporary storage - when you request a feature vector this is what is returned in...
  float * fv;
  
 // Optimisation structure for the feature extraction - how many feature dimensions exist, and their indices...
  int feat_dims;
  int * feat_indices;
  
 // Function pointer to accelerate conversion...
  ToFloat to_float;
};



// For initialising the pointers within the data matrix object, and deinitialising them when done...
void DataMatrix_init(DataMatrix * dm);
void DataMatrix_deinit(DataMatrix * dm);

// Allows you to set a numpy data matrix for use; you have to assign a type to each dimension as well. If you want a weight that is not 1 you can assign one of the features to be a weight, by giving its index - this feature will not appear in the feature count and return of the fv method. Otherwise set the weight index to be negative...
void DataMatrix_set(DataMatrix * dm, PyArrayObject * array, DimType * dt, int weight_index);

// Returns basic stats...
int DataMatrix_exemplars(DataMatrix * dm);
int DataMatrix_features(DataMatrix * dm);

// Allows you to set the multipliers for the features - the scale array passed in better have the right length, as returned by the feats method. You should also provide a weight_scale...
void DataMatrix_set_scale(DataMatrix * dm, float * scale, float weight_scale);

// Fetches a feature vector, using a single index to do row-major indexing into all dimensions marked as data or dual. Note that the returned pointer is to internal storage, that is replaced every time this method is called. The dual dimensions will always be first, followed by all the feature dimensions in row major flattened order. If you want the weight as well provide a pointer and it will be filled...
float * DataMatrix_fv(DataMatrix * dm, int index, float * weight);



#endif
