#ifndef DATA_MATRIX_H
#define DATA_MATRIX_H

// Copyright 2014 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



// Provides a wrapper around numpy objects that allows you to stitch several together to make a data matrix with both continuous and discrete values in...

#include <Python.h>
#include <structmember.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>



// Types of value...
typedef enum {DISCRETE, CONTINUOUS} NumberType;

// Given the kind of a number array returns which type it makes sense to think of it as...
NumberType KindToType(const PyArray_Descr * descr);

// A function that converts a pointer to an integer...
typedef int (*ToDiscrete)(void * data);

// Helper function that returns the correct function for converting a pointer to a position in a numpy array into a categorical value...
ToDiscrete KindToDiscreteFunc(const PyArray_Descr * descr);

// A function that converts a pointer to a float...
typedef float (*ToContinuous)(void * data);

// Helper function that returns the correct function for converting a pointer to a position in a numpy array into a real value...
ToContinuous KindToContinuousFunc(const PyArray_Descr * descr);



// Defines a block of features to form part of a data matrix - basically a pointer to a numpy array, a type, and functions to convert to continuous or discrete, so each block has fixed type...
typedef struct FeatureBlock FeatureBlock;

struct FeatureBlock
{
 // Dimensions...
  int offset; // Index of first feature.
  int features; // Number of features provided by this struct.

 // Actual data matrix...
  PyArrayObject * array;
  
 // The the type of the data in the array...
  NumberType type;
  
 // Conversion functions from the true array type to both discrete and continuous - both are kept at all times even though only one should be used - allows a little less error checking and allows it to act sanish in a few crazy situations...
  ToDiscrete discrete;
  ToContinuous continuous;
};



// Intialise and deinitalise FeatureBlock-s from an array object; note that offset is always set to 0 and has to be filled in manually...
void FeatureBlock_init(FeatureBlock * this, PyArrayObject * array);
void FeatureBlock_deinit(FeatureBlock * this);

// Returns the feature at given exemplar/feature coordinates, noting that exemplars is done modulus the number in the internal array and feature is offset from the start of this block...
int FeatureBlock_GetDiscrete(FeatureBlock * this, int exemplar, int feature);
float FeatureBlock_GetContinuous(FeatureBlock * this, int exemplar, int feature);



// A DataMatrix object - just a data matrix, except it accepts both continuous and discrete entrys, and can be initialised using a list of arrays to get this feature. Can also use arrays with not enough exemplars in, which will be accessed modulus their length...
typedef struct DataMatrix DataMatrix;

struct DataMatrix
{
 // Dimensions...
  int exemplars;
  int features;
 
 // Maximum values, for making categorical distributions...
  int * max;
  
 // Weights...
  PyArrayObject * weights;
  ToContinuous weights_continuous;
 
 // Feature blocks...
  int blocks;
  FeatureBlock block[0];
};


// New and delete for a DataMatrix - its flexibility means that it has varying malloc sizes, so this gets complicated internally. Constructor accepts a single numpy array or a list of numpy arrays, where the arrays would typically be 2D. 1D arrays can be accepted under the assumption that the feature dimension is of size 1. New will return null with an error set if something is pear shaped. max is optional but if not null then it must be an array of maximum discrete values for each channel, noting that it will be ignored for continuous values - for sizing categorical distributions created from the data. Negative values within maximum will be ignored, and automatically calculated if required...
DataMatrix * DataMatrix_new(PyObject * obj, int * max);
void DataMatrix_delete(DataMatrix * this);


// Accessors for the DataMatrix object - pretty straight forward really...
NumberType DataMatrix_Type(DataMatrix * this, int feature);
int DataMatrix_GetDiscrete(DataMatrix * this, int exemplar, int feature);
float DataMatrix_GetContinuous(DataMatrix * this, int exemplar, int feature);

float DataMatrix_GetWeight(DataMatrix * this, int exemplar);

// Returns the maximum value of a discrete feature, noting that it always includes zero and can be fixed in construction if the user wants space for extra values/to ignore values past a fixed point - its basically how big to make categorical distributions from the data...
int DataMatrix_Max(DataMatrix * this, int feature);



// Setup this module - for internal use only...
void Setup_DataMatrix(void);



#endif
