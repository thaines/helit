#ifndef BSPLINE
#define BSPLINE

// Copyright 2016 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

#include <Python.h>
#include <structmember.h>

#ifndef __APPLE__
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif
#include <numpy/arrayobject.h>

// B-spline kernel - kept seperate so I can include it into multiple modules...



// B-spline kernel - recursive definition that will do any degree, but has hard coded evaluations for lower degrees (0 to 3 inclusive, where 0=nearest neighbour, 1=linear, etc.). Symmetric around the origin, and can be thought of as having integer spaced uniform knots...
float B(int degree, float x);



// Does a standard B-spline sampling of a 2D float array at the given float position, with the given degree. Degree must not go higher than 5 (no mask support, hence must have been filled in if it has one to get sensible behaviour)...
float SampleB(int degree, float y, float x, PyArrayObject * data);



// Same as SampleB, but takes a 3D array instead which it indexes [layer, y, x]; for the situation where you have a stack of images to select from...
float LayerSampleB(int degree, int layer, float y, float x, PyArrayObject * data);



// Does a standard B-spline sampling of a multivariate image at the given float position, with the given degree. Degree must not go higher than 5 (no mask support, so image must have been filled in if it has one to get sensible behaviour)...
void MultivariateSampleB(int degree, float y, float x, int shape[2], int channels, PyArrayObject ** image, float * out);



// Calls the numpy _import_array() function...
void PrepareBSpline(void);


#endif
