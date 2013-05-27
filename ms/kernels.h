#ifndef KERNELS_H
#define KERNELS_H

// Copyright 2013 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



#include <stddef.h>

// Defines the kernels that mean shift uses - as a set of structures containing function pointers that define the functionality, so they can easily be swapped out...



// Typedef the various function pointers...

// Given the number of dimensions, alpha  and the offsets of a point from the kernel centre (Pointer to an array of length dim, noting that they will have been scaled for a kernel size of 1), this returns the weight of the point in the calculation. Does not need to be normalised. alpha is an arbitrary parameter that the kernel can interprete at will, noting that most kernels ignore it...
typedef float (*KernelWeight)(int dims, float alpha, float * offset);

// Given the number of dimensions and alpha this returns the multiplicative constant to acheive normalisation that is missing from weight. Can be negative if it is not defined / the kernel coder is lazy...
typedef float (*KernelNorm)(int dims, float alpha);

// Given the number of dimensions, alpha, and a quality parameter this returns a maximum offset range after which it can clip the samples and not factor them into the kernel. Quality goes from 0, for low quality, to 1, for high quality...
typedef float (*KernelRange)(int dims, float alpha, float quality);

// Given the dims and alpha plus a feature vector and an offset vector, this applies the offset, returning a delta measure of how much the feature vector has changed. The feature vector is updated inplace. For most kernels this is simple addition, plus a basic vector norm...
typedef float (*KernelOffset)(int dims, float alpha, float * fv, const float * offset);



// Define the struct that defines a kernel type...
typedef struct Kernel Kernel;

struct Kernel
{
 const char * name;
 const char * description;
 
 KernelWeight weight;
 KernelNorm   norm;
 KernelRange  range;
 KernelOffset offset;
};



// Kernels provided by this code...

// Most basic of kernels - constant value if its within a hypersphere, zero if its outside...
const Kernel Uniform;

// Basic linear falloff from the centre of the hyper-sphere...
const Kernel Triangular;

// Basically triangular where the distance is squared - creates a nice bump...
const Kernel Epanechnikov;

// Uses a  consine curve - a slightly smoother version of epanechnikov, though considerably more expensive to compute...
const Kernel Cosine;

// Probably the most popular kernel - the Gaussian, in this case symmetric as its on the distance from the centre. Cutoff is required for this kernel...
const Kernel Gaussian;

// A fat tailed one - the Cauchy distribution on distance from the centre. Cutoff is required for this kernel - can be quite expensive as you need a fairly distant cutoff...
const Kernel Cauchy;

// A kernel based on the von-Mises-Fisher distribution, for dealing with directional data. Requires all samples be on the unit circle. Uses the alpha parameter as the concentration of the kernel being used...
const Kernel Fisher;



// List of all Kernel objects known to the system - for automatic detection...
extern const Kernel * ListKernel[];



#endif
