#ifndef KERNELS_H
#define KERNELS_H

// Copyright 2013 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



#include <stddef.h>

// Defines the kernels that mean shift uses - as a set of structures containing function pointers that define the functionality, so they can easily be swapped out...



// Typedef of a pointer to a kernel configuration, which kernels can use as they choose...
typedef void * KernelConfig;



// Typedef the various function pointers...

// Constructs a configuration for the kernel in question, given a string describing it - the returned configuration will be a new reference. Note that the string might continue past the specification for this kernel - in which case it should automatically stop and ignore the rest...
typedef KernelConfig (*KernelConfigNew)(const char * config);

// Verifies that the given configuration string is good - returns null if its ok or an error message if its not; optional kicks out how many chars were consumed into length...
typedef const char * (*KernelConfigVerify)(const char * config, int * length);

// Incriments the reference count for the given KernelConfig...
typedef void (*KernelConfigAcquire)(KernelConfig config);

// Decriments the reference count for the given KernelConfig, terminating it if it hits zero...
typedef void (*KernelConfigRelease)(KernelConfig config);


// Given a configuration and the offsets of a point from the kernel centre (Pointer to an array of length dim, noting that they will have been scaled for a kernel size of 1), this returns the weight of the point in the calculation. Does not need to be normalised. alpha is an arbitrary parameter that the kernel can interprete at will, noting that most kernels ignore it...
typedef float (*KernelWeight)(int dims, KernelConfig config, float * offset);

// Given the number of dimensions and alpha this returns the multiplicative constant to acheive normalisation that is missing from weight. Can be negative if it is not defined / the kernel coder is lazy...
typedef float (*KernelNorm)(int dims, KernelConfig config);

// Given the configuration and a quality parameter this returns a maximum offset range after which it can clip the samples and not factor them into the kernel. Quality goes from 0, for low quality, to 1, for high quality...
typedef float (*KernelRange)(int dims, KernelConfig config, float quality);

// Given the configuration plus a feature vector and an offset vector, this applies the offset, returning a delta measure of how much the feature vector has changed. The feature vector is updated inplace. For most kernels this is simple addition, plus a basic vector norm to measure the change...
typedef float (*KernelOffset)(int dims, KernelConfig config, float * fv, const float * offset);

// Allows you to draw from the kernel (At position centre in the scaled space) - you provide the first 3 indices for the philox rng (It then has the entire range of the 4th value for its own use, whilst remaining entirly predictable.), and also provide an output vector, as well as the configuration as per usual...
typedef void (*KernelDraw)(int dims, KernelConfig config, const unsigned int index[3], const float * center, float * out);



// Define the struct that defines a kernel type...
typedef struct Kernel Kernel;

struct Kernel
{
 const char * name;
 const char * description;
 
 KernelConfigNew     config_new;
 KernelConfigVerify  config_verify;
 KernelConfigAcquire config_acquire;
 KernelConfigRelease config_release;
 
 KernelWeight weight;
 KernelNorm   norm;
 KernelRange  range;
 KernelOffset offset;
 KernelDraw   draw;
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
