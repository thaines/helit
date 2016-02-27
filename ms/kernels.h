#ifndef KERNELS_H
#define KERNELS_H

// Copyright 2013 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



#include <stddef.h>

#include "philox.h"
#include "mult.h"

// Defines the kernels that mean shift uses - as a set of structures containing function pointers that define the functionality, so they can easily be swapped out...



// Typedef of a pointer to a kernel configuration, which kernels can use as they choose...
typedef void * KernelConfig;



// Typedef the various function pointers...

// Constructs a configuration for the kernel in question, given a string describing it - the returned configuration will be a new reference. Note that the string might continue past the specification for this kernel - in which case it should automatically stop and ignore the rest. It can assume that the string has passed the verify function (below)...
typedef KernelConfig (*KernelConfigNew)(int dims, const char * config);

// Verifies that the given configuration string is good - returns null if its ok or an error message if its not; optionaly kicks out how many chars were consumed into length (Only valid if NULL is returned)...
typedef const char * (*KernelConfigVerify)(int dims, const char * config, int * length);

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

// Converts the given feature vector into an offset from the base feature vector, modifying it inplace - it is this converted offset that has the weight function applied and is then averaged to get the mean shift. For most kernels this is simply subtracting the base feature vector from the feature vector in place - exists to support really weird kernels...
typedef void (*KernelToOffset)(int dims, KernelConfig config, float * fv, const float * base_fv);

// Given the configuration plus a feature vector and an offset vector, this applies the offset, returning a delta measure of how much the feature vector has changed. The feature vector is updated inplace. For most kernels this is simple addition, plus a basic vector norm to measure the change...
typedef float (*KernelOffset)(int dims, KernelConfig config, float * fv, const float * offset);

// Allows you to draw from the kernel (At position centre in the scaled space) - you provide the first 3 indices for the philox rng (It then has the entire range of the 4th value for its own use, whilst remaining entirly predictable.), and also provide an output vector, as well as the configuration as per usual...
typedef void (*KernelDraw)(int dims, KernelConfig config, PhiloxRNG * rng, const float * center, float * out);

// Given a set of feature vectors, each the centre of an instance of this kernel type, with different scales, this returns how much probability mass is in the multiplication of them - 1 if they all perfectly overlap, 0 if there is no overlap. It is the weight assigned to the multiplication of components if constructing a mixture model via multiplication. In addition to the usual takes as input pointers to arrays of feature vectors and scales, plus a MultCache, which controls how it behaves. Note that the configuration parameter is an array indexed by term, and can be NULL if its a kernel type that is not configured...
typedef float (*KernelMultMass)(int dims, KernelConfig * config, int terms, const float ** fv, const float ** scale, MultCache * cache);

// Given a set of features vectors, each the centre of an instance of this kernel type, with different scales, this outputs a draw from the distribution created by multiplying them together. The parameter fake indicates how fake it can be - 0 means it must be a proper draw, 1 means a mode of the resulting distribution is an acceptable alternative, 2 means the 'average' of the locations is allowed (If the distribution is not on Euclidean space this may still be complex.). Only mode 0 has to be truly supported - the others are optional optimisations. Rest of the parameters are identical to KernelMultMass, except for the new 'out' parameter - this must be of length dims so the draw can be dumped into it; note that the draw is output in unscaled space Note that the configuration parameter is an array indexed by term, and can be NULL if its a kernel type that is not configured...
typedef void (*KernelMultDraw)(int dims, KernelConfig * config, int terms, const float ** fv, const float ** scale, float * out, MultCache * cache, int fake);

// Returns the number of states a feature vector using this kernel can be in - this is used for convergance detection when sending vectors to the balls system for mean shift, with all states being tried...
typedef int (*KernelStates)(int dims, KernelConfig config);

// Transitions the given feature vector from the current state to the next one - note that this must be cyclic, such that if called KernelStates times you end up back where you started. State is how many times this method has been called on the given feature vector - it could loop many times!..
typedef void (*KernelNext)(int dims, KernelConfig config, int state, float * fv);

// Returns the size in bytes of the kernel, in most cases this is zero. You can optionally provide ref_count, which will be filled in with how many MeanShift objects use the given configuration, so you can amortize their memory between them...
typedef size_t (*KernelByteSize)(int dims, KernelConfig config, int * ref_count);



// Define the struct that defines a kernel type...
typedef struct Kernel Kernel;

struct Kernel
{
 const char * name;
 const char * description;
 const char * configuration; // NULL if it requires no configuration, otherwise a human readable string describing what is required. If NULL the kernel is assumed to be initialised with its name alone - this fact is used by the code. Yeah, this is staggering drunk into Daily WTF terratory...
 
 KernelConfigNew     config_new;
 KernelConfigVerify  config_verify;
 KernelConfigAcquire config_acquire;
 KernelConfigRelease config_release;
 
 KernelWeight   weight;
 KernelNorm     norm;
 KernelRange    range;
 KernelToOffset to_offset;
 KernelOffset   offset;
 KernelDraw     draw;
 
 KernelMultMass mult_mass;
 KernelMultDraw mult_draw;
 
 KernelStates states;
 KernelNext next;
 
 KernelByteSize byte_size;
};



// Random constant used by Fisher and MirrorFisher kernels...
#define CONC_SWITCH 256.0 // When the concentration excedes this value it switches to a Gaussian approximation.



// Kernels provided by this code...

// Discrete distribution in the sense it treats every value as rounded towards the nearest integer - kinda pointless in the sense there are much faster ways of doing this, but valuable because it can be combined with continuous distributions via the Composite Kernel, so you can do distributions over mixed discrete/continuous entities...
extern const Kernel Discrete;

// Most basic of (continuous) kernels - constant value if its within a hypersphere, zero if its outside...
extern const Kernel Uniform;

// Basic linear falloff from the centre of the hyper-sphere...
extern const Kernel Triangular;

// Basically triangular where the distance is squared - creates a nice bump...
extern const Kernel Epanechnikov;

// Uses a  consine curve - a slightly smoother version of epanechnikov, though considerably more expensive to compute...
extern const Kernel Cosine;

// Probably the most popular kernel - the Gaussian, in this case symmetric as its on the distance from the centre. Cutoff is required for this kernel...
extern const Kernel Gaussian;

// A fat tailed one - the Cauchy distribution on distance from the centre. Cutoff is required for this kernel - can be quite expensive as you need a fairly distant cutoff...
extern const Kernel Cauchy;

// The other fat tailed one - the Logistic distribution on distance from the centre. Has the same distant cutoff therefore expensive issue as the Cauchy...
extern const Kernel Logistic;


// A kernel based on the von-Mises-Fisher distribution, for dealing with directional data. Requires all samples be on the unit circle. Uses the alpha parameter as the concentration of the kernel being used...
extern const Kernel Fisher;

// A Fisher (partial-)wrapper - this is for when a direction vector and its negated direction vector are equivalent, i.e. when rotations of 180 degrees don't matter. Whilst traditionally this is the realm of the Bingham distribution this simply takes two Fisher distributions, one with the direction of the input, the other its negative, and sums them together with equal probability. In the 2D case this is equivalent to the Bingham distribution anyway...
extern const Kernel MirrorFisher;


// A kernel that allows you to combine kernels, so you can have different kernels on different features within a feature vector.
extern const Kernel Composite;



// List of all Kernel objects known to the system - for automatic detection...
extern const Kernel * ListKernel[];



// Would normally be kept private, but the multiplication system requires access to this...
typedef struct FisherConfig FisherConfig;

struct FisherConfig
{
 int ref_count; // Reference count.
 float alpha; // Concentration parameter.
 float log_norm; // log of the normalising constant.
 
 int inv_culm_size; // Length of below.
 float * inv_culm; // Array containing the inverse culmative of the distribution over the dot product result.
 
 int * order; // Array of length dims that contains the integers 0..dims-1; used when drawing.
};



#endif
