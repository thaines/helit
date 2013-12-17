#ifndef MULT_C_H
#define MULT_C_H

// Copyright 2013 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



// Definitions of a kernel, repeated to handle loops in the dependencies...
typedef void * KernelConfig;
typedef struct Kernel Kernel;



// A cache of all the buffers required by the multiplication system, with intialisation/cleanup/resize functions, plus it contains sampling parameters and a philox rng index...
typedef struct MultCache MultCache;

struct MultCache
{
 int max_dims;
 int max_terms;
 
 float * temp_dims1;
 float * temp_dims2;
 float * temp_terms1;
 float * temp_terms2;
 
 unsigned int rng_index[2];
 
 int gibbs_samples;
 int mci_samples;
 int mh_proposals;
};


// Constructor & destructor...
void MultCache_new(MultCache * self); // Sets samples/proposals to -1!
void MultCache_delete(MultCache * self);

// Makes sure it has enough cache for the given size, re-allocating it if need be...
void MultCache_ensure(MultCache * self, int dims, int terms);



// Uses monte carlo integration to approximate the area under the multiplication of several exemplars with associated kernel - effectivly the weight to use if you multiply these mixtures components together and put them into a mixture model with other ones...
// Inputs are: kernel - the kernel applied to all feature vectors plus its configuration; dims - the number of dimensions; terms is the number of terms in the multiplication; fv is an array of length terms pointing to an array of length dims - they are the feature centers fot the kernel to multiply together; scale is the scale parameters associated with each fv, for if they do not match (It has been applied to those passed in - for conversion between spaces); finally the cache is an initialised MultCache.
// Return value is the area under the function created by multiplying the pdf's together.
float mult_area_mci(const Kernel * kernel, KernelConfig config, int dims, int terms, const float ** fv, const float ** scale, MultCache * cache);



// Draws from the distribution implied by the multiplication of a bunch of distributions (kernels) - uses Metropolis-Hastings, where the proposal distribution is taken as one of the multiplicands, with no dependence on the current state (This is mathematically elegant, as stuff cancels out and there is no worry about tuning a proposal distribution to get a reasonable accept rate (which is not to say you will get a good accept rate, but under typical usage you probably will).)...
// First parameters are the kernel, the kernel configuration, the number of dimensions (features) and how many terms are in the multiplicand. The feature vectors and the scale that has been applied to them are the next two parameters, followed by the output array (length # of features), which will output without a scale applied (unlike the inputs, which do have scale applied!). Following that is an initialised MultCache. It returns the number of accepts that occured, for curiosities sake.
int mult_draw_mh(const Kernel * kernel, KernelConfig config, int dims, int terms, const float ** fv, const float ** scale, float * out, MultCache * cache);



#endif
