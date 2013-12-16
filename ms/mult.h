#ifndef MULT_C_H
#define MULT_C_H

// Copyright 2013 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



#include "kernels.h"



// Uses monte carlo integration to approximate the area under the multiplication of several exemplars with associated kernel - effectivly the weight to use if you multiply these mixtures components together and put them into a mixture model with other ones...
// Inputs are: kernel - the kernel applied to all feature vectors plus its configuration; dims - the number of dimensions; terms is the number of terms in the multiplication; fv is an array of length terms pointing to an array of length dims - they are the feature centers fot the kernel to multiply together; scale is the scale parameters associated with each fv, for if they do not match (It has been applied to those passed in - for conversion between spaces); temp1 and temp2 are both pointers to some temporary storage, of length dims; samples is how many importances samples to take, and index is the seed of the philox random number generator.
// Return value is the area under the function created by multiplying the pdf's together.
float mult_area_mci(const Kernel * kernel, KernelConfig config, int dims, int terms, const float ** fv, const float ** scale, float * temp1, float * temp2, int samples, const unsigned int index[2]);



// Draws from the distribution implied by the multiplication of a bunch of distributions (kernels) - uses Metropolis-Hastings.
// ************************
float mult_draw_mh(const Kernel * kernel, KernelConfig config, int dims, int terms, const float ** fv, const float ** scale, float * out, float * temp, int steps, const unsigned int index[2]);



#endif
