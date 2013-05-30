#ifndef MEAN_SHIFT_H
#define MEAN_SHIFT_H

// Copyright 2013 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



#include "kernels.h"
#include "spatial.h"
#include "balls.h"



// This calculates the probability of a given feature vector, as defined by the kernel density estimate defined by the provided spatial and kernel (with an associated alpha). You also provide the normalising multiplier, as that can be cached to save repeated calculation, and quality to define the search range around the kernel. The norm parameter must be the kernel normalising constant divided by the weight of the samples...
float prob(Spatial spatial, const Kernel * kernel, float alpha, const float * fv, float norm, float quality);



// Given a kernel (with an alpha parameter), spatial indexing data structure and a feature vector this updates that feature vector to be its mean shift converged point. A temporary vector of the same length as the feature must also be provided. The quality parameter goes from 0 to 1, and maps to the low and high spatial ranges provided by the kernel. There is also an epsilon parameter - it stops when movement drops below it, typically something like 1e-3 is good. The iteration cap ensures that no infinite loops cna occur if epsilon is too low. If the spatial has an ignore entry in the feature vector it uses that as a weight (Will call the get method of spatial.)...
void mode(Spatial spatial, const Kernel * kernel, float alpha, float * fv, float * temp, float quality, float epsilon, int iter_cap);



// Given a Spatial, a Kernel (with its alpha parameter) and an (empty) Balls this assigns modes to every single point in the data matrix contained within the Spatial - after running the Balls object contains the modes, and the output array, aligned with the exemplar index of the data matrix, contains the indices of the modes for each data point (check_step is how many iterations to do between checking if its intersected a hyper-sphere that indicates convergance - exists because that check is much slower than doing a bunch of iterations.) Note that if spatial has an ingored vector then the same vector must be ignored by balls...
void cluster(Spatial spatial, const Kernel * kernel, float alpha, Balls balls, int * out, float quality, float epsilon, int iter_cap, float ident_dist, float merge_range, int check_step);



// Given that a clustering has occured this takes a feature vector and calculates to which cluster it belongs, or returns -1 if its does not belong to any of them...
int assign_cluster(Spatial spatial, const Kernel * kernel, float alpha, Balls balls, float * fv, float * temp, float quality, float epsilon, int iter_cap, int check_step);



// This uses subspace constrained mean shift to project a given feature vector to a manifold. The degrees parameter indicates the degrees of freedom of the surface to converge to - 0 is standard mean shift (Don't do this - its much faster to do the normal thing), 1 will extract lines (1D surfaces), 2 will extract a 2D manifold and so on. Requires the spatial to define the density estimate, plus the feature vector, which is modified in place until it converges - output is in effect a point on the manifold. Also requires four temporaries - grad, for the gradient (Same size as fv); hess, for the hessian (Size of fv squared); eigen_vec, for the eigen vectors of the hessian (Size of fv squared); and eigen_val, for the eigen values (Size of fv). Additionally there are various parameters, for determining accuracy when evaluating the kernels and detecting convergance. The always_hessian parameter should be non-zero for the correct algorithm, but if 0 then it just calculates the hessain once at the start, which saves a lot of time and still works for clean data. Note that this is coded to only work with the unit isotropic Gaussian kernel, hence the kernel is not a parameter...
void manifold(Spatial spatial, int degrees, float * fv, float * grad, float * hess, float * eigen_val, float * eigen_vec, float quality, float epsilon, int iter_cap, int always_hessian);


#endif
