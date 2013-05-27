#ifndef EIGEN_H
#define EIGEN_H

// Copyright 2013 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



// Given a symmetric matrix this calculates its eigen decomposition - being symmetric it avoids the need for complex numbers, so real input and output. dims is the number of dimensions - the matrices are dims x dims, whilst the vector is of length dims; indexing is row major. a is the input matrix - it will be trashed. q is the output rotation matrix, that contains the eigenvectors in the columns. d is the output diagonal matrix, represented as a vector - the eigenvalues, aligned with q. The relationship q^T a q = d will hold. Returns 0 on success...
int symmetric_eigen_raw(int dims, float * a, float * q, float * d);

// Calculates the eigen-decomposition, sorting the eigenvalues from largest to smallest...
int symmetric_eigen(int dims, float * a, float * q, float * d);

// Calculates the eigen-decomposition, sorting the eigenvalues from largest to smallest using their absolute value...
int symmetric_eigen_abs(int dims, float * a, float * q, float * d);



#endif
