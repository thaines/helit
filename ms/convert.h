#ifndef CONVERT_H
#define CONVERT_H

// Copyright 2014 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



// Defines a system for converting the input into another format for processing - this is typically used when the data is encoded in such a way that none of the provided kernels make sense unless its converted into a different represntation first. Mainly used to convert angular representations into vectors for compatibility with euclidean and directional statistics kernels. Also has a memory advantage if the provided encoding takes up less space than the intermediate.



// Typedefs of assortd function pointers that are used for the convertor interface...

// Converts from the provided representaton to the representation on which the kernels work. Pointers to float arrays must be long enough...
typedef void (*ConvertToInt)(const float * external, float * internal);

// Converts from the kernel-suitable representation back to the original. Pointers to float arrays must be long enough...
typedef void (*ConvertToExt)(const float * internal, float * external);



// Define the struct that defines a convertor type...
typedef struct Convert Convert;

struct Convert
{
 const char code; // How the user chooses it.
 const char * name;
 const char * description;
 
 const int dim_ext;
 const int dim_int;
 
 ConvertToInt to_int;
 ConvertToExt to_ext;
};



// Convertors provided by this code...

// NoOp - just copies across a single value...
extern const Convert NoOp;

// Deletes a single feature - bit weird, but can be useful when you are constructing multiple estiamtes driven by different subsets of an input array...
extern const Convert Delete;

// Converts an angle into a 2D unit length vector that represents the angle...
extern const Convert Angle;

// Converts radial coordinates (angle and distance) to a 2D coordinate...
extern const Convert Radial;

// Converts spherical coordinates without a radius to a 3D unit length vector...
extern const Convert SphericalAngle;

// Converts spherical coordinates with a radius to a 3D coordinate...
extern const Convert SphericalCoord;

// Converts an angle axis rotation into a 4D unit quatrnion, for which it makes sense to apply a mirror Fisher distribution to...
extern const Convert AngleAxis;

// Converts a Euler rotation with the order of rotations as x-y-z into a 4D unit quaternion...
extern const Convert Euler;



// List of all Kconvert objects provided by the system...
extern const Convert * ListConvert[];



#endif
