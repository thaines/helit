#ifndef SPATIAL_H
#define SPATIAL_H

// Copyright 2013 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



#include "data_matrix.h"

// Defines an indexing structures for quickly finding all entities within a bounding box - fully modular so new ones can be added later...



// Typedef for a spatial indexing object...
typedef void * Spatial;



// Typedef the various methods that define a spatial indexing object...

// New and delete...
typedef Spatial (*SpatialNew)(DataMatrix * dm);
typedef void (*SpatialDelete)(Spatial this);

// Returns the data matrix it is a spatial structure for (Note that it does not own this data matrix - user must delete it when done.)...
typedef DataMatrix * (*SpatialDM)(Spatial this);

// Starts indexing around a given point - you provide an array of floats to define the point and a range for how far to go, definining a hyper-cube. Note that the centre point should remain valid until you stop calling next...
typedef void (*SpatialStart)(Spatial this, const float * centre, float range);

// You call this until it returns a negative number - each return is a value to process. Will include all values in the dounding box, and possibly some outside it as well...
typedef int (*SpatialNext)(Spatial this);



// Define the spatial type...
typedef struct SpatialType SpatialType;

struct SpatialType
{
 const char * name;
 const char * description;
 
 SpatialNew init;
 SpatialDelete deinit;
 
 SpatialDM dm;
 
 SpatialStart start;
 SpatialNext next;
};



// Define access functions for the spatial objects - they all assume the first entry in the structure pointed to by Spatial is a pointer to its type structure. These just match up with the function pointer typedefs...
Spatial Spatial_new(const SpatialType * type, DataMatrix * dm);
void Spatial_delete(Spatial this);

const SpatialType * Spatial_type(Spatial this);
DataMatrix * Spatial_dm(Spatial this);

void Spatial_start(Spatial this, const float * centre, float range);
int Spatial_next(Spatial this);



// The various spatial index implimentations provided by this module...

// Stupid spatial index - does nothing and just returns everything every time...
const SpatialType BruteForceType;

// This makes use of dual dimensions in the data matrix, and just brute forces within the range; often however the dual dimensions can cut down the amount of data that needs processing rather drammatically. Note that if there are no dual dimensions it ends up equivalent to brute forcing...
const SpatialType IterDualType;

// A classic - the binary kd tree...
const SpatialType KDTreeType;



// List of all spatial indexing types known to the system - for automatic detection...
extern const SpatialType * ListSpatial[];



#endif
