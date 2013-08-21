#ifndef BALLS_H
#define BALLS_H

// Copyright 2013 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



// Defines code to handle a set of hyper-spheres - essentially provides online algorithms to create them, and query if a given point is within one, where each ball is represented by a center and radius...



// Typedef for a hyper-sphere intersection object instance...
typedef void * Balls;



// Typedef the various methods that define the hyper-sphere index...

// New and delete - dims is the dimensionality of the hyper-spheres, radius is the typical radius of each sphere, so it can adapt its structure accordingly...
typedef Balls (*BallsNew)(int dims, float radius);
typedef void (*BallsDelete)(Balls this);

// Returns the dimensionality of the position vectors that it expects/outputs...
typedef int (*BallsDims)(Balls this);

// Returns how many balls are known to the system - note that their identifying integers must be tightly packed, so this allows them to be iterated...
typedef int (*BallsCount)(Balls this);

// Allows you to create a new ball, returning the number it is assigned...
typedef int (*BallsCreate)(Balls this, const float * pos, float radius);

// Returns the position of the given ball - you must not mess with the returned poiinter...
typedef const float * (*BallsPos)(Balls this, int index);

// Returns the radius of the given ball...
typedef float (*BallsRadius)(Balls this, int index);

// If the given point is within a ball this returns the index of that hyper-sphere, if its not it returns a negative number. If its within multiple it returns whichever one it feels like...
typedef int (*BallsWithin)(Balls this, const float * pos);



// Define the type for a ball intersection object...
typedef struct BallsType BallsType;

struct BallsType
{
 const char * name;
 const char * description;
 
 BallsNew init;
 BallsDelete deinit;
 
 BallsDims dims;
 BallsCount count;
 
 BallsCreate create;
 BallsPos pos;
 BallsRadius radius;
 
 BallsWithin within;
};



// Define basic access functions for Balls objects - relies on the first entry in the structure pointed to by a Balls being a pointer to its type...
Balls Balls_new(const BallsType * type, int dims, float radius);
void Balls_delete(Balls this);
int Balls_dims(Balls this);
int Balls_count(Balls this);
int Balls_create(Balls this, const float * pos, float radius);
const float * Balls_pos(Balls this, int index);
float Balls_radius(Balls this, int index);
int Balls_within(Balls this, const float * pos);



// The various implimentations provided by this module...

// Simple list - does a brute force search each time...
const BallsType BallsListType;

// Spatial hashing version - divides the space into grid cells, storing each hyper-sphere into all grid cells with which it collides. Makes within tests very fast...
const BallsType BallsHashType;



// List of all balls types known to the system - for automatic detection...
extern const BallsType * ListBalls[];



#endif
