#ifndef PHILOX_H
#define PHILOX_H

// Copyright 2013 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



// Random number generator, designed to go directly from a sequence position to a random number - out is the counter on entry, the output when done...
void philox(unsigned int out[4]);



// Converts the output of philox (any of the 4 output unsigned ints) into a uniform draw in [0, 1)...
float uniform(unsigned int ui);



// Returns a draw from a standard normal distribution given two outputs from philox. You can optionally provide a pointer into which a second (independent) output is written...
float box_muller(unsigned int pa, unsigned int pb, float * second);



// Helper struct - given a pointer to 4 unisgned ints to index into the philox rng this generates random numbers, efficiently...
typedef struct PhiloxRNG PhiloxRNG;

struct PhiloxRNG
{
 unsigned int * index;
 unsigned int data[4];
 unsigned int next;
};



// Initialises the RNG with a pointer to an index, that decides what random number to get next - the index will be incrimented as its used up, so it can be used to initalise a future PhiloxRNG instance...
void PhiloxRNG_init(PhiloxRNG * this, unsigned int * index);

// Returns the next random number...
unsigned int PhiloxRNG_next(PhiloxRNG * this);

// Returns a uniform draw from the rng, in [0, 1)...
float PhiloxRNG_uniform(PhiloxRNG * this);

// Returns a standard Normal draw from the rng; can optionally output a second, which is independent of the first...
float PhiloxRNG_Gaussian(PhiloxRNG * this, float * second);

// Returns a draw from a Gamma distribution, where beta (the scale) is fixed to one...
float PhiloxRNG_UnscaledGamma(PhiloxRNG * this, float alpha);

// Returns a draw from a Gamma distribution...
float PhiloxRNG_Gamma(PhiloxRNG * this, float alpha, float beta);

// Returns a draw from a Beta distribution...
float PhiloxRNG_Beta(PhiloxRNG * this, float alpha, float beta);



#endif
