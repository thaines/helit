#ifndef PHILOX_H
#define PHILOX_H

// Copyright 2013 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



// Random number generator, designed to go directly from a sequence position to a random number - out is the counter on entry, the output when done...
void philox(unsigned int out[4]);



// Converts the output of philox (any of the 4 output unisgned ints) into a uniform draw in [0, 1)...
float uniform(unsigned int ui);



// Returns a draw from a standard normal distribution given two outputs from philox. You can optionally provide a pointer into which a second (independent) output is written...
float box_muller(unsigned int pa, unsigned int pb, float * second);



#endif
