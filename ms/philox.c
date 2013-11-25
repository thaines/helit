// Copyright 2013 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

#include "philox.h"

#include <stdlib.h>
#include <stdint.h>
#include <math.h>



static unsigned int mul_hi(unsigned int a, unsigned int b)
{
 uint64_t _a = a;
 uint64_t _b = b;
 
 return (_a * _b) >> 32;
}



void philox(unsigned int out[4])
{
 const unsigned int key[2] = {0x4edbf6fa, 0x6aa1107f}; // Normally a parameter, but I only need one hash per input.
 const unsigned int mult[2] = {0xCD9E8D57, 0xD2511F53};
 int rnd, i;
 
 // Iterate and do each round in turn, updating the counter before we finally return it (Indexing from 1 is conveniant for the Weyl sequence.)...
 for (rnd=1;rnd<=10;rnd++)
 {
  // Calculate key for this step, by applying the Weyl sequence on it...
   unsigned int keyWeyl[2];
   keyWeyl[0] = key[0] * rnd;
   keyWeyl[1] = key[1] * rnd;

  // Apply the s-blocks, also swap the r values between the s-blocks...
   unsigned int next[4];
   next[0] = out[1] * mult[0];
   next[2] = out[3] * mult[1];
   
   next[3] = mul_hi(out[1],mult[0]) ^ keyWeyl[0] ^ out[0];
   next[1] = mul_hi(out[3],mult[1]) ^ keyWeyl[1] ^ out[2];
   
  // Prepare for the next step...
   for (i=0;i<4;i++) out[i] = next[i];
 }
}
