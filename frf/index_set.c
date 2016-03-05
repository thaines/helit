// Copyright 2014 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



#include "philox.h"
#include "learner.h"

#include "index_set.h"



// Methods for the IndexSet...
IndexSet * IndexSet_new(int size)
{
 IndexSet * this = (IndexSet*)malloc(sizeof(IndexSet) + size * sizeof(int));
 this->size = size;
 return this;  
}

void IndexSet_delete(IndexSet * this)
{
 free(this);  
}

IndexSet * IndexSet_new_reflect(IndexSet * other)
{
 int i;
 
 // Create a temporary array of tags - zero if not seen, non-zero if seen...
  char * tags = (char*)malloc(other->size * sizeof(char));
  for (i=0; i<other->size; i++) tags[i] = 0;

 // Go through and mark seen everything in the other...
  for (i=0; i<other->size; i++)
  {
   tags[other->vals[i]] = 1; 
  }

 // Count how many unseen indices exist...
  int unseen = 0;
  for (i=0; i<other->size; i++)
  {
   if (tags[i]==0) unseen += 1; 
  }

 // Create the return...
  IndexSet * this = (IndexSet*)malloc(sizeof(IndexSet) + unseen * sizeof(int));
  this->size = unseen;

 // Write out the unseen into the return...
  unseen = 0;
  for (i=0; i<other->size; i++)
  {
   if (tags[i]==0)
   {
    this->vals[unseen] = i;
    unseen += 1; 
   }
  }

 // Clean up and return...
  free(tags);
  return this;
}

void IndexSet_init_all(IndexSet * this)
{
 int i;
 for (i=0; i<this->size; i++)
 {
  this->vals[i] = i; 
 }
}

void IndexSet_init_bootstrap(IndexSet * this, unsigned int key[4])
{
 int i;
 
 PhiloxRNG rng;
 PhiloxRNG_init(&rng, key);
    
 for (i=0; i<this->size; i++)
 {
  // Get some random data... 
   unsigned int r = PhiloxRNG_next(&rng);
   
  // Store a random value in the range...
   this->vals[i] = r % this->size;
 }
}



// Methods for the IndexView...
void IndexView_init(IndexView * this, IndexSet * source)
{
 this->size = source->size;
 this->vals = source->vals;
}

void IndexView_split(IndexView * this, DataMatrix * dm, char test_code, void * test, IndexView * pass, IndexView * fail)
{
 // Code to test function...
  DoTest test_func = CodeToTest[(unsigned char)test_code];
  
 // The following procedes like a single step of quick sort - shuffling the indices in the array until we have a fail half and pass half...
  int low = 0;
  int high = this->size - 1;
  
  while (1)
  {
   // Find next low value that passes...
    while (low<high)
    {
     int eval = test_func(test, dm, this->vals[low]);
     if (eval!=0) break;
     low += 1;
    }
    
   // Find next high value that fails...
    while (low<high)
    {
     int eval = test_func(test, dm, this->vals[high]);
     if (eval==0) break;
     high -= 1;
    }
    
   // End if needed...
    if (!(low<high)) break;

   // Swap them, move onwards...
    int temp = this->vals[low];
    this->vals[low] = this->vals[high];
    this->vals[high] = temp;
    
    low += 1;
    high -= 1;
  }
  
 // At this point we have seperated the view into a pass half and fail half - dump the indices into the output...
  if (test_func(test, dm, this->vals[low])==0) low += 1;
  
  fail->size = low;
  fail->vals = this->vals;
  
  pass->size = this->size - low;
  pass->vals = this->vals + low;
}



void Setup_IndexSet(void)
{
 import_array();  
}
