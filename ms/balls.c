// Copyright 2013 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



#include "balls.h"

#include <stdlib.h>
#include <stdint.h>
#include <math.h>



// The conveniance methods that handle all Balls objects...
Balls Balls_new(const BallsType * type, int dims, float radius)
{
 Balls this = type->init(dims, radius);
 return this;
}

void Balls_delete(Balls this)
{
 const BallsType * type = *(const BallsType**)this;
 type->deinit(this);
}

int Balls_dims(Balls this)
{
 const BallsType * type = *(const BallsType**)this;
 return type->dims(this);
}

int Balls_count(Balls this)
{
 const BallsType * type = *(const BallsType**)this;
 return type->count(this); 
}

int Balls_create(Balls this, const float * pos, float radius)
{
 const BallsType * type = *(const BallsType**)this;
 return type->create(this, pos, radius);
}

const float * Balls_pos(Balls this, int index)
{
 const BallsType * type = *(const BallsType**)this;
 return type->pos(this, index);
}

float Balls_radius(Balls this, int index)
{
 const BallsType * type = *(const BallsType**)this;
 return type->radius(this, index);
}

int Balls_within(Balls this, const float * pos)
{
 const BallsType * type = *(const BallsType**)this;
 return type->within(this, pos);
}



// Implimentation using a simple (linked) list of coordinates/radii and a brute force search...
typedef struct BallsNode BallsNode;
struct BallsNode
{
 BallsNode * next;
 
 int index;
 
 float radius;
 float pos[0];
};

typedef struct BallsList BallsList;
struct BallsList
{
 const BallsType * type;
 
 int dims;
 int count;
 
 BallsNode * first;
};



Balls BallsList_new(int dims, float radius)
{
 BallsList * this = (BallsList*)malloc(sizeof(BallsList));
 
 this->type = &BallsListType;
 this->dims = dims;
 this->count = 0;
 this->first = NULL;
 
 return (Balls)this;
}

void BallsList_delete(Balls self)
{
 BallsList * this = (BallsList*)self;
 
 while (this->first!=NULL)
 {
  BallsNode * to_die = this->first;
  this->first = this->first->next;
  free(to_die);
 }
 
 free(this);
}

int BallsList_dims(Balls self)
{
 BallsList * this = (BallsList*)self;
 return this->dims;
}

int BallsList_count(Balls self)
{
 BallsList * this = (BallsList*)self;
 return this->count;
}

int BallsList_create(Balls self, const float * pos, float radius)
{
 BallsList * this = (BallsList*)self;
 
 BallsNode * nbn = (BallsNode*)malloc(sizeof(BallsNode) + this->dims * sizeof(float));
 
 nbn->next = this->first;
 this->first = nbn;
 
 nbn->index = this->count;
 this->count += 1;
 
 nbn->radius = radius;
 int i;
 for (i=0; i<this->dims; i++) nbn->pos[i] = pos[i];
 
 return nbn->index;
}

const float * BallsList_pos(Balls self, int index)
{
 BallsList * this = (BallsList*)self;
 
 BallsNode * targ = this->first;
 while (targ!=NULL)
 {
  if (targ->index==index)
  {
   return targ->pos; 
  }
  targ = targ->next; 
 }
  
 return NULL;
}

float BallsList_radius(Balls self, int index)
{
 BallsList * this = (BallsList*)self;
 
 BallsNode * targ = this->first;
 while (targ!=NULL)
 {
  if (targ->index==index)
  {
   return targ->radius; 
  }
  targ = targ->next; 
 }
  
 return 0.0;
}

int BallsList_within(Balls self, const float * pos)
{
 BallsList * this = (BallsList*)self;
 
 BallsNode * targ = this->first;
 while (targ!=NULL)
 {
  float distSqr = 0.0;
  int i;
  for (i=0; i<this->dims; i++)
  {
   float delta = targ->pos[i] - pos[i];
   distSqr += delta*delta;
  }
  
  if (distSqr<=targ->radius*targ->radius)
  {
   return targ->index; 
  }
  
  targ = targ->next; 
 }
 
 return -1;
}



const BallsType BallsListType =
{
 "list",
 "Simple linked list of hyper-spheres, with brute force search.",
 BallsList_new,
 BallsList_delete,
 BallsList_dims,
 BallsList_count,
 BallsList_create,
 BallsList_pos,
 BallsList_radius,
 BallsList_within,
};



// Implimentation using spatial hashing - using cryptographic hashing (The philox 4x32 algorithm, with only 7 rounds.) alongside cuckoo hashing, as indexing speed is much more important than insertion, and the fact that baby cuckoos are murdering sociopaths amuses me...
typedef struct Ball Ball;
struct Ball
{
 float radius; 
 float pos[0];
};

typedef struct BallsBucket BallsBucket;
struct BallsBucket
{
 BallsBucket * next; // Not for the hash - multiple balls can intersect the same space.
 int index; // Of the ball.
 unsigned int hashA; // For the hash, before it is reduced to the range of the table.
 unsigned int hashB;
};

typedef struct BallsHash BallsHash;
struct BallsHash
{
 const BallsType * type;
 
 int dims;
 float step; // For discretising coordinates
 
 int count;
 int storage; // Size of the storage - has to be resized when count excedes it.
 Ball * balls;
 
 int size; // Of the hash tables - doubles each time we run out.
 BallsBucket ** tableA;
 BallsBucket ** tableB;
 
 int * low; // Temporary storage.
 int * pos;
 int * high;
};



Ball * BallsHash_get(BallsHash * this, int index)
{
 char * ptr = (char*)this->balls;
 ptr += index * (sizeof(Ball) + this->dims * sizeof(float));
 return (Ball*)ptr;
}



static unsigned int mul_hi(unsigned int a, unsigned int b)
{
 uint64_t _a = a;
 uint64_t _b = b;
 
 return (_a * _b) >> 32;
}

// out is the counter on entry, the output when done.
static void philox(unsigned int out[4])
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

// Given a position this hashes it, outputing its two hash values...
void BallsHash_hash(Balls self, const float * pos, unsigned int * hashA, unsigned int * hashB)
{
 BallsHash * this = (BallsHash*)self;
 
 *hashA = 0;
 *hashB = 0;
 
 // Loop and xor random data for each dimension into the hashes - input for each hash is 4 values, so we do 4 dimensions per hash, until they are consumed...
  int base = 0;
  int prev = 0;
  while (base<this->dims)
  {
   unsigned int out[4];
   
   int i;
   for (i=0; i<4; i++)
   {
    int oi = base + i;
    if (oi<this->dims)
    {
     prev += (int)floor(pos[oi]/this->step);
     out[i] = (unsigned int)prev;
    }
    else out[i] = 0; 
   }
   
   philox(out);
   
   *hashA = *hashA ^ out[0];
   *hashB = *hashB ^ out[1];
    
   base += 4; 
  }
}

// Given a position, already quantised, outputs its two hashes...
void BallsHash_hash_int(Balls self, const int * pos, unsigned int * hashA, unsigned int * hashB)
{
 BallsHash * this = (BallsHash*)self;
 
 *hashA = 0;
 *hashB = 0;
 
 // Loop and xor random data for each dimension into the hashes - input for each hash is 4 values, so we do 4 dimensions per hash, until they are consumed...
  int base = 0;
  int prev = 0;
  while (base<this->dims)
  {
   unsigned int out[4];
   
   int i;
   for (i=0; i<4; i++)
   {
    int oi = base + i;
    if (oi<this->dims)
    {
     prev += pos[oi];
     out[i] = (unsigned int)prev;
    }
    else out[i] = 0; 
   }
   
   philox(out);
   
   *hashA = *hashA ^ out[0];
   *hashB = *hashB ^ out[1];
    
   base += 4; 
  }
}

// Inserts an item - if it is unable to it will return an entry that has been dropped from the hash table (Usually means its time to grow!)...
BallsBucket * BallsHash_basic_insert(Balls self, BallsBucket * targ)
{
 BallsHash * this = (BallsHash*)self;
 
 int i;
 for (i=0; (i<1024) && (targ!=NULL); i++)
 {
  int side = i%2;
  
  int locA = targ->hashA % this->size;
  int locB = targ->hashB % this->size;
  
  if (this->tableB[locB]==NULL) side = 1;
  if (this->tableA[locA]==NULL) side = 0;
    
  if (side==0)
  {
   BallsBucket * temp = this->tableA[locA];
   this->tableA[locA] = targ;
   if ((temp!=NULL)&&(targ->hashA==temp->hashA))
   {
    targ->next = temp;
    targ = NULL;
   }
   else
   {
    targ = temp;
   }
  }
  else
  {
   BallsBucket * temp = this->tableB[locB];
   this->tableB[locB] = targ;
   if ((temp!=NULL)&&(targ->hashB==temp->hashB))
   {
    targ->next = temp;
    targ = NULL;
   }
   else
   {
    targ = temp;
   }
  }
 }
 
 return targ;
}

// Rehashes everything - typically part of grow...
BallsBucket * BallsHash_rehash(Balls self)
{
 BallsHash * this = (BallsHash*)self;
 
 // Loop through and move all items...
  int i;
  for (i=0; i<this->size; i++)
  {
   BallsBucket * targ = this->tableA[i];
   if (targ!=NULL)
   {
    this->tableA[i] = NULL;
    targ = BallsHash_basic_insert(self, targ);
    if (targ!=NULL) return targ;
   }
   
   targ = this->tableB[i];
   if (targ!=NULL)
   {
    this->tableB[i] = NULL;
    targ = BallsHash_basic_insert(self, targ);
    if (targ!=NULL) return targ;
   }
  }
  
 return NULL;
}

// Doubles the size of the hash table - will return a BallsBucket that has been dropped on error, leaving the table in an error state (Memory safe, but not everything in the right place)...
BallsBucket * BallsHash_grow(Balls self)
{
 BallsHash * this = (BallsHash*)self;
 
 // Make the hash table larger, leaving all items where they currently are...
  int oldSize = this->size;
  this->size = this->size * 2 + 1;
  this->tableA = realloc(this->tableA, this->size * sizeof(BallsBucket*));
  this->tableB = realloc(this->tableB, this->size * sizeof(BallsBucket*));
  
  int i;
  for (i=oldSize; i<this->size; i++)
  {
   this->tableA[i] = NULL; 
   this->tableB[i] = NULL;
  }
 
 // Rehash...
  return BallsHash_rehash(self);
}

// An insert that will grow etc until it works or runs out of memory...
void BallsHash_insert(Balls self, BallsBucket * targ)
{
 while (1)
 {
  targ = BallsHash_basic_insert(self, targ);
  if (targ==NULL) return;
  
  BallsBucket * targ2 = BallsHash_grow(self);
  while (targ2!=NULL)
  {
   BallsHash_insert(self, targ2);
   targ2 = BallsHash_rehash(self);
  }
 }
}
 


Balls BallsHash_new(int dims, float radius)
{
 BallsHash * this = (BallsHash*)malloc(sizeof(BallsHash));
 
 this->type = &BallsHashType;
 this->dims = dims;
 this->step = radius * 2.0; // We make out boxes about the size of the average hyper-sphere.
 
 this->count = 0;
 this->storage = 4;
 this->balls = (Ball*)malloc(this->storage * (sizeof(Ball) + this->dims * sizeof(float)));
 
 this->size = 3;
 this->tableA = (BallsBucket**)malloc(this->size * sizeof(BallsBucket*));
 this->tableB = (BallsBucket**)malloc(this->size * sizeof(BallsBucket*));
 
 int i;
 for (i=0; i<this->size; i++)
 {
  this->tableA[i] = NULL;
  this->tableB[i] = NULL;
 }
 
 this->low = (int*)malloc(this->dims * sizeof(int));
 this->pos = (int*)malloc(this->dims * sizeof(int));
 this->high = (int*)malloc(this->dims * sizeof(int));
 
 return (Balls)this;
}

void BallsHash_delete(Balls self)
{
 BallsHash * this = (BallsHash*)self;
 
 int i;
 for (i=0; i<this->size; i++)
 {
  while (this->tableA[i]!=NULL)
  {
   BallsBucket * to_die = this->tableA[i];
   this->tableA[i] = this->tableA[i]->next;
   free(to_die);
  }
  
  while (this->tableB[i]!=NULL)
  {
   BallsBucket * to_die = this->tableB[i];
   this->tableB[i] = this->tableB[i]->next;
   free(to_die);
  }
 }
 
 free(this->high);
 free(this->pos);
 free(this->low);
 
 free(this->tableB);
 free(this->tableA);
 free(this->balls);
 
 free(this);
}

int BallsHash_dims(Balls self)
{
 BallsHash * this = (BallsHash*)self;
 return this->dims;
}

int BallsHash_count(Balls self)
{
 BallsHash * this = (BallsHash*)self;
 return this->count;
}

int BallsHash_create(Balls self, const float * pos, float radius)
{
 BallsHash * this = (BallsHash*)self;
 
 // Create the ball...
  if (this->storage==this->count)
  {
   this->storage *= 2;
   this->balls = (Balls)realloc(this->balls, this->storage * (sizeof(Ball) + this->dims * sizeof(float))); 
  }
  
  Ball * targ = BallsHash_get(this, this->count);
  targ->radius = radius;
  int i;
  for (i=0; i<this->dims; i++) targ->pos[i] = pos[i];
  this->count += 1;
 
 // Calculate and iterate the set of bounding boxes that the hyper-sphere intersects - add an entry for each to the hash...
  for (i=0; i<this->dims; i++)
  {
   this->low[i] = (int)floor((pos[i] - radius)/this->step);
   this->pos[i] = this->low[i];
   this->high[i] = (int)floor((pos[i] + radius)/this->step) + 1;
  }
 
  while (this->pos[0]<this->high[0])
  {
   // Check if the hyper-sphere intersects with the current position...
    float distSqr = 0.0;
    for (i=0; i<this->dims; i++)
    {
     if (pos[i]<(this->pos[i]*this->step))
     {
      float delta = this->pos[i] * this->step - pos[i];
      distSqr += delta*delta;
     }
     else
     {
      if (pos[i]>((this->pos[i]+1)*this->step))
      {
       float delta = pos[i] - (this->pos[i]+1) * this->step;
       distSqr += delta*delta;
      }
     }
    }
    
    if (distSqr<radius*radius)
    {
     // Create the bucket...
      BallsBucket * nbb = (BallsBucket*)malloc(sizeof(BallsBucket));
      nbb->index = this->count - 1;
      
     // Fill in its hashes...
      BallsHash_hash_int(self, this->pos, &nbb->hashA, &nbb->hashB);
      
     // Insert it...
      nbb->next = NULL;
      BallsHash_insert(self, nbb);
    }
    
   // Move to the next position; break if done...
    for (i=this->dims-1;; i--)
    {
     this->pos[i] += 1;
     if (this->pos[i]<this->high[i]) break;
     if (i==0) break;
     this->pos[i] = this->low[i];
    }
  }
  
 return this->count - 1;
}

const float * BallsHash_pos(Balls self, int index)
{
 BallsHash * this = (BallsHash*)self;
 Ball * targ = BallsHash_get(this, index);
 return targ->pos;
}

float BallsHash_radius(Balls self, int index)
{
 BallsHash * this = (BallsHash*)self;
 Ball * targ = BallsHash_get(this, index);
 return targ->radius;
}

int BallsHash_within(Balls self, const float * pos)
{
 BallsHash * this = (BallsHash*)self;
 
 // Calculate the two possible hashes of the position...
  unsigned int hashA;
  unsigned int hashB;
  BallsHash_hash(self, pos, &hashA, &hashB);
  
  int limA = hashA % this->size;
  int limB = hashB % this->size;
  
 // Check both positions, checking all entries in them...
  BallsBucket * targ = this->tableA[limA];
  if ((targ!=NULL)&&(targ->hashA==hashA))
  {
   while (targ)
   {
    Ball * ball = BallsHash_get(this, targ->index);
    
    float distSqr = 0.0;
    int i;
    for (i=0; i<this->dims; i++)
    {
     float delta = ball->pos[i] - pos[i];
     distSqr += delta * delta;
    }
    
    if (distSqr<=(ball->radius*ball->radius))
    {
     return targ->index; 
    }
    
    targ = targ->next; 
   }
  }
  
  targ = this->tableB[limB];
  if ((targ!=NULL)&&(targ->hashB==hashB))
  {
   while (targ)
   {
    Ball * ball = BallsHash_get(this, targ->index);
    
    float distSqr = 0.0;
    int i;
    for (i=0; i<this->dims; i++)
    {
     float delta = ball->pos[i] - pos[i];
     distSqr += delta * delta;
    }
    
    if (distSqr<=(ball->radius*ball->radius))
    {
     return targ->index; 
    }
    
    targ = targ->next; 
   }
  }
  
 // Return -1 - we have missed...
  return -1;
}



const BallsType BallsHashType =
{
 "hash",
 "Uses spatial hashing, recording each sphere at all possible grid cells that a hyper-sphere can appear at. Hashing is quite sophisticated, using the philox 4x32 algorithm and cuckoo hashing - the within query is crazy fast, as prefered.",
 BallsHash_new,
 BallsHash_delete,
 BallsHash_dims,
 BallsHash_count,
 BallsHash_create,
 BallsHash_pos,
 BallsHash_radius,
 BallsHash_within,
};



// List of hyper-sphere indexing structures...
const BallsType * ListBalls[] =
{
 &BallsListType,
 &BallsHashType,
 NULL
};
