// Copyright 2013 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



#include "spatial.h"

#include <stdlib.h>



// The conveniance methods that handle all Spatial objects...
Spatial Spatial_new(const SpatialType * type, DataMatrix * dm)
{
 return type->init(dm);
}

void Spatial_delete(Spatial this)
{
 const SpatialType * type = *(const SpatialType**)this;
 type->deinit(this);
}


const SpatialType * Spatial_type(Spatial this)
{
 return *(const SpatialType**)this;
}

DataMatrix * Spatial_dm(Spatial this)
{
 const SpatialType * type = *(const SpatialType**)this;
 return type->dm(this);
}


void Spatial_start(Spatial this, const float * centre, float range)
{
 const SpatialType * type = *(const SpatialType**)this;
 type->start(this, centre, range);
}

int Spatial_next(Spatial this)
{
 const SpatialType * type = *(const SpatialType**)this;
 return type->next(this); 
}



// Implimentation of the brute force spatial indexer...
typedef struct BruteForce BruteForce;

struct BruteForce
{
 const SpatialType * type;
 
 DataMatrix * dm;
 
 int next;
};



Spatial BruteForce_new(DataMatrix * dm)
{
 BruteForce * this = (BruteForce*)malloc(sizeof(BruteForce));
 
 this->type = &BruteForceType;
 this->dm = dm;
 this->next = -1;
 
 return this;
}


void BruteForce_delete(Spatial this)
{
 free(this); 
}


DataMatrix * BruteForce_dm(Spatial self)
{
 BruteForce * this = (BruteForce*)self;
 return this->dm;
}


void BruteForce_start(Spatial self, const float * centre, float range)
{
 BruteForce * this = (BruteForce*)self;
 this->next = 0;
}


int BruteForce_next(Spatial self)
{
 BruteForce * this = (BruteForce*)self;
 int ret = this->next;
 
 if (this->next>=0)
 {
  this->next += 1;
  if (this->next >= this->dm->exemplars)
  {
   this->next = -1; 
  }
 }
 
 return ret;
}



const SpatialType BruteForceType =
{
 "brute_force",
 "Does nothing - for every calculation all items in the data matrix will be considered. Only crazy people use this.",
 BruteForce_new,
 BruteForce_delete,
 BruteForce_dm,
 BruteForce_start,
 BruteForce_next,
};



// Implimentation of the dual dimension iterator spatial indexer...
typedef struct IterDual IterDual;

struct IterDual
{
 const SpatialType * type;
 
 DataMatrix * dm;
 
 int indices; // Number of indices that have a spatial component.
 int * original; // For each index its source in the dm object.
 
 int valid; // 1 if we are iterating, 0 if not.
 int * low; // Starting value for each index in the current range.
 int * pos; // Position in the current iteration.
 int * high; // Ending value for each index in the current range (exclusive).
};



Spatial IterDual_new(DataMatrix * dm)
{
 IterDual * this = (IterDual*)malloc(sizeof(IterDual));
 
 this->type = &IterDualType;
 this->dm = dm;
 
 this->indices = 0;
 int i;
 for (i=0; i<this->dm->array->nd; i++)
 {
  if (this->dm->dt[i]!=DIM_FEATURE) this->indices += 1; 
 }
 
 this->original = (int*)malloc(this->indices * sizeof(int));
 this->indices = 0;
 for (i=0; i<this->dm->array->nd; i++)
 {
  if (this->dm->dt[i]!=DIM_FEATURE)
  {
   this->original[this->indices] = i;
   this->indices += 1;
  }
 }
 
 this->valid = 0;
 this->low = (int*)malloc(this->indices * sizeof(int));
 this->pos = (int*)malloc(this->indices * sizeof(int));
 this->high = (int*)malloc(this->indices * sizeof(int));
 
 return this;
}


void IterDual_delete(Spatial self)
{
 IterDual * this = (IterDual*)self;
 
 free(this->high);
 free(this->pos);
 free(this->low);
 free(this->original);
 
 free(this);
}


DataMatrix * IterDual_dm(Spatial self)
{
 IterDual * this = (IterDual*)self;
 return this->dm;
}


void IterDual_start(Spatial self, const float * centre, float range)
{
 IterDual * this = (IterDual*)self;
 
 // Go through and calculate low and high for each dimension, setting pos to low...
  int i;
  for (i=0; i<this->indices; i++)
  {
   int oi = this->original[i];
   npy_intp size = this->dm->array->dimensions[oi];
   
   if (this->dm->dt[oi]==DIM_DATA)
   {
    // No optimisation possible - just set it to the entire range...
     this->low[i] = 0;
     this->pos[i] = 0;
     this->high[i] = size;
   }
   else
   {
    // We can optimise, and iterate only part of this dimension of the datamatrix...
     float low  = (centre[oi] - range) / this->dm->mult[oi];
     float high = (centre[oi] + range) / this->dm->mult[oi];
     
     this->low[i] = (int)ceil(low);
     if (this->low[i]<0) this->low[i] = 0;
     this->pos[i] = this->low[i];
     this->high[i] = (int)ceil(high);
     if (this->high[i]>size) this->high[i] = size;
   }
  }
 
 // Indicate that we are in a valid state...
  this->valid = 1;
}


int IterDual_next(Spatial self)
{
 IterDual * this = (IterDual*)self;
 if (this->valid==0) return -1;

 // Calculate the index for the current position...
  int ret = 0;
  int i;
  for (i=0; i<this->indices; i++)
  {
   int oi = this->original[i];
   ret *= this->dm->array->dimensions[oi];
   ret += this->pos[i];
  }

 // Move to the next index...
  for (i=this->indices-1;; i--)
  {
   this->pos[i] += 1;
   if (this->pos[i]<this->high[i]) break;
   if (i==0) break;
   this->pos[i] = this->low[i];
  }
  
  if (this->pos[0]>=this->high[0]) this->valid = 0;

 // Return the index...
  return ret;
}



const SpatialType IterDualType =
{
 "iter_dual",
 "Limits the values to consider by using the dual dimensions in the data matrix, as for each of them a search range can be calculated to include all within the bound. Within the region defined its brute force however - if the dual dimensions don't limit the data enough it might make more sense to use a different structure. If there are no dual dimensions it ends up equivalent to brute force. Its typical use is when doing mean shift on images.",
 IterDual_new,
 IterDual_delete,
 IterDual_dm,
 IterDual_start,
 IterDual_next,
};



// Implimentation of the KD-tree spatial indexer...
typedef struct PosFeat PosFeat;
struct PosFeat
{
 int pos;
 float feat;
};

int sort_pos_feat(const void * lhs, const void * rhs)
{
 const PosFeat * l = (PosFeat*)lhs;
 const PosFeat * r = (PosFeat*)rhs;
 
 if (l->feat < r->feat) return -1;
 if (l->feat > r->feat) return 1;
 return 0;
}



typedef struct KDNode KDNode;
struct KDNode
{
 KDNode * parent;
 KDNode * child_low;
 KDNode * child_high;
 
 int low; // Range of values the tree covers in the indices array.
 int high; // Exclusive.
 
 float range[0]; // 2* number of features, with the low range indexed for feature i as range[i*2], and the high range [i*2+1].
};



typedef struct KDTree KDTree;
struct KDTree
{
 const SpatialType * type;
 
 DataMatrix * dm;
 
 int * indices; // All the indices - the nodes of the KD tree just have to store ranges into this, as we rearrange them during construction - allows for some fun optimisation tricks.
 KDNode * root; // Root of the tree.
 
 KDNode * targ; // For when iterating - allways a node we are searching.
 int offset; // Offset of iterating.
 
 const float * centre; // Bounding box of current iterations.
 float range;
};



KDNode * KDNode_new(DataMatrix * dm, int * indices, int low, int high, int depth, PosFeat * scratch)
{
 KDNode * this = (KDNode*)malloc(sizeof(KDNode) + dm->feats * 2 * sizeof(float));
 
 // Basic initialisation...
  this->parent = NULL;
  this->child_low = NULL;
  this->child_high = NULL;
  this->low = low;
  this->high = high;
  
 // Calculate the range...
  float * fv = DataMatrix_fv(dm, indices[low], NULL);
  int i;
  for (i=0; i<dm->feats; i++)
  {
   this->range[i*2] = fv[i]; 
   this->range[i*2+1] = fv[i];
  }
  
  for (i=low+1; i<high; i++)
  {
   fv = DataMatrix_fv(dm, indices[i], NULL);
   int j;
   for (j=0; j<dm->feats; j++)
   {
    if (this->range[j*2]>fv[j]) this->range[j*2] = fv[j];
    if (this->range[j*2+1]<fv[j]) this->range[j*2+1] = fv[j];
   }
  }
  
 // Decide if we are going to divide further or not...
  if (depth>64) return this;
  if ((high-low)<6) return this;
 
 // Choose a division direction - the one with the largest range; possible condition for leaving as well...
  int split_feat = 0;
  float split_range = this->range[split_feat*2+1] - this->range[split_feat*2];
  for (i=1; i<dm->feats; i++)
  {
   float range = this->range[i*2+1] - this->range[i*2];
   if (range>split_range)
   {
    split_feat = i;
    split_range = range;
   }
  }
  if (split_range<1e-3) return this;
  
 // Sort the indices by the feature...
  for (i=low; i<high; i++)
  {
   scratch[i-low].pos = indices[i];
   scratch[i-low].feat = DataMatrix_fv(dm, indices[i], NULL)[split_feat];
  }
  
  qsort(scratch, high-low, sizeof(PosFeat), sort_pos_feat);
  
  for (i=low; i<high; i++)
  {
   indices[i] = scratch[i-low].pos; 
  }
  
 // Do the split...
  int half = (low + high) / 2;
  this->child_low = KDNode_new(dm, indices, low, half, depth+1, scratch);
  this->child_high = KDNode_new(dm, indices, half, high, depth+1, scratch);
  
 // Assign parents...
  this->child_low->parent = this;
  this->child_high->parent = this;
 
 // Return...
  return this;
}


void KDNode_delete(KDNode * this)
{
 if (this->child_low!=NULL) KDNode_delete(this->child_low);
 if (this->child_high!=NULL) KDNode_delete(this->child_high);
 free(this);
}


KDNode * KDNode_next_down(KDNode * this, DataMatrix * dm, const float * centre, float range)
{
 int i;
 // If its a miss return null - we are not going this way...
  int within = 1;
  
  for (i=0; i<dm->feats; i++)
  {
   float req_low  = centre[i] - range;
   float req_high = centre[i] + range;
   float bb_low = this->range[i*2];
   float bb_high = this->range[i*2+1];
   
   if (req_high<bb_low) return NULL;
   if (req_low>bb_high) return NULL;
   
   if (req_low>bb_low) within = 0;
   if (req_high<bb_high) within = 0;
  }
 
 // If the entire node is within the space return it - actual calculation is above...
  if (within!=0) return this;
 
 // If it has no children return it - we have got as close to the split line as we can...
  if (this->child_low==NULL) return this;
 
 // Check each child in turn...
  KDNode * ret = KDNode_next_down(this->child_low, dm, centre, range);
  if (ret!=NULL) return ret;
  return KDNode_next_down(this->child_high, dm, centre, range);
}


KDNode * KDNode_next_up(KDNode * this, DataMatrix * dm, const float * centre, float range)
{
 while (this->parent!=NULL)
 {
  KDNode * child = this;
  this = this->parent;
  
  if (this->child_low==child)
  {
   KDNode * ret = KDNode_next_down(this->child_high, dm, centre, range);
   if (ret!=NULL) return ret;
  }
 }
 
 return NULL;
}



Spatial KDTree_new(DataMatrix * dm)
{
 KDTree * this = (KDTree*)malloc(sizeof(KDTree));
 
 this->type = &KDTreeType;
 this->dm = dm;
 
 this->indices = (int*)malloc(this->dm->exemplars * sizeof(int));
 int i;
 for (i=0; i<this->dm->exemplars; i++) this->indices[i] = i;
 
 PosFeat * scratch = (PosFeat*)malloc(this->dm->exemplars * sizeof(PosFeat));
 this->root = KDNode_new(this->dm, this->indices, 0, this->dm->exemplars, 0, scratch);
 free(scratch);
 
 this->targ = NULL;
 
 return this;
}


void KDTree_delete(Spatial self)
{
 KDTree * this = (KDTree*)self;
 
 free(this->indices);
 KDNode_delete(this->root);
 
 free(this);
}


DataMatrix * KDTree_dm(Spatial self)
{
 KDTree * this = (KDTree*)self;
 return this->dm;
}


void KDTree_start(Spatial self, const float * centre, float range)
{
 KDTree * this = (KDTree*)self;
 
 this->targ = KDNode_next_down(this->root, this->dm, centre, range);
 this->offset = 0;
 this->centre = centre;
 this->range = range;
}


int KDTree_next(Spatial self)
{
 KDTree * this = (KDTree*)self;
 if (this->targ==NULL) return -1;
 
 // Calculate the return...
  int ret = this->indices[this->targ->low + this->offset];
 
 // Move to the next position...
  this->offset += 1;
  if ((this->offset+this->targ->low)>=this->targ->high)
  {
   this->targ = KDNode_next_up(this->targ, this->dm, this->centre, this->range);
   this->offset = 0;
  }
 
 // Return the return!..
  return ret;
}



const SpatialType KDTreeType =
{
 "kd_tree",
 "A standard kd-tree on the feature vectors - best choice if the data has no dual dimensions.",
 KDTree_new,
 KDTree_delete,
 KDTree_dm,
 KDTree_start,
 KDTree_next,
};



// List of spatial indexing methods provide by the system...
const SpatialType * ListSpatial[] =
{
 &BruteForceType,
 &IterDualType,
 &KDTreeType,
 NULL
};



// Dummy function, to make a warning go away because it was annoying me...
void SpatialModule_IgnoreMe(void)
{
 import_array();  
}
