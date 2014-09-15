// Copyright 2014 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



#include "learner.h"

#include "tree.h"



// Define the Node object - just a structure to layout the memory block used...
typedef struct Node Node;

struct Node
{
 // Indices to its children objects - could be either a node or a summary (leaf)...
  int fail;
  int pass;

 // The test to decide which direction to go - code stored here, with the actual test data stored immediatly afterwards...
  char code;
  char test[0];
};



// Array of pointers, used to do the indexing whilst building a tree; includes type codes - simple two layer structure, with top layer being resized as needed...
#define PtrArrayBlockSize 256
typedef struct PtrArrayBlock PtrArrayBlock;
typedef struct PtrArray PtrArray;

struct PtrArrayBlock
{
 void * ptr[PtrArrayBlockSize];
 char code[PtrArrayBlockSize];
};

struct PtrArray
{
 int count; // Number of items within.
 
 int block_count; // Number of block pointers within - can be null pointers.
 PtrArrayBlock ** data; 
};



// Methods for above...
PtrArray * PtrArray_new(void)
{
 PtrArray * this = (PtrArray*)malloc(sizeof(PtrArray));
 this->count = 0;
 this->block_count = 0;
 this->data = NULL;
 
 return this;
}

void PtrArray_delete(PtrArray * this)
{
 int i;
 for (i=0; i<this->block_count; i++)
 {
  PtrArrayBlock * targ = this->data[i];
  if (targ!=NULL)
  {
   int j;
   for (j=0; j<PtrArrayBlockSize; j++) free(targ->ptr[j]);  
   
   free(targ);
  }
 }
 
 free(this->data);
 free(this);
}

// Stores the given memory block - takes ownership and will free is when this dies/is overwritten...
void PtrArray_set(PtrArray * this, int index, char code, void * ptr)
{
 if (index>=this->count) this->count = index + 1; 
  
 int block = index / PtrArrayBlockSize;
 int offset = index % PtrArrayBlockSize;
 
 if (block>=this->block_count)
 {
  int new_count = block + 64;
  this->data = (PtrArrayBlock**)realloc(this->data, new_count*sizeof(PtrArrayBlock*));
  
  int i;
  for (i=this->block_count; i<new_count; i++) this->data[i] = NULL; 
  this->block_count = new_count;
 }
 
 if (this->data[block]==NULL)
 {
  this->data[block] = (PtrArrayBlock*)malloc(sizeof(PtrArrayBlock));
  
  int i;
  for (i=0; i<PtrArrayBlockSize; i++)
  {
   this->data[block]->ptr[i] = NULL;
   this->data[block]->code[i] = 0;
  }
 }
 
 PtrArrayBlock * targ = this->data[block];
 free(targ->ptr[offset]);
 targ->ptr[offset] = ptr;
 targ->code[offset] = code;
}

// Returns the memory block at the given position, including the tag if requested. Do not call if you never set it in the first palce...
void * PtrArray_get(PtrArray * this, int index, char * code)
{
 int block = index / PtrArrayBlockSize;
 int offset = index % PtrArrayBlockSize;
 
 PtrArrayBlock * targ = this->data[block];
 if (code!=NULL) *code = targ->code[offset];
 return targ->ptr[offset];
}



// Little helper - returns the size of a code block, for the given number of type codes...
size_t Tree_type_size(int count)
{
 size_t ret = count;
 if ((ret%sizeof(int))!=0)
 {
  ret += sizeof(int) - (ret%sizeof(int));
 }
 return ret;
}



// The learn method and supporting function - fairly involved due to the insane packing requirements. Learning is done recursivly using a general array of pointers structure above, so it can all be packed at the end...
void Node_learn(PtrArray * store, int index, int depth, TreeParam * param, IndexView * view, ReportSummarisation rs, void * rs_ptr)
{
 // Attempt to learn a split, unless we have already reached some split-preventing limit...
  int do_node = 0;
  if (depth<param->max_splits)
  {
   do_node = LearnerSet_optimise(param->ls, param->is, view, param->opt_features, depth, param->key);
  }
  
 // If we have found a split create the node...
  Node * node = NULL;
  if (do_node!=0)
  {
   node = (Node*)malloc(sizeof(Node) + LearnerSet_size(param->ls));
   node->code = LearnerSet_code(param->ls);
   LearnerSet_fetch(param->ls, (void*)node->test);
  }
 
 // Apply the node to the data, and split it...
  IndexView pass;
  IndexView fail;
  
  if (node!=NULL)
  {
   IndexView_split(view, param->x, node->code, (void*)node->test, &pass, &fail);
  
   // If either split is too small unroll and cancel the node...
    if ((pass.size<param->min_exemplars)||(fail.size<param->min_exemplars))
    {
     free(node);
     node = NULL;
    }
  }
 
 // If we have a node record it, otherwise create and store a summary...
  if (node!=NULL)
  {
   PtrArray_set(store, index, 'N', (void*)node); 
  }
  else
  {
   size_t size = SummarySet_init_size(param->y, view, param->summary_codes);
   SummarySet * ss = (SummarySet*)malloc(size);
   SummarySet_init(ss, param->y, view, param->summary_codes);
   
   PtrArray_set(store, index, 'S', (void*)ss);
   if (rs!=NULL) rs(view->size, rs_ptr);
  }
 
 // If its a node then we need to recurse...
  if (node!=NULL)
  {
   // First do the fail half...
    node->fail = store->count;
    Node_learn(store, node->fail, depth+1, param, &fail, rs, rs_ptr);
    
   // Then do the pass half...
    node->pass = store->count;
    Node_learn(store, node->pass, depth+1, param, &pass, rs, rs_ptr);
  }
}


Tree * Tree_learn(TreeParam * param, IndexSet * indices, ReportSummarisation rs, void * rs_ptr)
{
 // Create the temporary storage of all the blocks...
  PtrArray * store = PtrArray_new();
  
 // Start from the top and learn the tree with a recursive function...
  IndexView view;
  IndexView_init(&view, indices);
  
  Node_learn(store, 1, 0, param, &view, rs, rs_ptr);
 
 // Build memory block zero - the block type codes; count how many bytes all the blocks consume at the same time...
  size_t type_size = Tree_type_size(store->count);
  
  char * types = (char*)malloc(type_size);
  PtrArray_set(store, 0, 'T', (void*)types);
  types[0] = 'T';
  
  int i;
  size_t total_size = type_size;
  for (i=1; i<store->count; i++)
  {
   void * block = PtrArray_get(store, i, types + i);
   
   if (types[i]=='N') // Node or summary
   {
    Node * targ = (Node*)block;
    total_size += sizeof(Node) + Test_size(targ->code, (void*)targ->test);
   }
   else // types[i] = 'S'
   {
    total_size += SummarySet_size((SummarySet*)block);
   }
  }
 
 // Allocate the Tree memory block and copy all of the data over...
  size_t tree_size = Tree_head_size();
  
  Tree * this = (Tree*)malloc(tree_size + total_size);
  
  this->magic[0] = 'F';
  this->magic[1] = 'R';
  this->magic[2] = 'F';
  this->magic[3] = 'T';
  this->revision = 1;
  this->size = tree_size + total_size;
  this->index = NULL;
  
  this->objects = store->count;
  size_t offset = tree_size;
  
  for (i=0; i<this->objects; i++)
  {
   // Fetch the block and calculate its size...
    void * block = PtrArray_get(store, i, NULL);
    size_t size;
    if (i==0) size = type_size;
    else
    {
     if (types[i]=='N') // Node or summary
     {
      Node * targ = (Node*)block;
      size = sizeof(Node) + Test_size(targ->code, (void*)targ->test);
     }
     else // types[i] = 'S'
     {
      size = SummarySet_size((SummarySet*)block);
     }
    }
   
   // Copy it over...
    memcpy((char*)this + offset, block, size);
    offset += size;
  }
 
 // Clean up the store...
  PtrArray_delete(store);
 
 // Calculate the index structure for the tree...
  Tree_init(this);
 
 // Return the shiny new tree...
  return this;
}



// The rest of the Tree methods...
int Tree_safe(Tree * this)
{
 if ((this->magic[0]!='F')||(this->magic[1]!='R')||(this->magic[2]!='F')||(this->magic[3]!='T')||(this->revision!=1))
 {
  // Not a tree...
   PyErr_SetString(PyExc_ValueError, "Tree has bad header");
   return 0; 
 }
 
 return 1;
}

int Tree_init(Tree * this)
{
 if (Tree_safe(this)==0) return 0;
 
 this->index = (void**)malloc(this->objects * sizeof(void*));
 
 size_t offset = Tree_head_size();
 this->index[0] = (char*)this + offset;
 offset += Tree_type_size(this->objects);
 
 int i;
 for (i=1; i<this->objects; i++)
 {
  this->index[i] = (char*)this + offset;
  if ('N'==((char*)this->index[0])[i])
  {
   // Node...
    Node * targ = (Node*)this->index[i];
    offset += sizeof(Node) + Test_size(targ->code, targ->test);
  }
  else
  {
   // Summary...
    offset += SummarySet_size((SummarySet*)this->index[i]);
  }
  
  if (offset>this->size)
  {
   free(this->index);
   this->index = NULL;
   PyErr_SetString(PyExc_ValueError, "Tree structure overruns tree size - corruption.");
   return 0; // Overrun of block - data must be corrupted! 
  }
 }
 
 if (offset!=this->size)
 {
  free(this->index);
  this->index = NULL;
  PyErr_SetString(PyExc_ValueError, "Tree size does not match consumed memory.");
  return 0; // If we don't eat precisely all the data we have an issue.
 }
 
 return 1; // Success.
}

void Tree_deinit(Tree * this)
{
 free(this->index); 
}



size_t Tree_head_size(void)
{
 return sizeof(Tree) + sizeof(long long) - sizeof(void*);
}

size_t Tree_size(Tree * this)
{
 return this->size; 
}

int Tree_objects(Tree * this)
{
 return this->objects;
}



SummarySet * Tree_run_rec(Tree * this, int object, DataMatrix * x, int exemplar)
{
 // Fetch the object, behavour depends on type...
  char code = ((char*)this->index[0])[object];
  void * block = this->index[object];
  
  if (code=='N')
  {
   Node * targ = (Node*)block;
   int res = Test(targ->code, (void*)targ->test, x, exemplar);
   if (res==0) return Tree_run_rec(this, targ->fail, x, exemplar);
          else return Tree_run_rec(this, targ->pass, x, exemplar);
  }
  else
  {
   return (SummarySet*)block; 
  }
}

SummarySet * Tree_run(Tree * this, DataMatrix * x, int exemplar)
{
 return Tree_run_rec(this, 1, x, exemplar);
}


void Tree_run_many_rec(Tree * this, int object, DataMatrix * x, IndexView * view, SummarySet ** out, int step)
{
 // Fetch the object, behavour depends on type...
  char code = ((char*)this->index[0])[object];
  void * block = this->index[object];
  
  if (code=='N')
  {
   // Node...
    Node * targ = (Node*)block;
    
    IndexView fail;
    IndexView pass;
    IndexView_split(view, x, targ->code, (void*)targ->test, &pass, &fail);
    
    if (fail.size!=0) Tree_run_many_rec(this, targ->fail, x, &fail, out, step);
    if (pass.size!=0) Tree_run_many_rec(this, targ->pass, x, &pass, out, step);
  }
  else
  {
   // Summary...
    SummarySet * targ = (SummarySet*)block;
    
    int i;
    for (i=0; i<view->size; i++)
    {
     out[step * view->vals[i]] = targ; 
    }
  }
}

void Tree_run_many(Tree * this, DataMatrix * x, IndexSet * is, SummarySet ** out, int step)
{
 IndexView view;
 IndexView_init(&view, is);
 
 Tree_run_many_rec(this, 1, x, &view, out, step);
}


PyObject * Tree_human_rec(Tree * this, int object)
{
 // Fetch the object, behavour depends on type...
  char code = ((char*)this->index[0])[object];
  void * block = this->index[object];
  
  if (code=='N')
  {
   // Node...
    Node * targ = (Node*)block;
    
    PyObject * test = Test_string(targ->code, targ->test);
    PyObject * pass = Tree_human_rec(this, targ->pass);
    PyObject * fail = Tree_human_rec(this, targ->fail);
    
    return Py_BuildValue("{sNsNsN}", "test", test, "pass", pass, "fail", fail);
  }
  else
  {
   // Summary...
    SummarySet * targ = (SummarySet*)block;
    return SummarySet_string(targ);
  }
}

PyObject * Tree_human(Tree * this)
{
 return Tree_human_rec(this, 1); 
}



void Setup_Tree(void)
{
 import_array();  
}
