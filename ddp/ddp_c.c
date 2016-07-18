// Copyright 2016 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

#include "ddp_c.h"



#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>



// PairCost methods that use a v-table...
void DeletePC(PairCost this)
{
 if (this!=NULL)
 {
  const PairCostType * type = *(const PairCostType **)this;
  type->delete(this);
 }
}

float CostPC(PairCost this, int first, int second)
{
 const PairCostType * type = *(const PairCostType **)this;
 return type->cost(this, first, second);
}

void CostsPC(PairCost this, int first_count, float * first_total, int second_count, float * second_out, int * second_back)
{
 const PairCostType * type = *(const PairCostType **)this;
 return type->costs(this, first_count, first_total, second_count, second_out, second_back);
}

void CostsRevPC(PairCost this, int first_count, float * first_out, int * first_forward, int second_count, float * second_total)
{
 const PairCostType * type = *(const PairCostType **)this; 
 return type->costs_rev(this, first_count, first_out, first_forward, second_count, second_total);
}



// Code for the different PairCost type...
typedef struct Different Different;

struct Different
{
 const PairCostType * type;
 
 float cost_same;
 float cost_diff;
 int offset;
};


PairCost Different_new(PyObject * data)
{
 PyArrayObject * param = (PyArrayObject*)PyArray_FromObject(data, NPY_FLOAT, 1, 1);
 if (param==NULL) return NULL;
 
 Different * this = (Different*)malloc(sizeof(Different));
 this->type = &DifferentType;
 
 this->cost_same = 0.0;
 this->cost_diff = 1.0;
 this->offset = 0;
 
 switch (PyArray_DIMS(param)[0])
 {
  case 0:
  break;
  
  case 1:
   this->cost_diff = *(float*)PyArray_GETPTR1(param, 0);
  break;

  case 2:
   this->cost_diff = *(float*)PyArray_GETPTR1(param, 0);
   this->cost_same = *(float*)PyArray_GETPTR1(param, 1);
  break;
  
  default:
   this->cost_diff = *(float*)PyArray_GETPTR1(param, 0);
   this->cost_same = *(float*)PyArray_GETPTR1(param, 1);
   this->offset = (int)*(float*)PyArray_GETPTR1(param, 2);
  break;
 }
 
 Py_DECREF(param);
 
 return this;
}

void Different_delete(PairCost this_ptr)
{
 Different * this = (Different*)this_ptr;
 free(this);
}

float Different_cost(PairCost this_ptr, int first, int second)
{
 Different * this = (Different*)this_ptr;
 if (first==(second + this->offset)) return this->cost_same;
                                else return this->cost_diff;
}

void Different_costs(PairCost this_ptr, int first_count, float * first_total, int second_count, float * second_out, int * second_back)
{
 Different * this = (Different*)this_ptr;
 int i;
 
 // Special case one label - kinda simple...
  if (first_count==1)
  {
   for (i=0; i<second_count; i++)
   {
    second_out[i] = first_total[0] + (((i+this->offset)==0) ? this->cost_same : this->cost_diff);
    second_back[i] = 0;
   }
   return; 
  }
  
 // Find the indices of the minimum and second minimum in first_total...
  int min;
  int second_min;
  
  if (first_total[0]<first_total[1])
  {
   min = 0;
   second_min = 1;
  }
  else
  {
   min = 1;
   second_min = 0;    
  }
  
  for (i=2; i<first_count;i++)
  {
   if (first_total[i]<first_total[second_min])
   {
    if (first_total[i]<first_total[min])
    {
     second_min = min;
     min = i;
    }
    else
    {
     second_min = i;
    }
   }
  }
  
 // Loop and calculate the value for each entry - the minimum values allow us to shortcut calculation of the other term...
  for (i=0; i<second_count; i++)
  {
   int same = i + this->offset;
   
   if (same!=min)
   {
    second_out[i] = first_total[min] + this->cost_diff;
    second_back[i] = min;
   }
   else
   {
    second_out[i] = first_total[second_min] + this->cost_diff;
    second_back[i] = second_min;
   }
   
   if ((same>=0)&&(same<first_count))
   {
    float total = first_total[same] + this->cost_same;
    if (total<second_out[i])
    {
     second_out[i] = total;
     second_back[i] = same;
    }
   }
  }
}

void Different_costs_rev(PairCost this_ptr, int first_count, float * first_out, int * first_forward, int second_count, float * second_total)
{
 Different * this = (Different*)this_ptr;
 int i;
 
 // Special case one label - kinda simple...
  if (second_count==1)
  {
   for (i=0; i<first_count; i++)
   {
    first_out[i] = second_total[0] + ((i==this->offset) ? this->cost_same : this->cost_diff);
    first_forward[i] = 0;
   }
   return; 
  }
  
 // Find the indices of the minimum and second minimum in second_total...
  int min;
  int second_min;
  
  if (second_total[0]<second_total[1])
  {
   min = 0;
   second_min = 1;
  }
  else
  {
   min = 1;
   second_min = 0;    
  }
  
  for (i=2; i<second_count;i++)
  {
   if (second_total[i]<second_total[second_min])
   {
    if (second_total[i]<second_total[min])
    {
     second_min = min;
     min = i;
    }
    else
    {
     second_min = i;
    }
   }
  }
  
 // Loop and calculate the value for each entry - the minimum values allow us to shortcut calculation of the other term...
  for (i=0; i<first_count; i++)
  {
   int same = i - this->offset;
   
   if (same!=min)
   {
    first_out[i] = second_total[min] + this->cost_diff;
    first_forward[i] = min;
   }
   else
   {
    first_out[i] = second_total[second_min] + this->cost_diff;
    first_forward[i] = second_min;
   }
   
   if ((same>=0)&&(same<second_count))
   {
    float total = second_total[same] + this->cost_same;
    if (total<first_out[i])
    {
     first_out[i] = total;
     first_forward[i] = same;
    }
   }
  }
}


const PairCostType DifferentType =
{
 "different",
 "Sets one cost if they have the same label, another if they are different. Initialisation is an object that can be interpreted as a 1D numpy array of length 2 - first entry is the cost of being different, second entry the cost of being the same. It can optionally have a third entry, which will be an (integer) offset applied to the second set of indices (second node in the chain) to define which is the same. If a length 1 array is provided it assumes the cost of being the same is 0.",
 Different_new,
 Different_delete,
 Different_cost,
 Different_costs,
 Different_costs_rev,
};



// Code for the linear falloff PairCost type...
typedef struct Linear Linear;

struct Linear
{
 const PairCostType * type;
 
 float mult; // Scale of cost function, applied before cap to absolute distance.
 float cap; // Maximum cost.

 float offset; // Added to second random variables position, after scaling.
 float scale; // Scale of spacing between states of second.
};


PairCost Linear_new(PyObject * data)
{
 PyArrayObject * param = (PyArrayObject*)PyArray_FromObject(data, NPY_FLOAT, 1, 1);
 if (param==NULL) return NULL;
 
 Linear * this = (Linear*)malloc(sizeof(Linear));
 this->type = &LinearType;
 
 this->mult = (PyArray_DIMS(param)[0]>=1) ? fabs(*(float*)PyArray_GETPTR1(param, 0)) : 1.0;
 this->offset = (PyArray_DIMS(param)[0]>=2) ? (*(float*)PyArray_GETPTR1(param, 1)) : 0.0;
 this->scale = (PyArray_DIMS(param)[0]>=3) ? (*(float*)PyArray_GETPTR1(param, 2)) : 1.0;
 this->cap = (PyArray_DIMS(param)[0]>=4) ? fabs(*(float*)PyArray_GETPTR1(param, 3)) : INFINITY;
 
 Py_DECREF(param);
 
 return this;
}

void Linear_delete(PairCost this_ptr)
{
 Linear * this = (Linear*)this_ptr;
 free(this);
}

float Linear_cost(PairCost this_ptr, int first, int second)
{
 Linear * this = (Linear*)this_ptr;
 
 float ret = first - (second * this->scale + this->offset);
 ret = this->mult * fabs(ret);
 if (ret>this->cap) return this->cap;
 return ret;
}

void Linear_costs(PairCost this_ptr, int first_count, float * first_total, int second_count, float * second_out, int * second_back)
{
 Linear * this = (Linear*)this_ptr;
 
 // Use the standard convex function trick - gets a little fiddly because we allow negatives in the scalar of label positions...
  // Forwards pass...
   int fi = (this->scale>=0.0) ? 0 : (first_count-1);
   int bfi = fi;
   int si;
   for (si=0; si<second_count; si++)
   {
    // Calculate the position of this entry...
     float pos = si * this->scale + this->offset;
    
    // Walk through the firsts until we have caught up, updating bfi, the index of the lowest cost entry...
     if (this->scale>=0.0)
     {
      while ((fi+1<=pos)&&(fi+1<first_count))
      {
       ++fi;
       
       float cost = first_total[bfi] + this->mult * fabs(fi - bfi);
       if (first_total[fi] <= cost)
       {
        bfi = fi; 
       }
      }
     }
     else
     {
      while ((fi-1>=pos)&&(fi>0))
      {
       --fi;
       
       float cost = first_total[bfi] + this->mult * fabs(fi - bfi);
       if (first_total[fi] <= cost)
       {
        bfi = fi; 
       }
      }
     }
    
    // Calculate and store the cost...
     float cost = this->mult * fabs(bfi - pos);
     if (cost>this->cap) cost = this->cap;
     cost += first_total[bfi];
    
     second_out[si] = cost;
     second_back[si] = bfi;
   }
 
  // Backwards pass...
   fi = (this->scale>=0.0) ? (first_count-1) : 0;
   bfi = fi;
   for (si=second_count-1; si>=0; si--)
   {
    // Calculate the position of this entry...
     float pos = si * this->scale + this->offset;
    
    // Walk through the firsts until we have caught up, updating bfi, the index of the lowest cost entry...
     if (this->scale<=0.0)
     {
      while ((fi+1<=pos)&&(fi+1<first_count))
      {
       ++fi;
       
       float cost = first_total[bfi] + this->mult * fabs(fi - bfi);
       if (first_total[fi] <= cost)
       {
        bfi = fi; 
       }
      }
     }
     else
     {
      while ((fi-1>=pos)&&(fi>0))
      {
       --fi;
       
       float cost = first_total[bfi] + this->mult * fabs(fi - bfi);
       if (first_total[fi] <= cost)
       {
        bfi = fi; 
       }
      }
     }
    
    // Calculate and store the cost...
     float cost = this->mult * fabs(bfi - pos);
     if (cost>this->cap) cost = this->cap;
     cost += first_total[bfi];
     
     if (cost<second_out[si])
     {
      second_out[si] = cost;
      second_back[si] = bfi;
     }
   }
}

void Linear_costs_rev(PairCost this_ptr, int first_count, float * first_out, int * first_forward, int second_count, float * second_total)
{
 Linear * this = (Linear*)this_ptr;
 
 // Use the standard convex function trick - gets a little fiddly because we allow negatives in the scalar of label positions...
  // Forwards pass...
   int fi;
   int si = (this->scale>=0.0) ? 0 : (second_count-1);
   int bsi = si;
   float bsi_pos = bsi * this->scale + this->offset;
   for (fi=0; fi<first_count; fi++)
   {
    // Walk through the seconds until we have caught up, updating bsi, the index of the lowest cost entry...
     if (this->scale>=0.0)
     {
      while (si+1<second_count)
      {
       float pos = (si + 1) * this->scale + this->offset;
       if (pos>fi) break;
       
       ++si;
       
       float cost = second_total[bsi] + this->mult * fabs(pos - bsi_pos);
       if (second_total[si] <= cost)
       {
        bsi = si; 
        bsi_pos = pos;
       }
      }
     }
     else
     {
      while (si>0)
      {
       float pos = (si - 1) * this->scale + this->offset;
       if (pos<fi) break;
       
       --si;
       
       float cost = second_total[bsi] + this->mult * fabs(pos - bsi_pos);
       if (second_total[si] <= cost)
       {
        bsi = si; 
        bsi_pos = pos;
       }
      }
     }
    
    // Calculate and store the cost...
     float cost = this->mult * fabs(bsi_pos - fi);
     if (cost>this->cap) cost = this->cap;
     cost += second_total[bsi];
    
     first_out[fi] = cost;
     first_forward[fi] = bsi;
   }
 
  // Backwards pass...
   si = (this->scale>=0.0) ? (second_count-1) : 0;
   bsi = si;
   bsi_pos = bsi * this->scale + this->offset;
   for (fi=first_count-1; fi>=0; fi--)
   {
    // Walk through the seconds until we have caught up, updating bsi, the index of the lowest cost entry...
     if (this->scale<=0.0)
     {
      while (si+1<second_count)
      {
       float pos = (si + 1) * this->scale + this->offset;
       if (pos>fi) break;
       
       ++si;
       
       float cost = second_total[bsi] + this->mult * fabs(pos - bsi_pos);
       if (second_total[si] <= cost)
       {
        bsi = si; 
        bsi_pos = pos;
       }
      }
     }
     else
     {
      while (si>0)
      {
       float pos = (si - 1) * this->scale + this->offset;
       if (pos<fi) break;
       
       --si;
       
       float cost = second_total[bsi] + this->mult * fabs(pos - bsi_pos);
       if (second_total[si] <= cost)
       {
        bsi = si; 
        bsi_pos = pos;
       }
      }
     }
    
    // Calculate and store the cost...
     float cost = this->mult * fabs(bsi_pos - fi);
     if (cost>this->cap) cost = this->cap;
     cost += second_total[bsi];
     
     if (cost < first_out[fi])
     {
      first_out[fi] = cost;
      first_forward[fi] = bsi;
     }
   }
}


const PairCostType LinearType =
{
 "linear",
 "A falloff based cost function, based on the linear distance between labels. The first variable is at the position of its state index (0, 1, 2 etc), but the second is at (i * scale + offset), where i is the state index. The cost is then the absolute distance between state positions, multiplied by mult. There is also cap, which limits how how the cost can go. You initialise with an entity that can be interpreted as a numpy array, 1D, 4 entries: [mult = 1.0, offset = 0.0, scale = 1.0, cap = inf]. If its too short then the defaults just given are used. Note that mult and cap both have abs applied before use, so you can't have a dissimilarity-encouraging cost.",
 Linear_new,
 Linear_delete,
 Linear_cost,
 Linear_costs,
 Linear_costs_rev,
};



// Code for the ordered PairCost type...
typedef struct Ordered Ordered;

struct Ordered
{
 const PairCostType * type;
 
 int steps;
 float * step; // Cost of taking 0, 1 etc steps.
};


PairCost Ordered_new(PyObject * data)
{
 PyArrayObject * param = (PyArrayObject*)PyArray_FromObject(data, NPY_FLOAT, 1, 1);
 if (param==NULL) return NULL;
 
 if (PyArray_DIMS(param)[0]<2)
 {
  Py_DECREF(param);
  PyErr_SetString(PyExc_RuntimeError, "The ordered PairType requires at least two costs to make sense");
  return NULL; 
 }
 
 Ordered * this = (Ordered*)malloc(sizeof(Ordered));
 this->type = &OrderedType;
 
 this->steps = PyArray_DIMS(param)[0];
 this->step = (float*)malloc(this->steps * sizeof(float));
 int i;
 for (i=0; i<this->steps; i++)
 {
  this->step[i] = *(float*)PyArray_GETPTR1(param, i);
 }
 
 Py_DECREF(param);
 
 return this;
}

void Ordered_delete(PairCost this_ptr)
{
 Ordered * this = (Ordered*)this_ptr;
 free(this->step);
 free(this);
}

float Ordered_cost(PairCost this_ptr, int first, int second)
{
 Ordered * this = (Ordered*)this_ptr;
 
 int offset = second - first;
 
 if ((offset<0)||(offset>=this->steps)) return INFINITY;
 return this->step[offset];
}

void Ordered_costs(PairCost this_ptr, int first_count, float * first_total, int second_count, float * second_out, int * second_back)
{
 Ordered * this = (Ordered*)this_ptr;
 
 // Straight brute force whilst not bothering to iterate the infinities...
  int si;
  for (si=0; si<second_count; si++)
  {
   second_out[si] = INFINITY;
   second_back[si] = -1;
   
   int offset;
   for (offset=0; offset<this->steps; offset++)
   {
    int fi = si - offset;
    if ((fi>=0)&&(fi<first_count))
    {
     float cost = first_total[fi] + this->step[offset];
     if (cost<second_out[si])
     {
      second_out[si] = cost;
      second_back[si] = fi; 
     }
    }
   } 
  }
}

void Ordered_costs_rev(PairCost this_ptr, int first_count, float * first_out, int * first_forward, int second_count, float * second_total)
{
 Ordered * this = (Ordered*)this_ptr;
 
 // As above, but with the signs the other way...
  int fi;
  for (fi=0; fi<first_count; fi++)
  {
   first_out[fi] = INFINITY;
   first_forward[fi] = -1;
   
   int offset;
   for (offset=0; offset<this->steps; offset++)
   {
    int si = fi + offset;
    if (si<second_count)
    {
     float cost = second_total[si] + this->step[offset];
     if (cost<first_out[fi])
     {
      first_out[fi] = cost;
      first_forward[fi] = si;
     }
    }
   }
  }
}


const PairCostType OrderedType =
{
 "ordered",
 "For when you are labeling something that is strictly ordered, and the optimisation is where to put the cuts - the output is such that the state indices are increasing. Typically when requesting the best solution you ask for it fixed to use the last state of the last variable. Allows you to set the cost of putting the cut versus not cutting, and also includes the option of skipping a label entirly, with you setting the cost of doing so. Takes as input an object that can be interpreted as a 1D numpy array of floats, of at least length two. The first entry is the cost of staying on the same label, the second of moving to the next label. If included the third is the cost of skipping one label, and so on.",
 Ordered_new,
 Ordered_delete,
 Ordered_cost,
 Ordered_costs,
 Ordered_costs_rev,
};



// Code for the full PairCost type...
typedef struct Full Full;

struct Full
{
 const PairCostType * type;
 
 PyArrayObject * cost; // Indexed [fi, si].
};


PairCost Full_new(PyObject * data)
{
 PyArrayObject * param = (PyArrayObject*)PyArray_FromObject(data, NPY_FLOAT, 2, 2);
 if (param==NULL) return NULL;
 
 Full * this = (Full*)malloc(sizeof(Full));
 this->type = &FullType;
 
 this->cost = param;
 
 return this;
}

void Full_delete(PairCost this_ptr)
{
 Full * this = (Full*)this_ptr;
 Py_DECREF(this->cost);
 free(this);
}

float Full_cost(PairCost this_ptr, int first, int second)
{
 Full * this = (Full*)this_ptr;
 
 first = first % PyArray_DIMS(this->cost)[0];
 second = second % PyArray_DIMS(this->cost)[1];
 
 return *(float*)PyArray_GETPTR2(this->cost, first, second);
}

void Full_costs(PairCost this_ptr, int first_count, float * first_total, int second_count, float * second_out, int * second_back)
{
 Full * this = (Full*)this_ptr;
 
 int si;
 for (si=0; si<second_count; si++)
 {
  int smi = si % PyArray_DIMS(this->cost)[1];
  
  second_out[si] = INFINITY; 
  second_back[si] = -1;
  
  int fi;
  for (fi=0; fi<first_count; fi++)
  {
   int fmi = fi % PyArray_DIMS(this->cost)[0];
   float cost = first_total[fi] + *(float*)PyArray_GETPTR2(this->cost, fmi, smi);
   
   if (cost<second_out[si])
   {
    second_out[si] = cost;
    second_back[si] = fi;
   }
  }
 }
}

void Full_costs_rev(PairCost this_ptr, int first_count, float * first_out, int * first_forward, int second_count, float * second_total)
{
 Full * this = (Full*)this_ptr;

 int fi;
 for (fi=0; fi<first_count; fi++)
 {
  int fmi = fi % PyArray_DIMS(this->cost)[0];
  
  first_out[fi] = INFINITY;
  first_forward[fi] = -1;
  
  int si;
  for (si=0; si<second_count; si++)
  {
   int smi = si % PyArray_DIMS(this->cost)[1];
   float cost = second_total[si] + *(float*)PyArray_GETPTR2(this->cost, fmi, smi);
   
   if (cost<first_out[fi])
   {
    first_out[fi] = cost;
    first_forward[fi] = si;
   }
  }
 }
}


const PairCostType FullType =
{
 "full",
 "Simply uses a complete cost matrix, which is provided as its only parameter as an entity that can be interpreted as a 2D float32 numpy array indexed [first index, second index]. Note that this is O(#first * #second) as there are no tricks here. If the input is a numpy float32 array then it will keep a reference and use it directly - good for memory usage, but be aware that editting such a matrix afterwards will matter. If the matrix is too small in either dimension it will access modulus the size in that dimension - this can be used to repeat costs without the storage cost.",
 Full_new,
 Full_delete,
 Full_cost,
 Full_costs,
 Full_costs_rev,
};



// List of PairCost types so the system can identify and use them...
const PairCostType * ListPairCostType[] =
{
 &DifferentType,
 &LinearType,
 &OrderedType,
 &FullType,
 NULL
};



// Code for actual DDP object...
void DDP_new(DDP * this)
{
 this->variables = 0;
 this->count = NULL;
 
 this->offset = NULL;
 this->cost = NULL;
 
 this->pair_cost = NULL;
 
 this->state = 0;
 
 this->total = NULL;
 this->back = NULL;
 this->forward = NULL;
}

void DDP_dealloc(DDP * this)
{
 free(this->count);
  
 free(this->offset);
 free(this->cost);
 
 int i;
 for (i=0; i<this->variables-1; i++)
 {
  DeletePC(this->pair_cost[i]);
 }
 free(this->pair_cost);
 
 free(this->total);
 free(this->back);
}


static PyObject * DDP_new_py(PyTypeObject * type, PyObject * args, PyObject * kwds)
{
 // Allocate the object...
  DDP * self = (DDP*)type->tp_alloc(type, 0);

 // On success construct it...
  if (self!=NULL) DDP_new(self);

 // Return the new object...
  return (PyObject*)self;
}

static void DDP_dealloc_py(DDP * self)
{
 DDP_dealloc(self);
 self->ob_type->tp_free((PyObject*)self);
}



static PyObject * DDP_names_py(DDP * self, PyObject * args)
{
 // Create the return list...
  PyObject * ret = PyList_New(0);
  
 // Add each pair cost type in turn...
  int i = 0;
  while (ListPairCostType[i]!=NULL)
  {
   PyObject * name = PyString_FromString(ListPairCostType[i]->name);
   PyList_Append(ret, name);
   Py_DECREF(name);
   
   ++i; 
  }
 
 // Return...
  return ret;
}


static PyObject * DDP_description_py(DDP * self, PyObject * args)
{
 // Parse the parameters...
  char * name;
  if (!PyArg_ParseTuple(args, "s", &name)) return NULL;
  
 // Try and find the relevant entity - if found assign it and return...
  int i = 0;
  while (ListPairCostType[i]!=NULL)
  {
   if (strcmp(ListPairCostType[i]->name, name)==0)
   {
    return PyString_FromString(ListPairCostType[i]->description);
   }
   
   ++i; 
  }
 
 // Failed to find it - throw an error...
  PyErr_SetString(PyExc_TypeError, "unrecognised pair cost name");
  return NULL; 
}



void DDP_prepare_simple(DDP * this, int variables, int labels)
{
 int total = variables * labels;
 
 // Clean up previous pairwise terms...
  int i;
  for (i=0; i<this->variables-1; i++)
  {
   DeletePC(this->pair_cost[i]);
  }
  
 // Allocate all memory...
  this->variables = variables;
  this->count = (int*)realloc(this->count, variables * sizeof(int));
 
  this->offset = (int*)realloc(this->offset, variables * sizeof(int));
  this->cost = (float*)realloc(this->cost, total * sizeof(float));
  
  this->pair_cost = (PairCost*)realloc(this->pair_cost, (variables-1) * sizeof(PairCost));
  
  this->state = 0;
  
  this->total = (float*)realloc(this->total, total * sizeof(float));
  this->back = (int*)realloc(this->back, total * sizeof(int));
  this->forward = (int*)realloc(this->forward, total * sizeof(int));
 
 // Loop and set everything to sane starting values...
  for (i=0; i<variables; i++)
  {
   this->count[i] = labels;
   this->offset[i] = i * labels;
  }
  
  for (i=0; i<this->variables-1; i++)
  {
   this->pair_cost[i] = NULL;
  }
  
  for (i=0; i<total; i++)
  {
   this->cost[i] = 0.0;
   this->total[i] = 0.0;
   this->back[i] = 0;
   this->forward[i] = 0;
  }
}


void DDP_prepare_complex(DDP * this, PyArrayObject * labels)
{
 // Clean up previous pairwise terms...
  int i;
  for (i=0; i<this->variables-1; i++)
  {
   DeletePC(this->pair_cost[i]);
  }
  
 // Do the basic stuff, so we know how big to make the rest of the arrays... 
  this->variables = PyArray_DIMS(labels)[0];
  this->count = (int*)realloc(this->count, this->variables * sizeof(int));
  this->offset = (int*)realloc(this->offset, this->variables * sizeof(int));
 
  int total = 0;
  for (i=0; i<this->variables; i++)
  {
   this->count[i] = *(int*)PyArray_GETPTR1(labels, i);
   this->offset[i] = total;
   total += this->count[i];
  }
  
 // Make the rest of it...
  this->cost = (float*)realloc(this->cost, total * sizeof(float));
  
  this->pair_cost = (PairCost*)realloc(this->pair_cost, (this->variables-1) * sizeof(PairCost));
  
  this->state = 0;
  
  this->total = (float*)realloc(this->total, total * sizeof(float));
  this->back = (int*)realloc(this->back, total * sizeof(int));
  this->forward = (int*)realloc(this->forward, total * sizeof(int));
  
  for (i=0; i<this->variables-1; i++)
  {
   this->pair_cost[i] = NULL;
  }
  
  for (i=0; i<total; i++)
  {
   this->cost[i] = 0.0;
   this->total[i] = 0.0;
   this->back[i] = 0;
   this->forward[i] = 0;
  }
}


static PyObject * DDP_prepare_py(DDP * self, PyObject * args)
{
 // Extract the parameters and determine the scenario, checking the right stuff has been handed in...
  PyObject * param1;
  PyObject * param2 = NULL;
  if (!PyArg_ParseTuple(args, "O|O", &param1, &param2)) return NULL;
 
  if (param2!=NULL)
  {
   // Two integers - first is the number of random variables, second the number of labels per random variable...
    int vars = PyInt_AsLong(param1);
    int labels = PyInt_AsLong(param2);
    if ((vars<1)||(labels<1))
    {
     PyErr_SetString(PyExc_ValueError, "random variable or label count not recognised");
     return NULL;
    }
    
    DDP_prepare_simple(self, vars, labels);
  }
  else
  {
   // One array, containing the number of labels per random variable...
    PyArrayObject * labels = (PyArrayObject*)PyArray_FromObject(param1, NPY_INT, 1, 1);
    if (labels==NULL) return NULL;
    
    
    DDP_prepare_complex(self, labels);
    
    Py_DECREF(labels);
  }
  
 // Return None...
  Py_INCREF(Py_None);
  return Py_None;
}



static PyObject * DDP_unary_py(DDP * self, PyObject * args)
{
 // Handle the parameters...
  int offset;
  PyObject * values;
  if (!PyArg_ParseTuple(args, "iO", &offset, &values)) return NULL;

  if ((offset<0)||(offset>=self->variables))
  {
   PyErr_SetString(PyExc_KeyError, "offset out of range");
   return NULL; 
  }
  
  PyArrayObject * costs = (PyArrayObject*)PyArray_FromObject(values, NPY_FLOAT, 1, 2);
  if (costs==NULL) return NULL;
  
  self->state = 0;
  
 // Load in the data, the method depending on its dimensionality... 
  if (PyArray_NDIM(costs)==1)
  {
   // 1 dimensional...
    offset = self->offset[offset]; // One token, 3 meanings in one line. I should be shot.
    int read = PyArray_DIMS(costs)[0];
    int total = self->offset[self->variables-1] + self->count[self->variables-1];
    if (offset+read>total)
    {
     read = total - offset;
    }
    
    int i;
    for (i=0; i<read; i++)
    {
     self->cost[offset+i] = *(float*)PyArray_GETPTR1(costs, i);
    }
  }
  else
  {
   // 2 dimensional...
    int read_vars = PyArray_DIMS(costs)[0];
    if (offset+read_vars>self->variables)
    {
     read_vars = self->variables - offset; 
    }
    
    int i, j;
    for (i=0; i<read_vars; i++)
    {
     int v = offset + i;
     
     int read_labels = PyArray_DIMS(costs)[1];
     if (read_labels>self->count[v])
     {
      read_labels = self->count[v];  
     }
     
     for (j=0; j<read_labels; j++)
     {
      self->cost[self->offset[v] + j] = *(float*)PyArray_GETPTR2(costs, i, j);
     }
    }
  }
  
 // Clean up...
  Py_DECREF(costs);
  
 // Return None...
  Py_INCREF(Py_None);
  return Py_None;
}



static PyObject * DDP_pairwise_py(DDP * self, PyObject * args)
{
 // Handle the parameters...
  int offset;
  PyObject * names;
  PyObject * data;
  if (!PyArg_ParseTuple(args, "iOO", &offset, &names, &data)) return NULL;

  if ((offset<0)||(offset>=self->variables-1))
  {
   PyErr_SetString(PyExc_KeyError, "offset out of range");
   return NULL; 
  }
  
  self->state = 0;
  
 // Detect if we are setting one entry or multiple entries...
  if (PyString_Check(names)!=0)
  {
   // Single item - fairly simple...
    DeletePC(self->pair_cost[offset]);
    self->pair_cost[offset] = NULL;
    
    const char * name = PyString_AsString(names);
    if (name[0]==0) // Allow a zero length string to be used to set NULL, as strange as that may be.
    {
     Py_INCREF(Py_None);
     return Py_None; 
    }
    
    int i = 0;
    while (ListPairCostType[i]!=NULL)
    {
     if (strcmp(ListPairCostType[i]->name, name)==0)
     {
      // Call through to the constructor...
       self->pair_cost[offset] = ListPairCostType[i]->new(data);
       if (self->pair_cost[offset]==NULL)
       {
        PyErr_SetString(PyExc_RuntimeError, "could not construct pair cost");
        return NULL; 
       }
       
      // Return None...
       Py_INCREF(Py_None);
       return Py_None;
     }
     
     ++i;
    }
    
    PyErr_SetString(PyExc_KeyError, "unrecognised pair cost name");
    return NULL; 
  }
  else
  {
   // Verify it is a list...
    if (PyList_Check(names)==0)
    {
     PyErr_SetString(PyExc_ValueError, "name parameter could not be interpreted as either a string or a list");
    }
    
   // Get its size and iterate the items...
    int read = PyList_Size(names);
    
    if (offset+read>self->variables-1)
    {
     read = self->variables-1-offset; 
    }
    
    int i;
    for (i=0; i<read; i++)
    {
     PyObject * n = PyList_GetItem(names, i);
     int dec_d = 0;
     PyObject * d = PyList_GetItem(data, i);
     
     if (d==NULL)
     {
      PyErr_Clear(); // PyListGetItem set an error - clear it!
      // Try the more general purpose request - much slower hence the screwing around...
       PyObject * ii = PyInt_FromLong(i);
       d = PyObject_GetItem(data, ii);
       Py_DECREF(ii);
       if (d!=NULL) dec_d = 1;
     }
     
     if ((n==NULL)||(d==NULL))
     {
      if (dec_d!=0) Py_DECREF(d);
      PyErr_SetString(PyExc_ValueError, "could not interprete input names or data as a list.");
      return NULL;  
     }
     
     int pc = offset + i;
     DeletePC(self->pair_cost[pc]);
     self->pair_cost[pc] = NULL;
     
     const char * name = PyString_AsString(n);
     if (name==NULL)
     {
      if (dec_d!=0) Py_DECREF(d);
      PyErr_SetString(PyExc_ValueError, "could not interprete a name as a string");
      return NULL;  
     }
     
     int ok = 0;
     int j = 0;
     while (ListPairCostType[j]!=NULL)
     {
      if (strcmp(ListPairCostType[j]->name, name)==0)
      {
       ok = 1;
       
       self->pair_cost[pc] = ListPairCostType[j]->new(d);
       if (self->pair_cost[pc]==NULL)
       {
        if (dec_d!=0) Py_DECREF(d);
        PyErr_SetString(PyExc_RuntimeError, "could not construct pair cost");
        return NULL; 
       }
      }
      
      ++j;
     }
     
     if (dec_d!=0) Py_DECREF(d);
     
     if ((ok==0)&&(strlen(name)!=0))
     {
      PyErr_SetString(PyExc_KeyError, "unrecognised pair cost name");
      return NULL;  
     }
    }
    
   // Return None...
    Py_INCREF(Py_None);
    return Py_None;
  }
}



void DDP_solve(DDP * this)
{
 if (this->state>0) return; // Already been run - do nothing.
 int i;
 
 // Initalise the totals and backwards pointers...
  for (i=0; i<this->count[0]; i++)
  {
   this->total[this->offset[0] + i] = this->cost[this->offset[0] + i];
   this->back[this->offset[0] + i] = -1;
  }
  
 // Loop and pass the messages... 
  for (i=0; i<this->variables-1; i++)
  {
   if (this->pair_cost[i]!=NULL)
   {
    CostsPC(this->pair_cost[i], this->count[i], this->total + this->offset[i], this->count[i+1], this->total + this->offset[i+1], this->back + this->offset[i+1]);
   
    int j;
    for (j=0; j<this->count[i+1]; j++)
    {
     this->total[this->offset[i+1] + j] += this->cost[this->offset[i+1] + j];
    }
   }
   else
   {
    // Link broken - set totals to unary cost as its effectivly a new problem...
     int j;
     for (j=0; j<this->count[i+1]; j++)
     {
      this->total[this->offset[i+1] + j] = this->cost[this->offset[i+1] + j];
      this->back[this->offset[i+1] + j] = -1;
     }
   }
  }
  
 // Set the state acordingly...
  this->state = 1;
}


static PyObject * DDP_solve_py(DDP * self, PyObject * args)
{
 DDP_solve(self);
 
 Py_INCREF(Py_None);
 return Py_None;
}



void DDP_backpass(DDP * this)
{
 if (this->state>1) return; // Already been run - do nothing.
 if (this->state==0) DDP_solve(this); // Need to solve first.
 int i;

 // Setup the state - need some temporary storage...
  int max_count = 1;
  for (i=0; i<this->variables; i++)
  {
   if (this->count[i]>max_count)
   {
    max_count = this->count[i];
   }
  }
  
  float * first = (float*)malloc(max_count * sizeof(float));
  float * second = (float*)malloc(max_count * sizeof(float));
  
  for (i=0; i<max_count; i++) first[i] = 0.0;
  
  for (i=0; i<this->count[this->variables-1]; i++)
  {
   this->forward[this->offset[this->variables-1] + i] = -1; 
  }
 
 // Do the backwards pass...
  for (i=this->variables-2; i>=0; i--) // i is the index of first.
  {
   // Swap the pointers ready for this dance...
    float * temp = first;
    first = second;
    second = temp;
    
   // Sum in the current nodes costs...
    int j;
    for (j=0; j<this->count[i+1]; j++)
    {
     second[j] += this->cost[this->offset[i+1] + j];
    }
    
   // Factor in the pairwise cost...
    if (this->pair_cost[i]!=NULL)
    {
     CostsRevPC(this->pair_cost[i], this->count[i], first, this->forward + this->offset[i], this->count[i+1], second);
    
     // Sum in the cost to the total on the first node...
      for (j=0; j<this->count[i]; j++)
      {
       this->total[this->offset[i] + j] += first[j];
      }
    }
    else
    {
     // Link broken - reset first...
      for (j=0; j<this->count[i]; j++)
      {
       first[j] = 0.0;
       this->forward[this->offset[i] + j] = -1;
      }
    }
  }
 
 // Clean up the temporary storage...
  free(second);
  free(first);
 
 // Set the state acordingly...
  this->state = 2;
}


static PyObject * DDP_backpass_py(DDP * self, PyObject * args)
{
 DDP_backpass(self);
 
 Py_INCREF(Py_None);
 return Py_None;
}



static PyObject * DDP_best_py(DDP * self, PyObject * args)
{
 // See if we have any parameters...
  int variable = -1;
  int state = -1;
  if (!PyArg_ParseTuple(args, "|ii", &variable, &state)) return NULL;
  
 // Make sure the forward pass has been done...
  DDP_solve(self);
  
 // Convert to a state where we have parameters, possibly throwing an error if we don't like our input (this technically involves part of the optimisation!)...
  if (variable<0)
  {
   variable = self->variables - 1;
   state = 0;
   
   int i;
   for (i=1; i<self->count[variable]; i++)
   {
    int off = self->offset[variable];
    if (self->total[off + state] > self->total[off + i])
    {
     state = i; 
    }
   }
  }
  else
  {
   if (state<0)
   {
    state = variable;
    variable = self->variables - 1;
    
    if (state>=self->count[variable])
    {
     PyErr_SetString(PyExc_KeyError, "state of last random variable out of range");
     return NULL;
    }
   }
   else
   {
    if (variable>=self->variables)
    {
     PyErr_SetString(PyExc_KeyError, "random variable index out of range");
     return NULL;
    }
    
    if (state>=self->count[variable])
    {
     PyErr_SetString(PyExc_KeyError, "state of random variable out of range");
     return NULL;
    }
   }
  }
  
 // Do the backpass if needed..
  if (variable<self->variables-1) DDP_backpass(self);

 // Create the output...
  float cost = self->total[self->offset[variable] + state];
  npy_intp dims = self->variables;
  PyArrayObject * map = (PyArrayObject*)PyArray_SimpleNew(1, &dims, NPY_INT32);
  
  *(int*)PyArray_GETPTR1(map, variable) = state;
  
 // Do passes in both directions as required...
  // Backwards...
   int targ;
   for (targ = variable-1; targ>=0; targ--)
   {
    int prev = *(int*)PyArray_GETPTR1(map, targ+1);
    int cur = self->back[self->offset[targ+1] + prev];
    
    if (cur<0)
    {
     cur = 0;
     
     int i;
     for (i=1; i<self->count[targ]; i++)
     {
      if (self->total[self->offset[targ] + cur] > self->total[self->offset[targ] + i])
      {
       cur = i; 
      }
     }
     
     cost += self->total[self->offset[targ] + cur];
    }
    
    *(int*)PyArray_GETPTR1(map, targ) = cur;
   }
   
  // Forwards...
   for (targ = variable+1; targ<self->variables; targ++)
   {
    int prev = *(int*)PyArray_GETPTR1(map, targ-1);
    int cur = self->forward[self->offset[targ-1] + prev];
    
    if (cur<0)
    {
     cur = 0;
     
     int i;
     for (i=1; i<self->count[targ]; i++)
     {
      if (self->total[self->offset[targ] + cur] > self->total[self->offset[targ] + i])
      {
       cur = i; 
      }
     }
     
     cost += self->total[self->offset[targ] + cur];
    }
    
    *(int*)PyArray_GETPTR1(map, targ) = cur;
   }
  
 // Return the required tuple...
  return Py_BuildValue("(N,f)", map, cost);
}



static PyObject * DDP_costs_py(DDP * self, PyObject * args)
{
 // Get the single parameter...
  int index;
  if (!PyArg_ParseTuple(args, "i", &index)) return NULL;

  if ((index<0)||(index>=self->variables))
  {
   PyErr_SetString(PyExc_KeyError, "index out of range");
   return NULL; 
  }
  
 // Verify that sufficient computation has occured for us to know the answer to the query...
  if (index+1==self->variables) DDP_solve(self);
                           else DDP_backpass(self);
  
 // Create and fill the return object...
  npy_intp dims = self->count[index];
  PyArrayObject * ret = (PyArrayObject*)PyArray_SimpleNew(1, &dims, NPY_FLOAT32);
  
  int i;
  for (i=0; i<self->count[index]; i++)
  {
   *(float*)PyArray_GETPTR1(ret, i) = self->total[self->offset[index] + i];
  }
 
 // Return...
  return (PyObject*)ret;
}



// All the python interface stuff for DDP...
static PyMemberDef DDP_members[] =
{
 {"variables", T_INT, offsetof(DDP, variables), 0, "Number of random variables in the dynamic programming object."},
 {NULL}
};



static PyMethodDef DDP_methods[] =
{
 {"names", (PyCFunction)DDP_names_py, METH_NOARGS | METH_STATIC, "A static method that returns a list of the pair cost types, as strings (names) that can be used to instanciate them."},
 {"description", (PyCFunction)DDP_description_py, METH_VARARGS | METH_STATIC, "A static method that is given a pair cost type name and then returns a string describing it. Returns None if its not recognised."},
 
 {"prepare", (PyCFunction)DDP_prepare_py, METH_VARARGS, "Prepares the object - must be called before you do anything. Wipes out any unary or pairwise potentials that have been set (All unary values are set to 0, all pairwise terms are cut). You provide either two integers as parameters - number of variables followed by number of labels per variable, or one input, which is interpreted as a 1D numpy array of integers: the number of labels per random variable, with the length of the array being the number of random variables."},
 {"unary", (PyCFunction)DDP_unary_py, METH_VARARGS, "Allows you to set the costs (negative log likelihoods) of the unary term for each random variable. Takes two parameters - the first an offset into the random variables, the second something that can be interpreted as a numpy array. If the array is 1D then it effectivly eats every value within until the end of the array, starting at the first label of the indexed random variable and overflowing into further random variables - this can be very useful if the random variables have variable label counts, as you can pack them densely into the provided array. If the array is 2D then it interprets the first dimension as indexing a random variable, added to the offset, and the second a label for that random variable. In both cases limits are respected, and either the input not used or the unary cost left at whatever value it was previously (It defaults to 0)."},
 {"pairwise", (PyCFunction)DDP_pairwise_py, METH_VARARGS, "Allows you to set the pairwise terms - this gets complicated as the system uses a modular system for deciding the cost of label pairs - see the names and description method to find out about the modules avaliable (The provided info.py script prints this all out). Parameters are (offset, name, data): offset - the random variable to offset to - it is the index of then first one, so the cost is between offset and offset+1; name is the name of the cost module system to invoke - if its a single name then we are setting a single pairwise term, but if its a list of them then we are setting multiple costs, starting from offset (['name'] * count is your friend!); data is the data required - this depends on the pairwise cost module being invoked. If a single name is provided it is passed straight through, but if a list of names is provided it is interpreted as a list and the relevent entry passed through for each initialisation. Be warned that it may keep views to the input rather than copies, so its generally best to not edit any data passed in afterwards. This can get quite clever - it will happily handle a data matrix for instance. Be warned that this methods modular nature forces it to be quite intensive - it can be relativly slow. Note that the default state of a pairwise term, which can be set by passing in a zero length string as a name, is to have no link between the adjacent terms - using this you can store multiple independent dynamic programming problems in a single object. I have no idea why you might want to do this, but it seemed like a reasonable default."},
 
 {"solve", (PyCFunction)DDP_solve_py, METH_NOARGS, "Solves the problem, such that you can extract the MAP solution. Note that this gets called automatically by the map method if it has not been run, so you can ignore this if you want."},
 {"backpass", (PyCFunction)DDP_backpass_py, METH_NOARGS, "After solving the problem this does the reverse pass, such that you have pointers in both directions for all random variables - this allows you to find the best solution and its cost, under the constraint that a single random variable is set to a given state. Automatically runs solve if it has not already been run."},
 
 {"best", (PyCFunction)DDP_best_py, METH_VARARGS, "Returns (map solution, cost). The map solution is an array indexed by random variable that gives the state the random variable should be in to obtain the minimum cost state - cost is that minimum cost. You can optionally pass in two indices - the first an index to a random variable, the second its state. In this case it returns the optimal solution under the constraint that the given random variable is set accordingly. If solve has not been run it is run automatically. In the case of constrained solutions for any variable except the last it requires that backpass has been run - it will again automatically do this if it has not. If you only give it one parameter it assumes you mean the last variable with that state."},
 {"costs", (PyCFunction)DDP_costs_py, METH_VARARGS, "Given the index of a random variable returns an array indexed by the state of the random variable, that gives the minimum cost solution when the random variable is set to the given state. If this is called without solve and backpass (for any random variable except the last) having been called it will automatically call them."},
 
 {NULL}
};



static PyTypeObject DDPType =
{
 PyObject_HEAD_INIT(NULL)
 0,                                /*ob_size*/
 "ddp_c.DDP",                      /*tp_name*/
 sizeof(DDP),                      /*tp_basicsize*/
 0,                                /*tp_itemsize*/
 (destructor)DDP_dealloc_py,       /*tp_dealloc*/
 0,                                /*tp_print*/
 0,                                /*tp_getattr*/
 0,                                /*tp_setattr*/
 0,                                /*tp_compare*/
 0,                                /*tp_repr*/
 0,                                /*tp_as_number*/
 0,                                /*tp_as_sequence*/
 0,                                /*tp_as_mapping*/
 0,                                /*tp_hash */
 0,                                /*tp_call*/
 0,                                /*tp_str*/
 0,                                /*tp_getattro*/
 0,                                /*tp_setattro*/
 0,                                /*tp_as_buffer*/
 Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
 "An object for performing discrete dynamic programming - a fairly basic algorithm really. Supports multiple cost function forms however. Constructor takes no parameters; instead you call the prepare(...) method.", /* tp_doc */
 0,                                /* tp_traverse */
 0,                                /* tp_clear */
 0,                                /* tp_richcompare */
 0,                                /* tp_weaklistoffset */
 0,                                /* tp_iter */
 0,                                /* tp_iternext */
 DDP_methods,                      /* tp_methods */
 DDP_members,                      /* tp_members */
 0,                                /* tp_getset */
 0,                                /* tp_base */
 0,                                /* tp_dict */
 0,                                /* tp_descr_get */
 0,                                /* tp_descr_set */
 0,                                /* tp_dictoffset */
 0,                                /* tp_init */
 0,                                /* tp_alloc */
 DDP_new_py,                       /* tp_new */
};



// Module code...
static PyMethodDef ddp_c_methods[] =
{
 {NULL}
};



#ifndef PyMODINIT_FUNC
#define PyMODINIT_FUNC void
#endif

PyMODINIT_FUNC initddp_c(void)
{
 PyObject * mod = Py_InitModule3("ddp_c", ddp_c_methods, "Provides a fairly standard discrete dynamic programming implimentation, with good cost function specification.");
 
 import_array();
 
 if (PyType_Ready(&DDPType) < 0) return;
 
 Py_INCREF(&DDPType);
 PyModule_AddObject(mod, "DDP", (PyObject*)&DDPType);
}
