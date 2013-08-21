// Copyright 2013 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



#include "data_matrix.h"

#include <stdlib.h>



float ToFloat_zero(void * data)
{
 return 0.0; 
}

float ToFloat_char(void * data)
{
 return *(char*)data; 
}

float ToFloat_short(void * data)
{
 return *(short*)data; 
}

float ToFloat_int(void * data)
{
 return *(int*)data; 
}

float ToFloat_long_long(void * data)
{
 return *(long long*)data; 
}

float ToFloat_unsigned_char(void * data)
{
 return *(unsigned char*)data; 
}

float ToFloat_unsigned_short(void * data)
{
 return *(unsigned short*)data; 
}

float ToFloat_unsigned_int(void * data)
{
 return *(unsigned int*)data; 
}

float ToFloat_unsigned_long_long(void * data)
{
 return *(unsigned long long*)data; 
}

float ToFloat_float(void * data)
{
 return *(float*)data; 
}

float ToFloat_double(void * data)
{
 return *(double*)data; 
}

float ToFloat_long_double(void * data)
{
 return *(long double*)data; 
}


ToFloat KindToFunc(const PyArray_Descr * descr)
{
 switch (descr->kind)
 {
  case 'b': // Boolean
  return ToFloat_char;
     
  case 'i': // Signed integer
   switch (descr->elsize)
   {
    case sizeof(char):      return ToFloat_char;
    case sizeof(short):     return ToFloat_short;
    case sizeof(int):       return ToFloat_int;
    case sizeof(long long): return ToFloat_long_long;
   }
  break;
     
  case 'u': // Unsigned integer
   switch (descr->elsize)
   {
    case sizeof(unsigned char):      return ToFloat_unsigned_char;
    case sizeof(unsigned short):     return ToFloat_unsigned_short;
    case sizeof(unsigned int):       return ToFloat_unsigned_int;
    case sizeof(unsigned long long): return ToFloat_unsigned_long_long;
   }
  break;
     
  case 'f': // Floating point
   switch (descr->elsize)
   {
    case sizeof(float):       return ToFloat_float;
    case sizeof(double):      return ToFloat_double;
    case sizeof(long double): return ToFloat_long_double;
   }
  break;
 }
  
 return ToFloat_zero;
}



void DataMatrix_init(DataMatrix * dm)
{
 // Not 100% sure this is needed, as this module is used by a module that iwll inevitably call it, but just incase and to make the compilter shut up about me not using it...
  import_array();
  
 // Initalise everything to nulls etc...
  dm->array = NULL;
  dm->dt = NULL;
  dm->weight_index = -1;
  dm->weight_scale = 1.0;
  dm->exemplars = 0;
  dm->feats = 0;
  dm->dual_feats = 0;
  dm->mult = NULL;
  dm->fv = NULL;
  dm->feat_dims = 0;
  dm->feat_indices = NULL;
  dm->to_float = ToFloat_zero;
}


void DataMatrix_deinit(DataMatrix * dm)
{
 Py_XDECREF(dm->array);
 dm->array = NULL;
 
 free(dm->dt);
 dm->dt = NULL;
 
 dm->exemplars = 0;
 dm->feats = 0;
 dm->dual_feats = 0;
 
 free(dm->mult);
 dm->mult = NULL;
 
 free(dm->fv);
 dm->fv = NULL;
 
 dm->feat_dims = 0;
 free(dm->feat_indices);
 dm->feat_indices = NULL;

 dm->to_float = ToFloat_zero;
}



void DataMatrix_set(DataMatrix * dm, PyArrayObject * array, DimType * dt, int weight_index)
{
 int i;
 
 // Clean up any previous state...
  DataMatrix_deinit(dm);
  if (array==NULL) return;
  
 // Store the data matrix and its dimension types...
  dm->array = array;
  Py_INCREF(dm->array);
  
  dm->dt = (DimType*)malloc(dm->array->nd * sizeof(DimType));
  for (i=0; i<dm->array->nd; i++) dm->dt[i] = dt[i];
  
  dm->weight_index = weight_index;
  
 // Calculate the number of exemplars and features...
  dm->exemplars = 1;
  dm->feats = 1;
  
  for (i=0; i<dm->array->nd; i++)
  {
   switch(dm->dt[i])
   {
    case DIM_DATA:
     dm->exemplars *= dm->array->dimensions[i];
    break;
     
    case DIM_DUAL:
     dm->exemplars *= dm->array->dimensions[i];
     dm->dual_feats += 1;
    break;
     
    case DIM_FEATURE:
     dm->feats *= dm->array->dimensions[i];
     dm->feat_dims += 1;
    break;
   }
  }
  
  dm->feats += dm->dual_feats;
  
  if ((dm->weight_index>=0)&&(dm->weight_index<dm->feats))
  {
   dm->feats -= 1; 
  }
  else dm->weight_index = -1;
  
 // Create the scale and feature vector internal storage...
  dm->mult = (float*)malloc(dm->feats * sizeof(float));
  for (i=0; i<dm->feats; i++) dm->mult[i] = 1.0;
  
  dm->fv = (float*)malloc(dm->feats * sizeof(float));
  
 // Create the index of all feature indices...
  dm->feat_indices = (int*)malloc(dm->feat_dims * sizeof(int));
  int os = 0;
  for (i=0; i<dm->array->nd; i++)
  {
   if (dm->dt[i]==DIM_FEATURE)
   {
    dm->feat_indices[os] = i;
    os += 1;
   }
  }
  
 // Store a function pointer in the to_float variable that matches this array type...
  dm->to_float = KindToFunc(dm->array->descr);
}



int DataMatrix_exemplars(DataMatrix * dm)
{
 return dm->exemplars;  
}


int DataMatrix_features(DataMatrix * dm)
{
 return dm->feats; 
}



void DataMatrix_set_scale(DataMatrix * dm, float * scale, float weight_scale)
{
 int i;
 for (i=0; i<dm->feats; i++) dm->mult[i] = scale[i];
 dm->weight_scale = weight_scale;
}



float * DataMatrix_fv(DataMatrix * dm, int index, float * weight)
{
 int i;
 char * base = dm->array->data;
 int next_dual_feat = dm->dual_feats-1;
   
 if (weight!=NULL) *weight = dm->weight_scale;
 
 // Indexing pass - go backwards through the dimensions, updating the index as we go and figuring out the base index for the output; do the dual dimensions as we go...
  for (i=dm->array->nd-1; i>=0; i--)
  {
   npy_intp size = dm->array->dimensions[i];
   npy_intp step = index % size;
   
   if (dm->dt[i]==DIM_DUAL)
   {
    if (dm->weight_index==next_dual_feat)
    {
     if (weight!=NULL) *weight *= step;
    }
    else
    {
     dm->fv[next_dual_feat] = step * dm->mult[next_dual_feat];
    }
    next_dual_feat -= 1;
   }
   
   if (dm->dt[i]!=DIM_FEATURE)
   {
    base += dm->array->strides[i] * step;
    index /= size;
   }
  }
  
 // If the weight index is for a dual feature handle it we need to offset the positions...
  if ((dm->weight_index>=0)&&(dm->weight_index<dm->dual_feats))
  {
   for (i=dm->weight_index; i<dm->dual_feats-1; i++)
   {
    dm->fv[i] = dm->fv[i+1];
   }
  }
 
 // Second pass - iterate and do each feature in turn...
  int count = dm->feats - dm->dual_feats;
  int oi = dm->dual_feats;
  if (dm->weight_index>=0)
  {
   count += 1;
   if (dm->weight_index<dm->dual_feats) oi -= 1;
  }
  
  for (i=0; i<count; i++)
  {
   // Calculate the offset of this data item...
    char * pos = base;
    
    int j;
    int fi = i;
    for (j=dm->feat_dims-1; j>=0; j--)
    {
     int di = dm->feat_indices[j];
     npy_intp size = dm->array->dimensions[di];
     pos += dm->array->strides[di] * (fi % size);
     fi /= size;
    }
   
   // Translate the various possibilities - the use of a function pointer makes this relativly efficient...
    if (i!=(dm->weight_index-dm->dual_feats))
    {
     dm->fv[oi] = dm->to_float(pos) * dm->mult[oi];
     oi += 1;
    }
    else
    {
     if (weight!=NULL) *weight *= dm->to_float(pos);
    }
  }
  
 // Return the internal storage we have written the information into...
  return dm->fv;
}
