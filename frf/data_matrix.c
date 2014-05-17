// Copyright 2014 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



#include "data_matrix.h"

#include <stdlib.h>



NumberType KindToType(const PyArray_Descr * descr)
{
 if (descr->kind=='f') return CONTINUOUS;
                  else return DISCRETE;
}



int ToDiscrete_zero(void * data)
{
 return 0; 
}

int ToDiscrete_char(void * data)
{
 return *(char*)data; 
}

int ToDiscrete_short(void * data)
{
 return *(short*)data; 
}

int ToDiscrete_int(void * data)
{
 return *(int*)data; 
}

int ToDiscrete_long_long(void * data)
{
 return *(long long*)data; 
}

int ToDiscrete_unsigned_char(void * data)
{
 return *(unsigned char*)data; 
}

int ToDiscrete_unsigned_short(void * data)
{
 return *(unsigned short*)data; 
}

int ToDiscrete_unsigned_int(void * data)
{
 return *(unsigned int*)data; 
}

int ToDiscrete_unsigned_long_long(void * data)
{
 return *(unsigned long long*)data; 
}

int ToDiscrete_float(void * data)
{
 return *(float*)data; 
}

int ToDiscrete_double(void * data)
{
 return *(double*)data; 
}

int ToDiscrete_long_double(void * data)
{
 return *(long double*)data; 
}



ToDiscrete KindToDiscreteFunc(const PyArray_Descr * descr)
{
 switch (descr->kind)
 {
  case 'b': // Boolean
  return ToDiscrete_char;
     
  case 'i': // Signed integer
   switch (descr->elsize)
   {
    case sizeof(char):      return ToDiscrete_char;
    case sizeof(short):     return ToDiscrete_short;
    case sizeof(int):       return ToDiscrete_int;
    case sizeof(long long): return ToDiscrete_long_long;
   }
  break;
     
  case 'u': // Unsigned integer
   switch (descr->elsize)
   {
    case sizeof(unsigned char):      return ToDiscrete_unsigned_char;
    case sizeof(unsigned short):     return ToDiscrete_unsigned_short;
    case sizeof(unsigned int):       return ToDiscrete_unsigned_int;
    case sizeof(unsigned long long): return ToDiscrete_unsigned_long_long;
   }
  break;
     
  case 'f': // Floating point
   switch (descr->elsize)
   {
    case sizeof(float):       return ToDiscrete_float;
    case sizeof(double):      return ToDiscrete_double;
    case sizeof(long double): return ToDiscrete_long_double;
   }
  break;
 }
  
 return ToDiscrete_zero; 
}



float ToContinuous_zero(void * data)
{
 return 0.0; 
}

float ToContinuous_char(void * data)
{
 return *(char*)data; 
}

float ToContinuous_short(void * data)
{
 return *(short*)data; 
}

float ToContinuous_int(void * data)
{
 return *(int*)data; 
}

float ToContinuous_long_long(void * data)
{
 return *(long long*)data; 
}

float ToContinuous_unsigned_char(void * data)
{
 return *(unsigned char*)data; 
}

float ToContinuous_unsigned_short(void * data)
{
 return *(unsigned short*)data; 
}

float ToContinuous_unsigned_int(void * data)
{
 return *(unsigned int*)data; 
}

float ToContinuous_unsigned_long_long(void * data)
{
 return *(unsigned long long*)data; 
}

float ToContinuous_float(void * data)
{
 return *(float*)data; 
}

float ToContinuous_double(void * data)
{
 return *(double*)data; 
}

float ToContinuous_long_double(void * data)
{
 return *(long double*)data; 
}



ToContinuous KindToContinuousFunc(const PyArray_Descr * descr)
{
 switch (descr->kind)
 {
  case 'b': // Boolean
  return ToContinuous_char;
     
  case 'i': // Signed integer
   switch (descr->elsize)
   {
    case sizeof(char):      return ToContinuous_char;
    case sizeof(short):     return ToContinuous_short;
    case sizeof(int):       return ToContinuous_int;
    case sizeof(long long): return ToContinuous_long_long;
   }
  break;
     
  case 'u': // Unsigned integer
   switch (descr->elsize)
   {
    case sizeof(unsigned char):      return ToContinuous_unsigned_char;
    case sizeof(unsigned short):     return ToContinuous_unsigned_short;
    case sizeof(unsigned int):       return ToContinuous_unsigned_int;
    case sizeof(unsigned long long): return ToContinuous_unsigned_long_long;
   }
  break;
     
  case 'f': // Floating point
   switch (descr->elsize)
   {
    case sizeof(float):       return ToContinuous_float;
    case sizeof(double):      return ToContinuous_double;
    case sizeof(long double): return ToContinuous_long_double;
   }
  break;
 }
  
 return ToContinuous_zero; 
}



void FeatureBlock_init(FeatureBlock * this, PyArrayObject * array)
{
 this->offset = 0;
 this->features = (PyArray_NDIM(array)>1) ? PyArray_DIMS(array)[1] : 1;
 
 this->array = array;
 Py_INCREF(this->array);
 
 this->type = KindToType(PyArray_DESCR(array));
 this->discrete = KindToDiscreteFunc(PyArray_DESCR(array));
 this->continuous = KindToContinuousFunc(PyArray_DESCR(array));
}

void FeatureBlock_deinit(FeatureBlock * this)
{
 Py_DECREF(this->array);
 this->array = NULL;
}

int FeatureBlock_GetDiscrete(FeatureBlock * this, int exemplar, int feature)
{
 switch (PyArray_NDIM(this->array))
 {
  case 0: return 0;
  
  case 1:
  {
   exemplar = exemplar % PyArray_DIMS(this->array)[0];
   
   char * ptr = PyArray_DATA(this->array);
   ptr += PyArray_STRIDES(this->array)[0] * exemplar;
   
   return this->discrete(ptr);
  }
  
  default:
  {
   exemplar = exemplar % PyArray_DIMS(this->array)[0];
   
   char * ptr = PyArray_DATA(this->array);
   ptr += PyArray_STRIDES(this->array)[0] * exemplar;
   ptr += PyArray_STRIDES(this->array)[1] * feature;
   
   return this->discrete(ptr);
  } 
 }
 
 return 0;
}

float FeatureBlock_GetContinuous(FeatureBlock * this, int exemplar, int feature)
{
 switch (PyArray_NDIM(this->array))
 {
  case 0: return 0.0;
  
  case 1:
  {
   exemplar = exemplar % PyArray_DIMS(this->array)[0];
   
   char * ptr = PyArray_DATA(this->array);
   ptr += PyArray_STRIDES(this->array)[0] * exemplar;
   
   return this->continuous(ptr);
  }
  
  default:
  {
   exemplar = exemplar % PyArray_DIMS(this->array)[0];
   
   char * ptr = PyArray_DATA(this->array);
   ptr += PyArray_STRIDES(this->array)[0] * exemplar;
   ptr += PyArray_STRIDES(this->array)[1] * feature;
   
   return this->continuous(ptr);
  } 
 }
 
 return 0.0;
}



DataMatrix * DataMatrix_new(PyObject * obj, int * max)
{
 // Check that the provided object is one we can play with - it must either be a numpy array or a list/tuple etc. of them...
  int fbc = 1;
  int feats = 0;
  int i;
  
  if (PyArray_Check(obj)==0)
  {
   if (PySequence_Check(obj)==0)
   {
    PyErr_SetString(PyExc_TypeError, "Data matrix initialisation neither a numpy array nor a sequence.");
    return NULL;
   }
   
   // Its a sequence - check it only contains numpy arrays...
    fbc = PySequence_Size(obj);
    for (i=0; i<fbc; i++)
    {
     PyObject * member = PySequence_GetItem(obj, i);
     if (PyArray_Check(member)==0)
     {
      Py_DECREF(member);
      PyErr_SetString(PyExc_TypeError, "List element in data matrix initialisation not a numpy array.");
      return NULL;
     }
     
     feats += (PyArray_NDIM((PyArrayObject*)member)>1) ? PyArray_DIMS((PyArrayObject*)member)[1] : 1;
     Py_DECREF(member);
    }
  }
  else
  {
   feats = (PyArray_NDIM((PyArrayObject*)obj)>1) ? PyArray_DIMS((PyArrayObject*)obj)[1] : 1;
  }
  
 // Checks passed - malloc the object...
  DataMatrix * this = (DataMatrix*)malloc(sizeof(DataMatrix) + fbc*sizeof(FeatureBlock) + feats*sizeof(int));
  this->blocks = fbc;
  this->max = (int*)((char*)this + sizeof(DataMatrix) + fbc*sizeof(FeatureBlock));
  
 // Fill in the feature blocks...
  for (i=0; i<this->blocks; i++)
  {
   // Get the array for this object, with ownership...
    PyArrayObject * array;
    if (PyArray_Check(obj)!=0)
    {
     array = (PyArrayObject*)obj;
     Py_INCREF(array);
    }
    else
    {
     array = (PyArrayObject*)PySequence_GetItem(obj, i);
    }
    
   // Initalise the feature block...
    FeatureBlock_init(this->block + i, array);
    
   // Decriment the array ownership...
    Py_DECREF(array);
  }
  
 // Fill out the dimensions...
  this->exemplars = 0;
  this->features = 0;
  
  for (i=0; i<this->blocks; i++)
  {
   int e = PyArray_DIMS(this->block[i].array)[0];
   if (e>this->exemplars) this->exemplars = e;
   
   this->block[i].offset = this->features;
   this->features += this->block[i].features;
  }
  
 // If a maximum has been provided fill it in...
  for (i=0; i<this->features; i++) this->max[i] = -1;
 
  if (max!=NULL)
  {
   for (i=0; i<this->features; i++)
   {
    if (max[i]>=0) this->max[i] = max[i];
   }
  }
  
 // Return it...
  return this;
}

void DataMatrix_delete(DataMatrix * this)
{
 // Deinit the FeatureBlock-s...
  int i;
  for (i=0; i<this->blocks; i++)
  {
   FeatureBlock_deinit(this->block + i); 
  }
   
 // Terminate left leg with prejudice...
  free(this);
}



// Helper function - converts a feature index into a block/offset...
void DataMatrix_Pos(DataMatrix * this, int feature, int * block, int * offset)
{
 // Binary search to find the block...
  int low = 0;
  int high = this->blocks-1;
  while (low<high)
  {
   int half = (low + high) / 2;
   if (feature<this->block[half].offset) high = half;
   else
   {
    if (low!=half) low = half;
              else low = half+1;
   }
  }
 
 // Do the return...
  *block = low;
  *offset = feature - this->block[low].offset;
}

NumberType DataMatrix_Type(DataMatrix * this, int feature)
{
 // Translate the feature index...
  int block;
  int offset;
  DataMatrix_Pos(this, feature, &block, &offset);
 
 // Return the blocks type...
  return this->block[block].type;
}

int DataMatrix_GetDiscrete(DataMatrix * this, int exemplar, int feature)
{
 // Translate the feature index...
  int block;
  int offset;
  DataMatrix_Pos(this, feature, &block, &offset);
  
 // Do the block lookup...
  return FeatureBlock_GetDiscrete(this->block + block, exemplar, offset);
}

float DataMatrix_GetContinuous(DataMatrix * this, int exemplar, int feature)
{
 // Translate the feature index...
  int block;
  int offset;
  DataMatrix_Pos(this, feature, &block, &offset);

 // Do the block lookup...
  return FeatureBlock_GetContinuous(this->block + block, exemplar, offset);
}

int DataMatrix_Max(DataMatrix * this, int feature)
{
 // Create the max array automatically if required...
  if (this->max[feature]<0)
  {
   this->max[feature] = 0;
   
   int i;
   for (i=0; i<this->exemplars; i++)
   {
    int val = DataMatrix_GetDiscrete(this, i, feature);
    if (val>this->max[feature]) this->max[feature] = val;
   }
  }
 
 // Return whatever is recorded...
  return this->max[feature];
}



void Setup_DataMatrix(void)
{
 import_array();  
}
