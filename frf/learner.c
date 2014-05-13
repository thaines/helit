// Copyright 2014 Tom SF Haines

// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



#include "philox.h"
#include "data_matrix.h"
#include "index_set.h"
#include "information.h"

#include "learner.h"



// General learner access methopds, making the usual assumption that the Learner object starts with a pointer to its type...
Learner Learner_new(char code, DataMatrix * dm, int feature)
{
 int i = 0;
 while(ListLearner[i]!=NULL)
 {
  if (ListLearner[i]->code==code)
  {
   return ListLearner[i]->init(dm, feature);
  }
  ++i;
 }
 
 PyErr_SetString(PyExc_TypeError, "Unrecognised learner type code."); 
 return NULL;
}

void Learner_delete(Learner this)
{
 const LearnerType * type = *(const LearnerType **)this;
 type->deinit(this);
}

int Learner_optimise(Learner this, InfoSet * info, IndexView * view, int depth, float improve, unsigned int key[4])
{
 const LearnerType * type = *(const LearnerType **)this;
 return type->optimise(this, info, view, depth, improve, key);
}

char Learner_test_code(Learner this)
{
 const LearnerType * type = *(const LearnerType **)this; 
 return type->code_test;
}

float Learner_entropy(Learner this)
{
 const LearnerType * type = *(const LearnerType **)this; 
 return type->entropy(this);
}

size_t Learner_size(Learner this)
{
 const LearnerType * type = *(const LearnerType **)this;
 return type->size(this);
}

void Learner_fetch(Learner this, void * out)
{
 const LearnerType * type = *(const LearnerType **)this;
 return type->fetch(this, out);
}



// Structs for the test types...
// Continuous split: 'C'...
typedef struct ContinuousSplit ContinuousSplit;

struct ContinuousSplit
{
 int feature;
 float split; // less than == fail, greater than or equal = pass.
};

// Accept one class: 'D'...
typedef struct DiscreteSelect DiscreteSelect;

struct DiscreteSelect
{
 int feature;
 int accept; // Index of discrete feature it accepts.
};



// The no-op learner...
typedef struct Idiot Idiot;

struct Idiot
{
 const LearnerType * type;
};


Learner Idiot_new(DataMatrix * dm, int feature)
{
 Idiot * this = (Idiot*)malloc(sizeof(Idiot));
 this->type = &IdiotLearner;
 return this;
}

static void Idiot_delete(Learner this)
{
 free(this); 
}

static int Idiot_optimise(Learner this, InfoSet * info, IndexView * view, int depth, float improve, unsigned int key[4])
{
 return 0; // Its an idiot - it always fails!
}


const LearnerType IdiotLearner =
{
 'I',
 0,
 "Idiot",
 "A learner that can't learn - will never return a test and hence the system will ignore the feature its assigned to. Included for completeness - actual use cases all involve laziness.",
 Idiot_new,
 Idiot_delete,
 Idiot_optimise,
 NULL,
 NULL,
 NULL,
};



// The split learner...
typedef struct Split Split;

struct Split
{
 const LearnerType * type;
 
 DataMatrix * dm;
 int feature;
 
 float entropy;
 float split;
};


Learner Split_new(DataMatrix * dm, int feature)
{
 Split * this = (Split*)malloc(sizeof(Split));
 this->type = &SplitLearner;
 
 this->dm = dm;
 this->feature = feature;
 
 return this;
}

static void Split_delete(Learner this)
{
 free(this); 
}

// Helper for below - try not to think too much about what that static variable is for...
int sort_for_split(const void * a, const void * b)
{
 static const Split * this = NULL;
 if (b==NULL)
 {
  this = (const Split*)a;
  return 0;
 }
 
 float va = DataMatrix_GetContinuous(this->dm, *(const int*)a, this->feature);
 float vb = DataMatrix_GetContinuous(this->dm, *(const int*)b, this->feature);
  
 if (va<vb) return -1;
 if (vb>va) return 1;
 return 0;
}

static int Split_optimise(Learner self, InfoSet * info, IndexView * view, int depth, float improve, unsigned int key[4])
{
 Split * this = (Split*)self;
 int i;
 
 if (view->size<2) return 0;
 
 // Sort the indices of the IndexView by the associated feature - this involves a nasty qsort hack that I should probably be shot for (static variable that can be set by calling the sorting function with NULL as the second variable, so the sorting function has access to this.)...
  sort_for_split(this, NULL);
  qsort(view->vals, view->size, sizeof(int), sort_for_split);
 
 // Reset the InfoSet and fill in the pass half with all of the items...
  InfoSet_reset(info);
  for (i=0; i<view->size; i++)
  {
   InfoSet_pass_add(info, view->vals[i]); 
  }
  
 // Iterate moving one item from the 
  int success = 0;
  this->entropy = improve;
  
  for (i=0; i<view->size-1; i++)
  {
   InfoSet_pass_remove(info, view->vals[i]);
   InfoSet_fail_add(info, view->vals[i]);
   
   float e = InfoSet_entropy(info, depth);
   
   if (e<this->entropy)
   {
    this->entropy = e;
    this->split = 0.5 * (DataMatrix_GetContinuous(this->dm, view->vals[i], this->feature) + DataMatrix_GetContinuous(this->dm, view->vals[i+1], this->feature));
    success = 1;
   }
  }
  
 return success;
}

static float Split_entropy(Learner self)
{
 Split * this = (Split*)self;
 return this->entropy;
}

static size_t Split_size(Learner self)
{
 return sizeof(ContinuousSplit);
}

static void Split_fetch(Learner self, void * out)
{
 Split * this = (Split*)self;
 ContinuousSplit * dest = (ContinuousSplit*)out;
 
 dest->feature = this->feature;
 dest->split = this->split;
}


const LearnerType SplitLearner =
{
 'S',
 'C',
 "Split",
 "The standard feature splitting approach - finds the split point that maximises the information gain - default for continuous variables.",
 Split_new,
 Split_delete,
 Split_optimise,
 Split_entropy,
 Split_size,
 Split_fetch,
};



// The class selection learner...
typedef struct OneCat OneCat;

struct OneCat
{
 const LearnerType * type;
 
 DataMatrix * dm;
 int feature;
 int max;
 
 float entropy;
 int accept;
};


Learner OneCat_new(DataMatrix * dm, int feature)
{
 OneCat * this = (OneCat*)malloc(sizeof(OneCat));
 this->type = &OneCatLearner;
 
 this->dm = dm;
 this->feature = feature;
 this->max = DataMatrix_Max(dm, feature);
 
 return this;
}

static void OneCat_delete(Learner this)
{
 free(this); 
}

static int OneCat_optimise(Learner self, InfoSet * info, IndexView * view, int depth, float improve, unsigned int key[4])
{
 OneCat * this = (OneCat*)self;
 
 int success = 0;
 this->entropy = improve;
  
 
 // Loop and try accepting each class in turn...
  int i, j;
  for (i=0; i<this->max; i++)
  {
   // Reset...
    InfoSet_reset(info);
    
   // Add everyone in view into the correct halves...
    for (j=0; j<view->size; j++)
    {
     int exemplar = view->vals[j];
     int cat = DataMatrix_GetDiscrete(this->dm, exemplar, this->feature);
     
     if (cat==i) InfoSet_pass_add(info, exemplar);
            else InfoSet_fail_add(info, exemplar);
    }
    
   // Find if the entropy is good...
    float e = InfoSet_entropy(info, depth);
    if (e<this->entropy)
    {
     this->entropy = e;
     this->accept = i;
     success = 1; 
    }
  }
 
 return success;
}

static float OneCat_entropy(Learner self)
{
 OneCat * this = (OneCat*)self;
 return this->entropy;
}

static size_t OneCat_size(Learner self)
{
 return sizeof(DiscreteSelect);
}

static void OneCat_fetch(Learner self, void * out)
{
 OneCat * this = (OneCat*)self;
 DiscreteSelect * dest = (DiscreteSelect*)out;
 
 dest->feature = this->feature;
 dest->accept = this->accept;
}


const LearnerType OneCatLearner =
{
 'O',
 'D',
 "OneCat",
 "For discrete features - one category gets to pass, all others fail, optimised to maximise information. Default for discrete features.",
 OneCat_new,
 OneCat_delete,
 OneCat_optimise,
 OneCat_entropy,
 OneCat_size,
 OneCat_fetch,
};



// List of summary types that the system knows about...
const LearnerType * ListLearner[] =
{
 &IdiotLearner,
 &SplitLearner,
 &OneCatLearner,
 NULL
};



// Methods for LearnerSet...
LearnerSet * LearnerSet_new(DataMatrix * dm, const char * codes)
{
 // Verify that the codes array is the right length...
  if ((codes!=NULL)&&(strlen(codes)!=dm->features))
  {
   PyErr_SetString(PyExc_ValueError, "Learner codes do not match datamatrix feature count");
   return NULL; 
  }
  
 // Create the object...
  LearnerSet  * this = (LearnerSet*)malloc(sizeof(LearnerSet) + dm->features * (sizeof(Learner) + sizeof(int)));
 
 // Fill in the basics...
  this->best = -1;
  this->feat = (int*)((char*)this + sizeof(LearnerSet) + dm->features*sizeof(Learner));
  this->features = dm->features;
  
 // Fill it all in, creating the required learners...
  int i;
  for (i=0; i<this->features; i++)
  {
   if (codes!=NULL)
   {
    this->learn[i] = Learner_new(codes[i], dm, i);
    
    if (this->learn[i]==NULL)
    {
     // Error - unrecognised code - back out...
      i -= 1;
      for (; i>=0; i--)
      {
       Learner_delete(this->learn[i]);
      }
      free(this);
     
     return NULL;
    }
   }
   else
   {
    if (DataMatrix_Type(dm, i)==DISCRETE)
    {
     this->learn[i] = OneCat_new(dm, i);
    }
    else
    {
     this->learn[i] = Split_new(dm, i);
    }
   }
  }
 
 // Dump the indices into the feat array...
  for (i=0; i<this->features; i++)
  {
   this->feat[i] = i; 
  }
 
 // Return this...
 return this;
}

void LearnerSet_delete(LearnerSet * this)
{
 int i;
 for (i=0; i<this->features; i++)
 {
  Learner_delete(this->learn[i]); 
 }
 free(this);
}

int LearnerSet_optimise(LearnerSet * this, InfoSet * info, IndexView * view, int features, int depth, float improve, unsigned int key[4])
{
 int i;
 
 // Decide which features to optimise...
  if (features<this->features)
  {
   // We are not doing all of them - shuffle the feat array, at least enough entries for the later loop...
    unsigned int rand[4];
    int rand_index = 4;
    
    for (i=0; i<features; i++)
    {
     // Get some random data...
      if (rand_index>=4)
      {
       int j;
       for (j=0; j<4; j++) rand[j] = key[j];
       inc(key);
       
       philox(rand);       
       rand_index = 0; 
      }
      
      unsigned int r = rand[rand_index];
      rand_index += 1;
      
     // Select an index in feat to swap into the current position...
      int target = i + (r % (this->features - i));

     // Perform the swap...
      int temp = this->feat[i];
      this->feat[i] = this->feat[target];
      this->feat[target] = temp;
    }
  }
  else
  {
   features = this->features; 
  }
  
 // Loop and optimise each selected feature in turn, to choose the best...
  this->best = -1;
  
  for (i=0; i<features; i++)
  {
   int tf = this->feat[i];
   if (Learner_optimise(this->learn[tf], info, view, depth, improve, key)!=0)
   {
    float entropy = Learner_entropy(this->learn[tf]);
    if (entropy<improve)
    {
     improve = entropy;
     this->best = tf;
    }
   }
  }
  
 // Return success, as long as at least one optimisation was sucessful...
  if (this->best>=0) return 1;
                else return 0;
}

float LearnerSet_entropy(LearnerSet * this)
{
 return Learner_entropy(this->learn[this->best]); 
}

char LearnerSet_code(LearnerSet * this)
{
 return Learner_test_code(this->learn[this->best]); 
}

size_t LearnerSet_size(LearnerSet * this)
{
 return Learner_size(this->learn[this->best]); 
}

void LearnerSet_fetch(LearnerSet * this, void * out)
{
 Learner_fetch(this->learn[this->best], out); 
}



// Code to do the tests...
static int DoContinuousSplit(const void * test, DataMatrix * dm, int exemplar)
{
 const ContinuousSplit * this = test;
 
 float val = DataMatrix_GetContinuous(dm, exemplar, this->feature);
 
 if (val<this->split) return 0;
                 else return 1;
}

static size_t SizeContinuousSplit(const void * test)
{
 return sizeof(ContinuousSplit);
}

static int DoDiscreteSelect(const void * test, DataMatrix * dm, int exemplar)
{
 const DiscreteSelect * this = test;
 
 int val = DataMatrix_GetDiscrete(dm, exemplar, this->feature);
 
 if (val==this->accept) return 1;
                   else return 0;
}

static size_t SizeDiscreteSelect(const void * test)
{
 return sizeof(DiscreteSelect);
}



// Test calling management code...
DoTest   CodeToTest[256];
TestSize CodeToSize[256];

int Test(char code, const void * test, DataMatrix * dm, int exemplar)
{
 return CodeToTest[(unsigned char)code](test, dm, exemplar);
}

size_t Test_size(char code, const void * test)
{
 return CodeToSize[(unsigned char)code](test);
}

void SetupCodeToTest(void)
{
 int i;
 for (i=0; i<256; i++)
 {
  CodeToTest[i] = NULL;
  CodeToSize[i] = NULL;
 }
 
 CodeToTest['C'] = DoContinuousSplit;
 CodeToTest['D'] = DoDiscreteSelect;
 
 CodeToSize['C'] = SizeContinuousSplit;
 CodeToSize['D'] = SizeDiscreteSelect;
}



// Makes a warning go away...
void DoNotUse_learner_h(void)
{
 import_array();  
}
