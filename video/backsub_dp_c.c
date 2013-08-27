// Copyright 2012 Tom SF Haines

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.



#include <Python.h>
#include <structmember.h>
#include <numpy/arrayobject.h>

#include <fcntl.h>
#include <string.h>
#include <sys/time.h>



// Impliments the prefered counter-based pseudo random number generator from the paper 'Parallel Random Numbers: As easy as 1,2, 3'. This is the Philox model with a counter consisting of 4 values and a key consisting of 2 values, each 32 bits, with 10 rounds. Input is 4 32 bit unsigned integers as a counter and 2 32 bit unsigned integers as keys. The idea is that each key gives you a new sequence, which you step through using the counter. Note that it is designed with this indexing structure such that you can assign the details arbitarilly, and as long as you don't request the same counter/key combo twice you will get 'random' data. Typical use with a GPU is to have the counter increasing for each needed random input, whilst the key is set differently for each location the kernel is run, using get_global_id. You can just as easilly swap these roles around however. The original paper indicates that this algorithm passes a large battery of statistical tests for randomness, but this was implimented from scratch, without reference to the original authors code, so there is a risk of bugs which could cause significant bias in the results, so use at your own risk. Hasn't caused me any issues however, and I did run a (small) set of tests on it to verify it was returning a uniform distribution (p-values from binomial test done on each bucket of a histogram with 10 buckets, for multiple indexing strategies and sample counts.).
unsigned int mul_hi(unsigned int a, unsigned int b)
{
 uint64_t _a = a;
 uint64_t _b = b;
 
 return (_a * _b) >> 32;
}

// out is the counter on entry, the output when done.
void philox(unsigned int out[4], const unsigned int key[2])
{
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

// Wrapper for the above - bit inefficient as it only uses part of the 'random' data...
// (counter is trashed.)
float uniform(unsigned int counter[4], const unsigned int key[2])
{
 philox(counter, key);
 return ((float)counter[0]) / ((float)0xffffffff);
}



typedef struct Component Component;

struct Component
{
 // The parameters of the prior, for each colour channel, except for count which is shared. These are actually offsets from the prior parameters, so it will degrade back to the prior with time.
 float count;
 float mu[3];
 float sigma2[3]; // Actually divided by count, to make degradation simple.
};



typedef struct Pixel Pixel;

struct Pixel
{
 // The status of a pixel for the belief propagation labeling algorithm that thresholds and regularises the probability map to generate an actual mask...
 // For direction indices use 0=+ve x, 1=+ve y, 2=-ve x, 3 =-ve y.
 // Everything is done in terms of the background, as offsets in negative log probability space from foreground. This in effect means the cost values for foreground are stuck at 0, as all equations involve sums, which makes things very simple to evaluate. During evaluation a checkerboard update is used, hence why only input messages are stored.

 float bgCost; // negative log probability of assigning this pixel to background as an offset from the negative log probability of assigning this pixel to foreground.
 float changeCost[4]; // negative log probability of assigning a different label to a neighbour as an offset from the negative log probability of assigning the same label.

 float in[4]; // The 4 incomming messages from the 4 neighbours - negative log probability of assigning background as offsets from the negative log probability of assigning foreground.

 int count; // Number of pixels this is associated with - used to perform connected components.
 struct Pixel * parent; // Used for a disjoint set data structure during connected components.
};


// Used for the disjoint set data structure - given a pixel returns its parent; does path shortening...
Pixel * GetParent(Pixel * p)
{
 if (p->parent==0) return p;
 else
 {
  Pixel * ret = GetParent(p->parent);
  p->parent = ret;
  return ret;
 }
}



typedef struct BackSubCoreDP BackSubCoreDP;

struct BackSubCoreDP
{
 PyObject_HEAD

 int width;
 int height;
 int component_cap; // Maximum number of mixture components per pixel
 int frame;
 
 Component * comp;

 float prior_count; // Prior parameters for the Dirichlet processes Gaussian mixture model's Gaussians - a student-t distribution basically.
 float prior_mu[3]; // "
 float prior_sigma2[3]; // "

 float degradation; // Each frame the count for all samples is multiplied by this value, so old information degrades.
 float concentration; // Concentration for the DP at each pixel.
 float cap; // Maximum amount of weight that can be assigned to any given component, to prevent the model getting too certain - an alternative to degradation.

 float smooth; // Individual pixels are assumed to be the mean of a Gaussian with this variance.
 float weight; // Multiplier of pixel weight.
 float minWeight; // Minimum weight allowed for a pixel.

 float * temp; // Temporary buffer of size compCap used to cache multinomial over components.


 float threshold; // Threshold for mask generation - converted into a prior and used in a fully Bayesian sense.
 float cert_limit; // Limit on how extreme the probability of assignment can be.
 float change_limit; // Limit on how extreme the probability of a change can get.
 float min_same_prob; // At least one of the probabilities for the neighbours of a pixel being identical must be this high - distance is multiplied if needed...
 float change_mult; // Multiplier of change values, to fine tune the relative strength of regularisation versus the per-pixel model.
 float half_life; // At what colourmetric difference the probability of two adjacent pixels having the same label is 0.5, i.e. at which point they are considered sufficiently different to pass no information between each other.
 int iterations; // Number of iterations to do for bp mask creation.

 int con_comp_min; // For connected components, which is run after the bp step - any foreground segment smaller than the given size is removed.

 int minSize; // Minimum size for a level - basically how small the hierachy downsamples to.
 int itersPerLevel; // Iterations per level for hierachical BP.
 float com_count_mass; // Amount of probability mass to use for counting the number of components for each pixel.

 Pixel * pixel; // Array of pixel objects, for the bp masking and regularisation step, and also the connected components step.
};

Component * GetComponent(BackSubCoreDP * obj, int y, int x, int c)
{
 return &obj->comp[(y*obj->width + x)*obj->component_cap + c];
}



static PyObject * BackSubCoreDP_new(PyTypeObject * type, PyObject * args, PyObject * kwds)
{
 BackSubCoreDP * self = (BackSubCoreDP*)type->tp_alloc(type, 0);

 if (self!=NULL)
 {
  int i;

  self->width = 0;
  self->height = 0;
  self->component_cap = 0;
  self->frame = 0;
  
  self->comp = NULL;

  self->prior_count = 1.0;
  for (i=0;i<3;i++) self->prior_mu[i] = 0.5;
  for (i=0;i<3;i++) self->prior_sigma2[i] = 0.25;

  self->degradation = 1.0;
  self->concentration = 0.2;
  self->cap = 128.0;

  self->smooth = 0.0 / (255.0*255.0);
  self->weight = 1.0;
  self->minWeight = 0.01;

  self->temp = NULL;


  self->threshold = 0.5;
  self->cert_limit = 0.01;
  self->change_limit = 0.01;
  self->min_same_prob = 0.95;
  self->change_mult = 3.5;
  self->half_life = 0.1;
  self->iterations = 16;
  self->con_comp_min = 0;

  self->minSize = 8;
  self->itersPerLevel = 4;
  self->com_count_mass = 0.9;

  self->pixel = NULL;
 }

 return (PyObject*)self;
}

static void BackSubCoreDP_dealloc(BackSubCoreDP * self)
{
 free(self->comp);
 free(self->temp);
 free(self->pixel);
 self->ob_type->tp_free((PyObject*)self);
}



static PyMemberDef BackSubCoreDP_members[] =
{
    {"width", T_INT, offsetof(BackSubCoreDP, width), READONLY, "width of each frame"},
    {"height", T_INT, offsetof(BackSubCoreDP, height), READONLY, "height of each frame"},
    {"component_cap", T_INT, offsetof(BackSubCoreDP, component_cap), READONLY, "Maximum number of components allowed in the Dirichlet process assigned to each pixel."},
    {"prior_count", T_FLOAT, offsetof(BackSubCoreDP, prior_count), 0, "The number of samples the prior on each Gaussian is worth."},
    {"degradation", T_FLOAT, offsetof(BackSubCoreDP, degradation), 0, "The degradation of previous evidence, i.e. previous weights are multiplied by this term every frame. You can calculate the a value of this to acheive a half life of a given number of frames using degradation=0.5^(1/frames). Not supported by OpenCL version."},
    {"concentration", T_FLOAT, offsetof(BackSubCoreDP, concentration), 0, "The concentration used by the Dirichlet processes."},
    {"cap", T_FLOAT, offsetof(BackSubCoreDP, cap), 0, "The maximum weight that can be assigned to any given Dirichlet process - a limit on certainty. On reaching the cap *all* component weights are divided so the cap is maintained."},
    {"smooth", T_FLOAT, offsetof(BackSubCoreDP, smooth), 0, "Each sample is assumed to have a variance of this parameter - acts as a regularisation parameter to prevent extremelly pointy distributions that don't handle the occasional noise well. Not supported by OpenCL version."},
    {"weight", T_FLOAT, offsetof(BackSubCoreDP, weight), 0, "Multiplier for the weight used when adding new samples to the background distribution."},
    {"minWeight", T_FLOAT, offsetof(BackSubCoreDP, minWeight), 0, "Minimum weight for a new sample - used to avoid ignoring information completly."},
    {"threshold", T_FLOAT, offsetof(BackSubCoreDP, threshold), 0, "Threshold for the mask creation - gets converted into a prior for belief propagation, so this is not a hard limit."},
    {"cert_limit", T_FLOAT, offsetof(BackSubCoreDP, cert_limit), 0, "The probability of a pixel label assignment from the per-pixel density estimates is limited to be between this and one minus this."},
    {"change_limit", T_FLOAT, offsetof(BackSubCoreDP, change_limit), 0, "The probability of a pixel label change is capped between this value and one minus this value, to prevent extreme costs."},
    {"min_same_prob", T_FLOAT, offsetof(BackSubCoreDP, min_same_prob), 0, "For each pixel its probability of being identical to neighbours must, for at least one neighbour, not drop below this value. If needed the distances are scaled down to acheive this."},
    {"change_mult", T_FLOAT, offsetof(BackSubCoreDP, change_mult), 0, "Multiplier of the regularisation cost for changing label used to adjust the relative strength between the per-pixel model and the regularisation."},
    {"half_life", T_FLOAT, offsetof(BackSubCoreDP, half_life), 0, "The colourmetric distance between adjacent pixels after which they stop passing information between each other, because the probability of them being the same drops to 0.5."},
    {"iterations", T_INT, offsetof(BackSubCoreDP, iterations), 0, "Number of belief propagation iterations to do when creating a mask. Not supported by OpenCL version."},
    {"con_comp_min", T_INT, offsetof(BackSubCoreDP, con_comp_min), 0, "Sets the minimum foreground segment size in the final output - any that is less than this will be set as background. Not supported by OpenCL version."},
    {"minSize", T_INT, offsetof(BackSubCoreDP, minSize), 0, "Minimum size of either dimension when constructing the hierachy - the smallest level will get as close as possible without breaking this limit. Not supported by C version."},
    {"itersPerLevel", T_INT, offsetof(BackSubCoreDP, itersPerLevel), 0, "Number of iterations to do for each level of the BP hierachy. Not supported by C version."},
    {"com_count_mass", T_FLOAT, offsetof(BackSubCoreDP, com_count_mass), 0, "Amount of probability to consider when calculating how many mixture components a pixel has, to compensate for the fact the correct answer is infinity. Not avaliable in C version."},
    {NULL}
};



static PyObject * BackSubCoreDP_setup(BackSubCoreDP * self, PyObject * args)
{
 int width, height, comp_cap;
 if (!PyArg_ParseTuple(args, "iii", &width, &height, &comp_cap)) return NULL;

 Component * newComp = (Component*)malloc(width*height*comp_cap*sizeof(Component));
 float * newTemp = (float*)malloc(comp_cap*sizeof(float));
 Pixel * newPixel = (Pixel*)malloc(width*height*sizeof(Pixel));

 if ((newComp==NULL)||(newTemp==NULL)||(newPixel==NULL))
 {
  free(newComp);
  free(newTemp);
  free(newPixel);
  PyErr_NoMemory();
  return NULL;
 }

 free(self->comp);
 self->comp = newComp;
 free(self->temp);
 self->temp = newTemp;
 free(self->pixel);
 self->pixel = newPixel;

 self->width = width;
 self->height = height;
 self->component_cap = comp_cap;

 int y,x,c;
 Pixel * targ = self->pixel;
 for (y=0;y<self->height;y++)
 {
  for (x=0;x<self->width;x++)
  {
   for (c=0;c<self->component_cap;c++)
   {
    GetComponent(self,y,x,c)->count = 0.0;
   }
   for (c=0;c<4;c++) targ->in[c] = 0.0;
   ++targ;
  }
 }

 Py_INCREF(Py_None);
 return Py_None;
}



// Setters for the prior vectors...
static PyObject * BackSubCoreDP_set_prior_mu(BackSubCoreDP * self, PyObject * args)
{
 if (!PyArg_ParseTuple(args, "fff", &(self->prior_mu[0]), &(self->prior_mu[1]), &(self->prior_mu[2]))) return NULL;

 Py_INCREF(Py_None);
 return Py_None;
}

static PyObject * BackSubCoreDP_set_prior_sigma2(BackSubCoreDP * self, PyObject * args)
{
 if (!PyArg_ParseTuple(args, "fff", &(self->prior_sigma2[0]), &(self->prior_sigma2[1]), &(self->prior_sigma2[2]))) return NULL;

 Py_INCREF(Py_None);
 return Py_None;
}



// Given a new mean and variance this updates the prior, such that it does not distort the model - for use if changes are to be made during runtime...
static PyObject * BackSubCoreDP_prior_update(BackSubCoreDP * self, PyObject * args)
{
 // Extract the 2 parameters...
  float mean[3];
  float var[3];
  if (!PyArg_ParseTuple(args, "ffffff", &mean[0], &mean[1], &mean[2], &var[0], &var[1], &var[2])) return NULL;

 // Calculate the change required for each component, update the actual prior...
  int com;
  float deltaMean[3];
  float deltaVar[3];
  for (com=0;com<3;com++)
  {
   deltaMean[com] = mean[com] - self->prior_mu[com];
   deltaVar[com]  = var[com]  - self->prior_sigma2[com];
   self->prior_mu[com]     = mean[com];
   self->prior_sigma2[com] = var[com];
  }

 // Update the components...
  int total = self->width * self->height * self->component_cap;
  Component * targ = self->comp;
  while (total>0)
  {
   if (targ->count>1e-2)
   {
    for (com=0;com<3;com++)
    {
     targ->mu[com] -= deltaMean[com];
     targ->sigma2[com] -= deltaVar[com] / targ->count;
    }
   }

   ++targ;
   --total;
  }

 Py_INCREF(Py_None);
 return Py_None;
}



// Calculates the probability of the given rgb sample being drawn from the given component. Does not factor in the weighting of the component...
// (Includes some funky optimisations and approximations - doesn't look anything like the multiplication of 3 student-t distribution pdf's, but it is.)
float probComponent(BackSubCoreDP * self, Component * com, float * rgb)
{
 int i;

 // Calculate the parameters for the t-distributions...
  float n;
  float mean[3];
  float var[3];

  n = self->prior_count + com->count;
  float nMult = (n+1) / (n*n);
  for (i=0;i<3;i++)
  {
   mean[i] = self->prior_mu[i] + com->mu[i];
   var[i] = nMult * (self->prior_sigma2[i] + com->count*com->sigma2[i]);
  }

 // Calculate the shared parts of the student-t distribution - the normalising constant basically...
  float halfN = 0.5*n;
  float term = halfN + 0.5;
  //const float norm = 0.39894228040143276; // One hell of an approximation - conversion of Gamma terms to beta function, use of a large number approximation and then some canceling makes the normalising constant completly independent of all parameters, with some inaccuracy for lower values.
  const float norm_cube = 0.06349363593424101;

 // Evaluate the student-t distribution for each of the colour channels...
  float evalPart = 1.0;
  float evalCore = 1.0;
  for (i=0;i<3;i++)
  {
   float delta = rgb[i] - mean[i];
   evalCore *= 1.0 + (delta*delta / (n*var[i]));
   evalPart *= var[i];
  }

 // Return the multiplication of the terms, i.e. assume independence...
  return norm_cube / (sqrt(evalPart) * pow(evalCore,term));
}


static PyObject * BackSubCoreDP_process(BackSubCoreDP * self, PyObject * args)
{
 // Get the input and output numpy arrays...
  PyArrayObject * image;
  PyArrayObject * pixProb;
  if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &image, &PyArray_Type, &pixProb)) return NULL;

 // Zeroed out component, for calculating the probability of making a new one...
  int i;
  Component newbie;
  newbie.count = 0.0;
  for (i=0;i<3;i++)
  {
   newbie.mu[i] = 0.0;
   newbie.sigma2[i] = 0.0;
  }

 // Iterate them, processing each pixel in turn...
  int y,x,c;
  self->frame += 1;
  
  for (y=0;y<self->height;y++)
  {
   for (x=0;x<self->width;x++)
   {
    // Extract the relevant addresses - we assume we can index rgb as [0],[1] and [2]...
     float * rgb = (float*)(image->data + y*image->strides[0] + x*image->strides[1]);
     float * prob = (float*)(pixProb->data + y*pixProb->strides[0] + x*pixProb->strides[1]);

    // First pass over the pixels components - calculate the probability of assignment to each component and degrade the counts, whilst summing some useful values and finding a victim to replace if a new component is created...
     float probSum = self->concentration * probComponent(self, &newbie, rgb);
     float countSum = self->concentration;

     int lowIndex = 0;
     float lowValue = 1e9;
     float highValue = self->concentration;

     for (c=0;c<self->component_cap;c++)
     {
      Component * com = GetComponent(self,y,x,c);

      if (com->count>1e-2)
      {
       float prob = probComponent(self, com, rgb);

       self->temp[c] = com->count * prob;
       probSum += self->temp[c];
       countSum += com->count;

       com->count *= self->degradation;
      }
      else
      {
       self->temp[c] = 0.0;
      }

      if (com->count<lowValue)
      {
       lowIndex = c;
       lowValue = com->count;
      }

      if (com->count>highValue)
      {
       highValue = com->count;
      }
     }

    // Apply Bayes to get P(background|data), excluding the prior over background/foreground membership which is added in later (Effectivly uniform for the moment.). We don't have a foreground model, so assume a uniform distribution...
     *prob = probSum / countSum;
     *prob = *prob / (*prob + (highValue/self->cap)); // (highValue/self->cap) represents P(data|foreground), and is assuming a uniform distribution over the unit-sized colour space. The term used includes a fade in, so it is faded up as the model initialises, to avoid everything being marked as foreground at the beginning.


    // Calculate the weight - just reusing *prob...
     float weight = *prob;

    // Prevent the weight being too small for this step, as we don't want to completly ignore evidence...
     weight *= self->weight;
     if (weight<self->minWeight) weight = self->minWeight;

    // Draw a random number, for selecting a component...
     //unsigned int counter[4] = {x,y,self->frame,102349};
     //const unsigned int key[2] = {6546524,378946};
     float r = probSum * drand48();
     //float r = probSum * uniform(counter, key); // This option included to match the OpenCL version.

    // Second pass - assign it to a component, or create a new component...
     int done = 0;
     int home = lowIndex;
     for (c=0;c<self->component_cap;c++)
     {
      r -= self->temp[c];
      if (r<0.0)
      {
       done = 1;
       Component * com = GetComponent(self,y,x,c);
       home = c;

       float trueCount = self->prior_count + com->count;
       for (i=0;i<3;i++)
       {
        float trueMu = self->prior_mu[i] + com->mu[i];
        float trueSigma2 = self->prior_sigma2[i] + com->count*com->sigma2[i];

        float diff = rgb[i] - trueMu;
        com->mu[i] = (trueCount*trueMu + weight*rgb[i]) / (trueCount+weight);
        com->sigma2[i] = trueSigma2 + weight*self->smooth + trueCount*weight*diff*diff/(trueCount+weight);

        com->mu[i] -= self->prior_mu[i];
        com->sigma2[i] = (com->sigma2[i] - self->prior_sigma2[i]) / (com->count+weight);
       }
       com->count += weight;
       break;
      }
     }

     if (done==0) // New component time.
     {
      Component * com = GetComponent(self,y,x,lowIndex);

      com->count = weight;
      for (i=0;i<3;i++)
      {
       com->mu[i] = (self->prior_count*self->prior_mu[i] + weight*rgb[i]) / (self->prior_count+weight) - self->prior_mu[i];
       float diff = rgb[i] - self->prior_mu[i];
       com->sigma2[i] = (weight*self->smooth + self->prior_count*weight*diff*diff/(self->prior_count+weight))/weight;
      }
      // home = lowIndex;
     }

    // Apply the cap if needed...
     float top = GetComponent(self,y,x,home)->count;
     if (top>self->cap)
     {
      float mult = self->cap / top;
      for (c=0;c<self->component_cap;c++)
      {
       Component * com = GetComponent(self,y,x,c);
       com->count *= mult;
      }
     }
   }
  }

 Py_INCREF(Py_None);
 return Py_None;
}



static PyObject * BackSubCoreDP_light_update(BackSubCoreDP * self, PyObject * args)
{
 // Extract the 3 parameters...
 float mult[3];
 if (!PyArg_ParseTuple(args, "fff", &mult[0], &mult[1], &mult[2])) return NULL;

 // Iterate every single component and update the effective mean, taking the offset into account...
  int y,x,c,col;
  for (y=0;y<self->height;y++)
  {
   for (x=0;x<self->width;x++)
   {
    for (c=0;c<self->component_cap;c++)
    {
     Component * com = GetComponent(self,y,x,c);
     for (col=0;col<3;col++)
     {
      float est = (com->mu[col] + self->prior_mu[col]) * mult[col];
      if (est>1.0) est = 1.0; // No point in exceding the dynamic range - this seems to happen to skys a lot due to them being oversaturated.
      com->mu[col] = est - self->prior_mu[col];
     }
    }
   }
  }

 Py_INCREF(Py_None);
 return Py_None;
}



static PyObject * BackSubCoreDP_background(BackSubCoreDP * self, PyObject * args)
{
 // Get the output numpy array...
  PyArrayObject * image;
  if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &image)) return NULL;

 // Iterate the pixels and write the mode into each...
  int y,x,c,i;
  for (y=0;y<self->height;y++)
  {
   for (x=0;x<self->width;x++)
   {
    float * rgb = (float*)(image->data + y*image->strides[0] + x*image->strides[1]);

    float best = 0.0;
    for (c=0;c<self->component_cap;c++)
    {
     Component * com = GetComponent(self,y,x,c);
     if (com->count>best)
     {
      best = com->count;
      for (i=0;i<3;i++) rgb[i] = self->prior_mu[i] + com->mu[i];
     }
    }

    if (best<1e-3)
    {
     for (i=0;i<3;i++) rgb[i] = self->prior_mu[i];
    }
   }
  }

 Py_INCREF(Py_None);
 return Py_None;
}



static PyObject * BackSubCoreDP_make_mask(BackSubCoreDP * self, PyObject * args)
{
 int i,y,x,d,c;

 // Get the input and output numpy arrays...
  PyArrayObject * image;
  PyArrayObject * pixProb;
  PyArrayObject * mask;
  if (!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, &image, &PyArray_Type, &pixProb, &PyArray_Type, &mask)) return NULL;

 // Convert the threshold into a two label prior, as an offset to the background offset...
  float priorOffset = log(1.0-self->threshold) - log(self->threshold);


 // Fill in the pixel object with the relevant details extracted from the image and pixProb information...
  Pixel * targ = self->pixel;
  static const int dir_offset_x[] = {1,0,-1,0};
  static const int dir_offset_y[] = {0,1,0,-1};

  for (y=0;y<self->height;y++)
  {
   for (x=0;x<self->width;x++)
   {
    // Extract the relevant addresses - we assume we can index rgb as [0],[1] and [2]...
     float * rgb = (float*)(image->data + y*image->strides[0] + x*image->strides[1]);
     float * prob = (float*)(pixProb->data + y*pixProb->strides[0] + x*pixProb->strides[1]);

    // Calculate the relative nlp of making this a background pixel...
     float p = *prob;
     if (p<self->cert_limit) p = self->cert_limit;
     float omp = 1.0 - *prob;
     if (omp<self->cert_limit) omp = self->cert_limit;

     targ->bgCost = priorOffset + log(p/omp);

    // Fill in the distances from this pixel to its four neighbours, storing them in its change cost information - we convert them to actual nll change costs in the next step...
     for (d=0;d<4;d++)
     {
      int ox = x + dir_offset_x[d];
      int oy = y + dir_offset_y[d];
      if ((ox>=0)&&(ox<self->width)&&(oy>=0)&&(oy<self->height))
      {
       if (d>=2)
       {
        Pixel * other = self->pixel + self->width*oy + ox;
        targ->changeCost[d] = other->changeCost[(d+2)%4];
       }
       else
       {
        float * rgb2 = (float*)(image->data + oy*image->strides[0] + ox*image->strides[1]);
        float dist = 0.0;
        for (c=0;c<3;c++)
        {
         float delta = rgb[c] - rgb2[c];
         dist += delta*delta;
        }
        dist = sqrt(dist);

        if (dist<self->half_life) targ->changeCost[d] = dist;
        else targ->changeCost[d] = self->half_life;
       }
      }
      else
      {
       targ->changeCost[d] = self->half_life;
      }

      targ->in[d] = 0.0;
     }

    // Move to next...
     ++targ;
   }
  }


  targ = self->pixel;
  for (y=0;y<self->height;y++)
  {
   for (x=0;x<self->width;x++)
   {
    // Find the closest distance...
     float minDist = targ->changeCost[0];
     for (d=1;d<4;d++)
     {
      if (targ->changeCost[d]<minDist) minDist = targ->changeCost[d];
     }

    // Generate a multiplier for distance to acheive the min_same_prob requirement...
     float minDistTarget = (self->half_life / self->min_same_prob) - self->half_life;
     float distMult = (minDist<minDistTarget) ? 1.0 : (minDistTarget/minDist);

    // Convert the distances into nll offsets of changing class relative to remaining the same...
     for (d=0;d<4;d++)
     {
      float changeProb = 1.0 - (self->half_life / (self->half_life + distMult*targ->changeCost[d]));
      if (changeProb<self->change_limit) changeProb = self->change_limit;
      targ->changeCost[d] = self->change_mult * log((1.0-changeProb)/changeProb);
     }

    // Move to next...
     ++targ;
   }
  }


 // Iterate passing messages the given number of times - we hope for convergance...
  for (i=0;i<self->iterations;i++)
  {
   targ = self->pixel;
   if ((i%2)!=0) ++targ;

   for (y=0;y<self->height;y++)
   {
    for (x=((y+i)%2);x<self->width;x+=2)
    {
     float base = targ->bgCost;
     for (d=0;d<4;d++) base += targ->in[d];

     for (d=0;d<4;d++)
     {
      int dx = x + dir_offset_x[d];
      int dy = y + dir_offset_y[d];
      if ((dx>=0)&&(dx<self->width)&&(dy>=0)&&(dy<self->height))
      {
       Pixel * dest = self->pixel + self->width*dy + dx;
       int dd = (d+2)%4;

       float bgOffset = base - targ->in[d];

       // cost_<dest state>_<targ state>.
       float costFG_FG = 0.0;
       float costFG_BG = targ->changeCost[d] + bgOffset;
       float costBG_FG = targ->changeCost[d];
       float costBG_BG = bgOffset;

       float costFG = (costFG_FG<costFG_BG)?costFG_FG:costFG_BG;
       float costBG = (costBG_FG<costBG_BG)?costBG_FG:costBG_BG;

       dest->in[dd] = costBG - costFG;
      }
     }

     targ += 2;
    }
   }
  }

 // Extract the mask from the final set of messages...
  targ = self->pixel;
  for (y=0;y<self->height;y++)
  {
   for (x=0;x<self->width;x++)
   {
    float val = targ->bgCost;
    for (d=0;d<4;d++) val += targ->in[d];

    unsigned char * m = (unsigned char*)(mask->data + y*mask->strides[0] + x*mask->strides[1]);
    if (val<0.0) *m = 1;
            else *m = 0;

    ++targ;
   }
  }

 // If requested do connected components on the resulting mask...
  if (self->con_comp_min>1)
  {
   // Initialise the forest...
    targ = self->pixel;
    for (y=0;y<self->height;y++)
    {
     for (x=0;x<self->width;x++)
     {
      targ->count = 1;
      targ->parent = 0;
      ++targ;
     }
    }

   // Use the disjoint set data structure to find the connected regions...
    targ = self->pixel;
    for (y=0;y<self->height;y++)
    {
     for (x=0;x<self->width;x++)
     {
      unsigned char * m = (unsigned char*)(mask->data + y*mask->strides[0] + x*mask->strides[1]);
      if (m[0]==1)
      {
       // Merge right...
        if (((x+1)<self->width)&&(m[mask->strides[1]]==1))
        {
         Pixel * h1 = GetParent(targ);
         Pixel * h2 = GetParent(targ+1);

         if (h1!=h2)
         {
          h2->parent = h1;
          h1->count += h2->count;
         }
        }

       // Merge down...
        if (((y+1)<self->height)&&(m[mask->strides[0]]==1))
        {
         Pixel * h1 = GetParent(targ);
         Pixel * h2 = GetParent(targ+self->width);

         if (h1!=h2)
         {
          h2->parent = h1;
          h1->count += h2->count;
         }
        }
      }
      ++targ;
     }
    }

   // Iterate and make background all foreground pixels that belong to small segments...
    targ = self->pixel;
    for (y=0;y<self->height;y++)
    {
     for (x=0;x<self->width;x++)
     {
      if (GetParent(targ)->count<self->con_comp_min)
      {
       unsigned char * m = (unsigned char*)(mask->data + y*mask->strides[0] + x*mask->strides[1]);
       *m = 0;
      }
      ++targ;
     }
    }
  }

 Py_INCREF(Py_None);
 return Py_None;
}



static PyMethodDef BackSubCoreDP_methods[] =
{
 {"setup", (PyCFunction)BackSubCoreDP_setup, METH_VARARGS, "Sets the width, height and maximum number of components per pixel; the density estimates are reset."},
 {"set_prior_mu", (PyCFunction)BackSubCoreDP_set_prior_mu, METH_VARARGS, "Sets the mean of the prior."},
 {"set_prior_sigma2", (PyCFunction)BackSubCoreDP_set_prior_sigma2, METH_VARARGS, "Sets the sigma squared (variance) of the prior."},
 {"prior_update", (PyCFunction)BackSubCoreDP_prior_update, METH_VARARGS, "Updates the prior, in a way that is safe to be done during runtime, i.e. it also goes through and updates the rest of the model accordingly."},
 {"process", (PyCFunction)BackSubCoreDP_process, METH_VARARGS, "Given two inputs - a rgb frame indexed as [y,x,component] and a float32 output, indexed as [y,x]. It updates the model and writes the probability of seeing each pixel value into the output."},
 {"light_update", (PyCFunction)BackSubCoreDP_light_update, METH_VARARGS, "Given 3 floats, corresponding to red, green and blue - multiplies the means of all the components by these values - this allows the background model to track lighting changes."},
 {"background", (PyCFunction)BackSubCoreDP_background, METH_VARARGS, "Given an output float32 rgb array this fills it with the current mode of the per-pixel density estimates."},
 {"make_mask", (PyCFunction)BackSubCoreDP_make_mask, METH_VARARGS, "Helper method that is given 3 inputs: a rgb frame, a probability array, and a mask - it then uses the first two to fill in the third. Uses a two-label belief propagation implimentation that regularises the mask."},
 {NULL}
};



static PyTypeObject BackSubCoreDPType =
{
 PyObject_HEAD_INIT(NULL)
 0,                                 /*ob_size*/
 "backsub_dp_c.BackSubCoreCP",      /*tp_name*/
 sizeof(BackSubCoreDP),               /*tp_basicsize*/
 0,                                 /*tp_itemsize*/
 (destructor)BackSubCoreDP_dealloc, /*tp_dealloc*/
 0,                                 /*tp_print*/
 0,                                 /*tp_getattr*/
 0,                                 /*tp_setattr*/
 0,                                 /*tp_compare*/
 0,                                 /*tp_repr*/
 0,                                 /*tp_as_number*/
 0,                                 /*tp_as_sequence*/
 0,                                 /*tp_as_mapping*/
 0,                                 /*tp_hash */
 0,                                 /*tp_call*/
 0,                                 /*tp_str*/
 0,                                 /*tp_getattro*/
 0,                                 /*tp_setattro*/
 0,                                 /*tp_as_buffer*/
 Py_TPFLAGS_DEFAULT,                /*tp_flags*/
 "Impliments a background subtraction algorithm based on mixtures of Gaussians in a Dirchlet process mixture - this provides a basic interface with incomplete functionality, that needs to be encapsulated with something nicer in python.", /* tp_doc */
 0,                                 /* tp_traverse */
 0,                                 /* tp_clear */
 0,                                 /* tp_richcompare */
 0,                                 /* tp_weaklistoffset */
 0,                                 /* tp_iter */
 0,                                 /* tp_iternext */
 BackSubCoreDP_methods,             /* tp_methods */
 BackSubCoreDP_members,             /* tp_members */
 0,                                 /* tp_getset */
 0,                                 /* tp_base */
 0,                                 /* tp_dict */
 0,                                 /* tp_descr_get */
 0,                                 /* tp_descr_set */
 0,                                 /* tp_dictoffset */
 0,                                 /* tp_init */
 0,                                 /* tp_alloc */
 BackSubCoreDP_new,                 /* tp_new */
};



static PyMethodDef backsub_dp_c_methods[] =
{
 {NULL}
};



#ifndef PyMODINIT_FUNC
#define PyMODINIT_FUNC void
#endif

PyMODINIT_FUNC initbacksub_dp_c(void)
{
 PyObject * mod = Py_InitModule3("backsub_dp_c", backsub_dp_c_methods, "Provides a background subtraction algorithm based on a Dirichlet process Gaussian mixture model for each pixel.");
 import_array();

 if (PyType_Ready(&BackSubCoreDPType) < 0) return;

 Py_INCREF(&BackSubCoreDPType);
 PyModule_AddObject(mod, "BackSubCoreDP", (PyObject*)&BackSubCoreDPType);
}
