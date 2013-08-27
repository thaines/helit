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

#include <CL/cl.h>

#include "manager_cl.h"
#include "open_cl_help.h"



typedef struct BackSubCoreDP BackSubCoreDP;

struct BackSubCoreDP
{
 PyObject_HEAD

 cl_context context;
 cl_command_queue queue;
 cl_program program;


 int width;
 int height;
 int component_cap; // Maximum number of mixture components per pixel
 int frame;


 cl_mem image; // The image being processed.
 cl_mem mix; // The mixture components for the DP model.
 cl_mem pixel_prob; // The probabilities output by the DP model/used by BP.

 cl_float * image_temp; // Temporary for transfering into and out of images.
 cl_float * pixel_prob_temp; // As above, but for pixel_prob.
 cl_char * mask_temp; // Temporary, for transfering the mask back from the graphics card when it has been calculated.

 int layers; // The 5 below variables are all vectors of entities, of length layers.
 int * widthBL;
 int * heightBL;
 cl_mem * bgCost;
 cl_mem * changeCost;
 cl_mem * msgIn;

 cl_mem mask; // The final output of the algorithm!


 cl_kernel reset;            size_t reset_size;
 cl_kernel prior_update_mix; size_t prior_update_mix_size;
 cl_kernel light_update_mix; size_t light_update_mix_size;

 cl_kernel comp_prob;         size_t comp_prob_size;
 cl_kernel comp_prob_lum;     size_t comp_prob_lum_size;
 cl_kernel new_comp_prob;     size_t new_comp_prob_size;
 cl_kernel new_comp_prob_lum; size_t new_comp_prob_lum_size;
 cl_kernel update_pixel;      size_t update_pixel_size;

 cl_kernel extract_mode; size_t extract_mode_size;
 cl_kernel extract_component_count; size_t extract_component_count_size;

 cl_kernel setup_model_bgCost;        size_t setup_model_bgCost_size;
 cl_kernel setup_model_dist_boundary; size_t setup_model_dist_boundary_size;
 cl_kernel setup_model_dist_pos_x;    size_t setup_model_dist_pos_x_size;
 cl_kernel setup_model_dist_pos_y;    size_t setup_model_dist_pos_y_size;
 cl_kernel setup_model_dist_neg_x;    size_t setup_model_dist_neg_x_size;
 cl_kernel setup_model_dist_neg_y;    size_t setup_model_dist_neg_y_size;
 cl_kernel setup_model_changeCost;    size_t setup_model_changeCost_size;

 cl_kernel reset_in;   size_t reset_in_size;
 cl_kernel send_pos_x; size_t send_pos_x_size;
 cl_kernel send_pos_y; size_t send_pos_y_size;
 cl_kernel send_neg_x; size_t send_neg_x_size;
 cl_kernel send_neg_y; size_t send_neg_y_size;

 cl_kernel calc_mask;         size_t calc_mask_size;
 cl_kernel downsample_model;  size_t downsample_model_size;
 cl_kernel upsample_messages; size_t upsample_messages_size;


 int lum_only; // 0 for colour, anything else to only consider luminence.
 
 float prior_count; // Prior parameters for the Dirichlet processes Gaussian mixture model's Gaussians - a student-t distribution basically.
 float prior_mu[3]; // "
 float prior_sigma2[3]; // "

 float degradation; // Each frame the count for all samples is multiplied by this value, so old information degrades.
 float concentration; // Concentration for the DP at each pixel.
 float cap; // Maximum amount of weight that can be assigned to any given component, to prevent the model getting too certain - an alternative to degradation.

 float smooth; // Individual pixels are assumed to be the mean of a Gaussian with this variance.
 float varMult; // Multiplier of the variance used for handling pixel shake.
 float weight; // Multiplier of pixel weight.
 float minWeight; // Minimum weight allowed for a pixel.


 float threshold; // Threshold for mask generation - converted into a prior and used in a fully Bayesian sense.
 float cert_limit; // Limit on how extreme the probability of assignment can be.
 float change_limit; // Limit on how extreme the probability of a change can get.
 float min_same_prob; // At least one of the probabilities for the neighbours of a pixel being identical must be this high - distance is multiplied if needed...
 float change_mult; // Multiplier of change values, to fine tune the relative strength of regularisation versus the per-pixel model.
 float half_life; // At what colourmetric difference the probability of two adjacent pixels having the same label is 0.5, i.e. at which point they are considered sufficiently different to pass no information between each other.
 int iterations; // Number of iterations to do for bp mask creation.

 int con_comp_min; // For connected components, which is run after the bp step - any foreground segment smaller than the given size is removed.

 int minSize; // Minimum size for a level - basically how small the hierachy downsamples to.
 int maxLayers; // Maximum number of layers for hierachy.
 int itersPerLevel; // Iterations per level for hierachical BP, except for the bottom level, which is iterations.
 float com_count_mass; // Amount of probability mass to use for counting the number of components for each pixel.
};



static PyObject * BackSubCoreDP_new(PyTypeObject * type, PyObject * args, PyObject * kwds)
{
 BackSubCoreDP * self = (BackSubCoreDP*)type->tp_alloc(type, 0);

 if (self!=NULL)
 {
  int i;

  self->context = NULL;
  self->queue = NULL;
  self->program = NULL;

  self->width = 0;
  self->height = 0;
  self->component_cap = 0;
  self->frame = 0;

  self->image_temp = NULL;
  self->pixel_prob_temp = NULL;
  self->mask_temp = NULL;

  self->layers = 0;
  self->widthBL = NULL;
  self->heightBL = NULL;
  self->bgCost = NULL;
  self->changeCost = NULL;
  self->msgIn = NULL;

  self->mask = NULL;

  self->image = NULL;
  self->mix = NULL;
  self->pixel_prob = NULL;

  self->reset = NULL;            self->reset_size = 32;
  self->prior_update_mix = NULL; self->prior_update_mix_size = 32;
  self->light_update_mix = NULL; self->light_update_mix_size = 32;

  self->comp_prob = NULL;         self->comp_prob_size = 32;
  self->comp_prob_lum = NULL;     self->comp_prob_lum_size = 32;
  self->new_comp_prob = NULL;     self->new_comp_prob_size = 32;
  self->new_comp_prob_lum = NULL; self->new_comp_prob_lum_size = 32;
  self->update_pixel = NULL;      self->update_pixel_size = 32;

  self->extract_mode = NULL;            self->extract_mode_size = 32;
  self->extract_component_count = NULL; self->extract_component_count_size = 32;

  self->setup_model_bgCost = NULL;        self->setup_model_bgCost_size = 32;
  self->setup_model_dist_boundary = NULL; self->setup_model_dist_boundary_size = 32;
  self->setup_model_dist_pos_x = NULL;    self->setup_model_dist_pos_x_size = 32;
  self->setup_model_dist_pos_y = NULL;    self->setup_model_dist_pos_y_size = 32;
  self->setup_model_dist_neg_x = NULL;    self->setup_model_dist_neg_x_size = 32;
  self->setup_model_dist_neg_y = NULL;    self->setup_model_dist_neg_y_size = 32;
  self->setup_model_changeCost = NULL;    self->setup_model_changeCost_size = 32;

  self->reset_in   = NULL; self->reset_in_size   = 32;
  self->send_pos_x = NULL; self->send_pos_x_size = 32;
  self->send_pos_y = NULL; self->send_pos_y_size = 32;
  self->send_neg_x = NULL; self->send_neg_x_size = 32;
  self->send_neg_y = NULL; self->send_neg_y_size = 32;

  self->calc_mask = NULL;         self->calc_mask_size = 32;
  self->downsample_model = NULL;  self->downsample_model_size = 32;
  self->upsample_messages = NULL; self->upsample_messages_size = 32;

  
  self->lum_only = 0;
  
  self->prior_count = 1.0;
  for (i=0;i<3;i++) self->prior_mu[i] = 0.5;
  for (i=0;i<3;i++) self->prior_sigma2[i] = 0.25;

  self->degradation = 1.0;
  self->concentration = 0.2;
  self->cap = 128.0;

  self->smooth = 0.0 / (255.0*255.0);
  self->varMult = 1.0;
  self->weight = 1.0;
  self->minWeight = 0.01;


  self->threshold = 0.5;
  self->cert_limit = 0.01;
  self->change_limit = 0.01;
  self->min_same_prob = 0.95;
  self->change_mult = 3.5;
  self->half_life = 0.1;
  self->iterations = 16;
  self->con_comp_min = 0;

  self->minSize = 64;
  self->maxLayers = 4;
  self->itersPerLevel = 2;
  self->com_count_mass = 0.9;
 }

 return (PyObject*)self;
}

static void BackSubCoreDP_dealloc(BackSubCoreDP * self)
{
 clFinish(self->queue);


 clReleaseKernel(self->reset);
 clReleaseKernel(self->prior_update_mix);
 clReleaseKernel(self->light_update_mix);

 clReleaseKernel(self->comp_prob);
 clReleaseKernel(self->comp_prob_lum);
 clReleaseKernel(self->new_comp_prob);
 clReleaseKernel(self->new_comp_prob_lum);
 clReleaseKernel(self->update_pixel);

 clReleaseKernel(self->extract_mode);
 clReleaseKernel(self->extract_component_count);

 clReleaseKernel(self->setup_model_bgCost);
 clReleaseKernel(self->setup_model_dist_boundary);
 clReleaseKernel(self->setup_model_dist_pos_x);
 clReleaseKernel(self->setup_model_dist_pos_y);
 clReleaseKernel(self->setup_model_dist_neg_x);
 clReleaseKernel(self->setup_model_dist_neg_y);
 clReleaseKernel(self->setup_model_changeCost);

 clReleaseKernel(self->reset_in);
 clReleaseKernel(self->send_pos_x);
 clReleaseKernel(self->send_pos_y);
 clReleaseKernel(self->send_neg_x);
 clReleaseKernel(self->send_neg_y);

 clReleaseKernel(self->calc_mask);
 clReleaseKernel(self->downsample_model);
 clReleaseKernel(self->upsample_messages);


 free(self->widthBL);
 free(self->heightBL);
 if (self->bgCost)
 {
  int l;
  for (l=0;l<self->layers;l++) clReleaseMemObject(self->bgCost[l]);
  free(self->bgCost);
 }
 if (self->changeCost)
 {
  int l;
  for (l=0;l<self->layers;l++) clReleaseMemObject(self->changeCost[l]);
  free(self->changeCost);
 }
 if (self->msgIn)
 {
  int l;
  for (l=0;l<self->layers;l++) clReleaseMemObject(self->msgIn[l]);
  free(self->msgIn);
 }

 clReleaseMemObject(self->mask);

 clReleaseMemObject(self->image);
 clReleaseMemObject(self->mix);
 clReleaseMemObject(self->pixel_prob);

 clReleaseProgram(self->program);
 clReleaseCommandQueue(self->queue);
 clReleaseContext(self->context);

 free(self->image_temp);
 free(self->pixel_prob_temp);
 free(self->mask_temp);

 self->ob_type->tp_free((PyObject*)self);
}



static PyMemberDef BackSubCoreDP_members[] =
{
    {"width", T_INT, offsetof(BackSubCoreDP, width), READONLY, "width of each frame"},
    {"height", T_INT, offsetof(BackSubCoreDP, height), READONLY, "height of each frame"},
    {"component_cap", T_INT, offsetof(BackSubCoreDP, component_cap), READONLY, "Maximum number of components allowed in the Dirichlet process assigned to each pixel."},
    {"lum_only", T_INT, offsetof(BackSubCoreDP, lum_only), 0, "If 0 then all 3 channels are used, if anything else only the luminence (first) channel is used."},
    {"prior_count", T_FLOAT, offsetof(BackSubCoreDP, prior_count), 0, "The number of samples the prior on each Gaussian is worth."},
    {"degradation", T_FLOAT, offsetof(BackSubCoreDP, degradation), 0, "The degradation of previous evidence, i.e. previous weights are multiplied by this term every frame. You can calculate the a value of this to acheive a half life of a given number of frames using degradation=0.5^(1/frames). Not supported by OpenCL version."},
    {"concentration", T_FLOAT, offsetof(BackSubCoreDP, concentration), 0, "The concentration used by the Dirichlet processes."},
    {"cap", T_FLOAT, offsetof(BackSubCoreDP, cap), 0, "The maximum weight that can be assigned to any given Dirichlet process - a limit on certainty. On reaching the cap *all* component weights are divided so the cap is maintained."},
    {"smooth", T_FLOAT, offsetof(BackSubCoreDP, smooth), 0, "Each sample is assumed to have a variance of this parameter - acts as a regularisation parameter to prevent extremelly pointy distributions that don't handle the occasional noise well. Not supported by OpenCL version."},
    {"varMult", T_FLOAT, offsetof(BackSubCoreDP, varMult), 0, "Multiplier of variance calculated from area around pixel, to handle pixel-level camera shake. OpenCL version only."},
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
    {"minSize", T_INT, offsetof(BackSubCoreDP, minSize), 0, "Minimum size of either dimension when constructing the hierachy - the smallest layer will get as close as possible without breaking this limit. Not supported by C version."},
    {"maxLayers", T_INT, offsetof(BackSubCoreDP, maxLayers), 0, "Maximum number of layers to create for BP hierachy. Not supported by C version."},
    {"itersPerLevel", T_INT, offsetof(BackSubCoreDP, itersPerLevel), 0, "Number of iterations to do for each level of the BP hierachy. Not supported by C version."},
    {"com_count_mass", T_FLOAT, offsetof(BackSubCoreDP, com_count_mass), 0, "Amount of probability to consider when calculating how many mixture components a pixel has, to compensate for the fact the correct answer is infinity. Not avaliable in C version."},
    {NULL}
};



static PyObject * BackSubCoreDP_setup(BackSubCoreDP * self, PyObject * args)
{
 // Extract the parameters...
  ManagerCL * managerCL;
  int width, height, comp_cap;
  char * path;
  if (!PyArg_ParseTuple(args, "Oiiis", &managerCL, &width, &height, &comp_cap, &path)) return NULL;

  self->width = width;
  self->height = height;
  self->component_cap = comp_cap;

  cl_int status;

 // Get the context and queue from the manager...
  self->context = managerCL->context;
  clRetainContext(self->context);

  self->queue = managerCL->queue;
  clRetainCommandQueue(self->queue);


 // Load and compile the program...
  const char * file = "backsub_dp_cl.cl";
  self->program = prep_program(path, file, self->context, managerCL->device);
  if (self->program==NULL) return NULL;


 // Create the memory blocks needed on the device...
  // The image being processed, as [0,1] floats for each of the 3 colour components, for each pixel. Includes an extra float, for caching a single value in, and to get better alignment...
   self->image = clCreateBuffer(self->context, CL_MEM_READ_WRITE, self->height*self->width*4*sizeof(cl_float), NULL, &status);
   if (status!=CL_SUCCESS) return NULL;

  // 8 floats per DP component, for the right number of components per pixel, for all pixels. First entry is the count (n/k in the paper), the next 3 are then the 3 colour components mu values, as offsets from the prior, Then we have the sigma squared offsets from the prior, for each component, except they are divided by count for conveniance. Finally is a cache value, which is the probability of the current pixel belonging to that component, so we can calculate that in parallel...
   self->mix = clCreateBuffer(self->context, CL_MEM_READ_WRITE, self->height*self->width*self->component_cap*8*sizeof(cl_float), NULL, &status);
   if (status!=CL_SUCCESS) return NULL;

  // The output of the DP model, as probability of belonging to the background...
   self->pixel_prob = clCreateBuffer(self->context, CL_MEM_READ_WRITE, self->height*self->width*sizeof(cl_float), NULL, &status);
   if (status!=CL_SUCCESS) return NULL;

  // The hierarchy of models for the belief propagation post-processor...
   // First work out how many layers we need...
    self->layers = 0;
    int w = self->width;
    int h = self->height;
    while ((w>=self->minSize)&&(h>=self->minSize))
    {
     self->layers += 1;
     w /= 2;
     h /= 2;
    }

    if (self->layers==0) self->layers = 1; // In principal someone could have minSize set higher than the resolution of the input, most likelly to acheive non-hierarchical behaviour.
    if (self->layers>self->maxLayers) self->layers = self->maxLayers;

   // Fill in the width/height arrays...
    self->widthBL  = (int*)malloc(sizeof(int) * self->layers);
    self->heightBL = (int*)malloc(sizeof(int) * self->layers);

    int l;
    w = self->width;
    h = self->height;
    for (l=0;l<self->layers;l++)
    {
     self->widthBL[l] = w;
     self->heightBL[l] = h;

     w /= 2;
     h /= 2;
    }

   // Then allocate the arrays of memory buffers...
    self->bgCost = (cl_mem*)malloc(sizeof(cl_mem)*self->layers);
    self->changeCost = (cl_mem*)malloc(sizeof(cl_mem)*self->layers);
    self->msgIn = (cl_mem*)malloc(sizeof(cl_mem)*self->layers);

    for (l=0;l<self->layers;l++)
    {
     self->bgCost[l] = NULL;
     self->changeCost[l] = NULL;
     self->msgIn[l] = NULL;
    }

   // Finally, go through and allocate each buffer in turn...
    for (l=0;l<self->layers;l++)
    {
     self->bgCost[l] = clCreateBuffer(self->context, CL_MEM_READ_WRITE, self->widthBL[l]*self->heightBL[l]*sizeof(cl_float), NULL, &status);
     if (status!=CL_SUCCESS) return NULL;

     self->changeCost[l] = clCreateBuffer(self->context, CL_MEM_READ_WRITE, self->widthBL[l]*self->heightBL[l]*4*sizeof(cl_float), NULL, &status);
     if (status!=CL_SUCCESS) return NULL;

     self->msgIn[l] = clCreateBuffer(self->context, CL_MEM_READ_WRITE, self->widthBL[l]*self->heightBL[l]*4*sizeof(cl_float), NULL, &status);
     if (status!=CL_SUCCESS) return NULL;
    }

   // We also need somewhere to store the mask...
    self->mask = clCreateBuffer(self->context, CL_MEM_READ_WRITE, self->width*self->height*sizeof(cl_char), NULL, &status);
    if (status!=CL_SUCCESS) return NULL;


 // Some other memory blocks we need, as intermediaries when going between the OpenCL device and the main memory...
  self->image_temp = (cl_float*)malloc(self->height*self->width*4*sizeof(cl_float));
  self->pixel_prob_temp = (cl_float*)malloc(self->height*self->width*sizeof(cl_float));
  self->mask_temp = (cl_char*)malloc(self->height*self->width*sizeof(cl_char));


 // Setup the kernels...
  // Kernel to reset the model for each pixel...
   self->reset = clCreateKernel(self->program, "reset", &status);
   if (status!=CL_SUCCESS) return NULL;

   status |= clSetKernelArg(self->reset, 0, sizeof(cl_int), &self->width);
   status |= clSetKernelArg(self->reset, 1, sizeof(cl_int), &self->component_cap);
   status |= clSetKernelArg(self->reset, 2, sizeof(cl_mem), &self->mix);

   status |= clGetKernelWorkGroupInfo(self->reset, managerCL->device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &self->reset_size, NULL);

   if (status!=CL_SUCCESS) return NULL;

  // Kernel to update the mixture data structure when the prior is changed...
   self->prior_update_mix = clCreateKernel(self->program, "prior_update_mix", &status);
   if (status!=CL_SUCCESS) return NULL;

   status |= clSetKernelArg(self->prior_update_mix, 0, sizeof(cl_int), &self->width);
   status |= clSetKernelArg(self->prior_update_mix, 1, sizeof(cl_int), &self->component_cap);
   status |= clSetKernelArg(self->prior_update_mix, 4, sizeof(cl_mem), &self->mix);

   status |= clGetKernelWorkGroupInfo(self->prior_update_mix, managerCL->device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &self->prior_update_mix_size, NULL);

   if (status!=CL_SUCCESS) return NULL;

  // Kernel that updates the model for a multiplicative lighting change...
   self->light_update_mix = clCreateKernel(self->program, "light_update_mix", &status);
   if (status!=CL_SUCCESS) return NULL;

   status |= clSetKernelArg(self->light_update_mix, 0, sizeof(cl_int), &self->width);
   status |= clSetKernelArg(self->light_update_mix, 1, sizeof(cl_int), &self->component_cap);
   status |= clSetKernelArg(self->light_update_mix, 4, sizeof(cl_mem), &self->mix);

   status |= clGetKernelWorkGroupInfo(self->light_update_mix, managerCL->device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &self->light_update_mix_size, NULL);

   if (status!=CL_SUCCESS) return NULL;

  // Kernel that calculates the probability of each pixel being drawn from the associated component...
   self->comp_prob = clCreateKernel(self->program, "comp_prob", &status);
   if (status!=CL_SUCCESS) return NULL;

   status |= clSetKernelArg(self->comp_prob, 0, sizeof(cl_int), &self->width);
   status |= clSetKernelArg(self->comp_prob, 1, sizeof(cl_int), &self->component_cap);
   status |= clSetKernelArg(self->comp_prob, 2, sizeof(cl_float), &self->prior_count);
   status |= clSetKernelArg(self->comp_prob, 3, sizeof(cl_float4), self->prior_mu);
   status |= clSetKernelArg(self->comp_prob, 4, sizeof(cl_float4), self->prior_sigma2);
   status |= clSetKernelArg(self->comp_prob, 5, sizeof(cl_mem), &self->image);
   status |= clSetKernelArg(self->comp_prob, 6, sizeof(cl_mem), &self->mix);

   status |= clGetKernelWorkGroupInfo(self->comp_prob, managerCL->device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &self->comp_prob_size, NULL);

   if (status!=CL_SUCCESS) return NULL;
   
  // Luminence only version of the above...
   self->comp_prob_lum = clCreateKernel(self->program, "comp_prob_lum", &status);
   if (status!=CL_SUCCESS) return NULL;

   status |= clSetKernelArg(self->comp_prob_lum, 0, sizeof(cl_int), &self->width);
   status |= clSetKernelArg(self->comp_prob_lum, 1, sizeof(cl_int), &self->component_cap);
   status |= clSetKernelArg(self->comp_prob_lum, 2, sizeof(cl_float), &self->prior_count);
   status |= clSetKernelArg(self->comp_prob_lum, 3, sizeof(cl_float4), self->prior_mu);
   status |= clSetKernelArg(self->comp_prob_lum, 4, sizeof(cl_float4), self->prior_sigma2);
   status |= clSetKernelArg(self->comp_prob_lum, 5, sizeof(cl_mem), &self->image);
   status |= clSetKernelArg(self->comp_prob_lum, 6, sizeof(cl_mem), &self->mix);

   status |= clGetKernelWorkGroupInfo(self->comp_prob_lum, managerCL->device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &self->comp_prob_lum_size, NULL);

   if (status!=CL_SUCCESS) return NULL;

  // Kernel that calculates the new component probability (Stores it in the 4th colour channel of each pixel.)...
   self->new_comp_prob = clCreateKernel(self->program, "new_comp_prob", &status);
   if (status!=CL_SUCCESS) return NULL;

   status |= clSetKernelArg(self->new_comp_prob, 0, sizeof(cl_int), &self->width);
   status |= clSetKernelArg(self->new_comp_prob, 1, sizeof(cl_float), &self->prior_count);
   status |= clSetKernelArg(self->new_comp_prob, 2, sizeof(cl_float4), self->prior_mu);
   status |= clSetKernelArg(self->new_comp_prob, 3, sizeof(cl_float4), self->prior_sigma2);
   status |= clSetKernelArg(self->new_comp_prob, 4, sizeof(cl_mem), &self->image);

   status |= clGetKernelWorkGroupInfo(self->new_comp_prob, managerCL->device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &self->new_comp_prob_size, NULL);

   if (status!=CL_SUCCESS) return NULL;
   
  // Luminence only version of the above...
   self->new_comp_prob_lum = clCreateKernel(self->program, "new_comp_prob_lum", &status);
   if (status!=CL_SUCCESS) return NULL;

   status |= clSetKernelArg(self->new_comp_prob_lum, 0, sizeof(cl_int), &self->width);
   status |= clSetKernelArg(self->new_comp_prob_lum, 1, sizeof(cl_float), &self->prior_count);
   status |= clSetKernelArg(self->new_comp_prob_lum, 2, sizeof(cl_float4), self->prior_mu);
   status |= clSetKernelArg(self->new_comp_prob_lum, 3, sizeof(cl_float4), self->prior_sigma2);
   status |= clSetKernelArg(self->new_comp_prob_lum, 4, sizeof(cl_mem), &self->image);

   status |= clGetKernelWorkGroupInfo(self->new_comp_prob_lum, managerCL->device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &self->new_comp_prob_lum_size, NULL);

   if (status!=CL_SUCCESS) return NULL;

  // Kernel to combine all of the above and calculate the final probability, before selecting and updating one of the mixture components - this is where most of the work occurs...
   self->update_pixel = clCreateKernel(self->program, "update_pixel", &status);
   if (status!=CL_SUCCESS) return NULL;

   status |= clSetKernelArg(self->update_pixel,  0, sizeof(cl_int), &self->frame);
   status |= clSetKernelArg(self->update_pixel,  1, sizeof(cl_int), &self->width);
   status |= clSetKernelArg(self->update_pixel,  2, sizeof(cl_int), &self->height);
   status |= clSetKernelArg(self->update_pixel,  3, sizeof(cl_int), &self->component_cap);
   status |= clSetKernelArg(self->update_pixel,  4, sizeof(cl_float), &self->prior_count);
   status |= clSetKernelArg(self->update_pixel,  5, sizeof(cl_float4), self->prior_mu);
   status |= clSetKernelArg(self->update_pixel,  6, sizeof(cl_float4), self->prior_sigma2);
   status |= clSetKernelArg(self->update_pixel,  7, sizeof(cl_float), &self->concentration);
   status |= clSetKernelArg(self->update_pixel,  8, sizeof(cl_float), &self->cap);
   status |= clSetKernelArg(self->update_pixel,  9, sizeof(cl_float), &self->weight);
   status |= clSetKernelArg(self->update_pixel, 10, sizeof(cl_float), &self->minWeight);
   status |= clSetKernelArg(self->update_pixel, 11, sizeof(cl_mem), &self->image);
   status |= clSetKernelArg(self->update_pixel, 12, sizeof(cl_mem), &self->mix);
   status |= clSetKernelArg(self->update_pixel, 13, sizeof(cl_mem), &self->pixel_prob);
   status |= clSetKernelArg(self->update_pixel, 14, sizeof(cl_float), &self->varMult);

   status |= clGetKernelWorkGroupInfo(self->update_pixel, managerCL->device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &self->update_pixel_size, NULL);

   if (status!=CL_SUCCESS) return NULL;

  // Kernel used to extract the background image...
   self->extract_mode = clCreateKernel(self->program, "extract_mode", &status);
   if (status!=CL_SUCCESS) return NULL;

   status |= clSetKernelArg(self->extract_mode, 0, sizeof(cl_int), &self->width);
   status |= clSetKernelArg(self->extract_mode, 1, sizeof(cl_int), &self->component_cap);
   status |= clSetKernelArg(self->extract_mode, 2, sizeof(cl_float4), self->prior_mu);
   status |= clSetKernelArg(self->extract_mode, 3, sizeof(cl_mem), &self->image);
   status |= clSetKernelArg(self->extract_mode, 4, sizeof(cl_mem), &self->mix);

   status |= clGetKernelWorkGroupInfo(self->extract_mode, managerCL->device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &self->extract_mode_size, NULL);

   if (status!=CL_SUCCESS) return NULL;
   
  // Kernel used to extract component counts for each pixel...
   self->extract_component_count = clCreateKernel(self->program, "extract_component_count", &status);
   if (status!=CL_SUCCESS) return NULL;

   status |= clSetKernelArg(self->extract_component_count, 0, sizeof(cl_int), &self->width);
   status |= clSetKernelArg(self->extract_component_count, 1, sizeof(cl_int), &self->component_cap);
   status |= clSetKernelArg(self->extract_component_count, 2, sizeof(cl_float), &self->com_count_mass);
   status |= clSetKernelArg(self->extract_component_count, 3, sizeof(cl_float), &self->cap);
   status |= clSetKernelArg(self->extract_component_count, 4, sizeof(cl_mem), &self->image);
   status |= clSetKernelArg(self->extract_component_count, 5, sizeof(cl_mem), &self->mix);

   status |= clGetKernelWorkGroupInfo(self->extract_component_count, managerCL->device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &self->extract_component_count_size, NULL);

   if (status!=CL_SUCCESS) return NULL;

  // The kernel to convert the probability buffer into a nll offset buffer...
   self->setup_model_bgCost = clCreateKernel(self->program, "setup_model_bgCost", &status);
   if (status!=CL_SUCCESS) return NULL;

   status |= clSetKernelArg(self->setup_model_bgCost, 0, sizeof(cl_int), &self->width);
   status |= clSetKernelArg(self->setup_model_bgCost, 3, sizeof(cl_mem), &self->pixel_prob);
   status |= clSetKernelArg(self->setup_model_bgCost, 4, sizeof(cl_mem), &self->bgCost[0]);

   status |= clGetKernelWorkGroupInfo(self->setup_model_bgCost, managerCL->device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &self->setup_model_bgCost_size, NULL);

   if (status!=CL_SUCCESS) return NULL;

  // Used to fill in the distances at the boundary, for calculating changeCost...
   self->setup_model_dist_boundary = clCreateKernel(self->program, "setup_model_dist_boundary", &status);
   if (status!=CL_SUCCESS) return NULL;

   status |= clSetKernelArg(self->setup_model_dist_boundary, 0, sizeof(cl_int), &self->width);
   status |= clSetKernelArg(self->setup_model_dist_boundary, 2, sizeof(cl_mem), &self->image);
   status |= clSetKernelArg(self->setup_model_dist_boundary, 3, sizeof(cl_mem), &self->changeCost[0]);

   status |= clGetKernelWorkGroupInfo(self->setup_model_dist_boundary, managerCL->device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &self->setup_model_dist_boundary_size, NULL);

   if (status!=CL_SUCCESS) return NULL;

  // For calculating the distance between a pixel and the one next to it in the positive x direction, which is then stored in changeCost...
   self->setup_model_dist_pos_x = clCreateKernel(self->program, "setup_model_dist_pos_x", &status);
   if (status!=CL_SUCCESS) return NULL;

   status |= clSetKernelArg(self->setup_model_dist_pos_x, 0, sizeof(cl_int), &self->width);
   status |= clSetKernelArg(self->setup_model_dist_pos_x, 1, sizeof(cl_mem), &self->image);
   status |= clSetKernelArg(self->setup_model_dist_pos_x, 2, sizeof(cl_mem), &self->changeCost[0]);

   status |= clGetKernelWorkGroupInfo(self->setup_model_dist_pos_x, managerCL->device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &self->setup_model_dist_pos_x_size, NULL);

   if (status!=CL_SUCCESS) return NULL;

  // For calculating the distance between a pixel and the one next to it in the positive y direction, which is then stored in changeCost...
   self->setup_model_dist_pos_y = clCreateKernel(self->program, "setup_model_dist_pos_y", &status);
   if (status!=CL_SUCCESS) return NULL;

   status |= clSetKernelArg(self->setup_model_dist_pos_y, 0, sizeof(cl_int), &self->width);
   status |= clSetKernelArg(self->setup_model_dist_pos_y, 1, sizeof(cl_mem), &self->image);
   status |= clSetKernelArg(self->setup_model_dist_pos_y, 2, sizeof(cl_mem), &self->changeCost[0]);

   status |= clGetKernelWorkGroupInfo(self->setup_model_dist_pos_y, managerCL->device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &self->setup_model_dist_pos_y_size, NULL);

   if (status!=CL_SUCCESS) return NULL;

  // For calculating the distance between a pixel and the one next to it in the negative x direction, which is then stored in changeCost...
   self->setup_model_dist_neg_x = clCreateKernel(self->program, "setup_model_dist_neg_x", &status);
   if (status!=CL_SUCCESS) return NULL;

   status |= clSetKernelArg(self->setup_model_dist_neg_x, 0, sizeof(cl_int), &self->width);
   status |= clSetKernelArg(self->setup_model_dist_neg_x, 1, sizeof(cl_mem), &self->image);
   status |= clSetKernelArg(self->setup_model_dist_neg_x, 2, sizeof(cl_mem), &self->changeCost[0]);

   status |= clGetKernelWorkGroupInfo(self->setup_model_dist_neg_x, managerCL->device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &self->setup_model_dist_neg_x_size, NULL);

   if (status!=CL_SUCCESS) return NULL;

  // For calculating the distance between a pixel and the one next to it in the negative x direction, which is then stored in changeCost...
   self->setup_model_dist_neg_y = clCreateKernel(self->program, "setup_model_dist_neg_y", &status);
   if (status!=CL_SUCCESS) return NULL;

   status |= clSetKernelArg(self->setup_model_dist_neg_y, 0, sizeof(cl_int), &self->width);
   status |= clSetKernelArg(self->setup_model_dist_neg_y, 1, sizeof(cl_mem), &self->image);
   status |= clSetKernelArg(self->setup_model_dist_neg_y, 2, sizeof(cl_mem), &self->changeCost[0]);

   status |= clGetKernelWorkGroupInfo(self->setup_model_dist_neg_y, managerCL->device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &self->setup_model_dist_neg_y_size, NULL);

   if (status!=CL_SUCCESS) return NULL;

  // This converts the distances stored in changeCost by the previous 5 kernels and converts them into actual costs...
   self->setup_model_changeCost = clCreateKernel(self->program, "setup_model_changeCost", &status);
   if (status!=CL_SUCCESS) return NULL;

   status |= clSetKernelArg(self->setup_model_changeCost, 0, sizeof(cl_int), &self->width);
   status |= clSetKernelArg(self->setup_model_changeCost, 5, sizeof(cl_mem), &self->changeCost[0]);

   status |= clGetKernelWorkGroupInfo(self->setup_model_changeCost, managerCL->device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &self->setup_model_changeCost_size, NULL);

   if (status!=CL_SUCCESS) return NULL;

  // For zeroing out the msgIn variables...
   self->reset_in = clCreateKernel(self->program, "reset_in", &status);
   status |= clGetKernelWorkGroupInfo(self->reset_in, managerCL->device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &self->reset_in_size, NULL);
   if (status!=CL_SUCCESS) return NULL;

  // The 4 kernels for sending messages in the 4 compass directions...
   self->send_pos_x = clCreateKernel(self->program, "send_pos_x", &status);
   status |= clGetKernelWorkGroupInfo(self->send_pos_x, managerCL->device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &self->send_pos_x_size, NULL);
   if (status!=CL_SUCCESS) return NULL;

   self->send_pos_y = clCreateKernel(self->program, "send_pos_y", &status);
   status |= clGetKernelWorkGroupInfo(self->send_pos_y, managerCL->device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &self->send_pos_y_size, NULL);
   if (status!=CL_SUCCESS) return NULL;

   self->send_neg_x = clCreateKernel(self->program, "send_neg_x", &status);
   status |= clGetKernelWorkGroupInfo(self->send_neg_x, managerCL->device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &self->send_neg_x_size, NULL);
   if (status!=CL_SUCCESS) return NULL;

   self->send_neg_y = clCreateKernel(self->program, "send_neg_y", &status);
   status |= clGetKernelWorkGroupInfo(self->send_neg_y, managerCL->device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &self->send_neg_y_size, NULL);
   if (status!=CL_SUCCESS) return NULL;

  // When all the iterations are done, this generates the final mask, ready for transfer off the device for normal usage...
   self->calc_mask = clCreateKernel(self->program, "calc_mask", &status);
   if (status!=CL_SUCCESS) return NULL;

   status |= clSetKernelArg(self->calc_mask, 0, sizeof(cl_int), &self->width);
   status |= clSetKernelArg(self->calc_mask, 1, sizeof(cl_mem), &self->bgCost[0]);
   status |= clSetKernelArg(self->calc_mask, 2, sizeof(cl_mem), &self->msgIn[0]);
   status |= clSetKernelArg(self->calc_mask, 3, sizeof(cl_mem), &self->mask);

   status |= clGetKernelWorkGroupInfo(self->calc_mask, managerCL->device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &self->calc_mask_size, NULL);

   if (status!=CL_SUCCESS) return NULL;

  // Kernels for convertiong data between layers of the BP hierarchy...
   self->downsample_model = clCreateKernel(self->program, "downsample_model", &status);
   status |= clGetKernelWorkGroupInfo(self->downsample_model, managerCL->device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &self->downsample_model_size, NULL);
   if (status!=CL_SUCCESS) return NULL;

   self->upsample_messages = clCreateKernel(self->program, "upsample_messages", &status);
   status |= clGetKernelWorkGroupInfo(self->upsample_messages, managerCL->device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &self->upsample_messages_size, NULL);
   if (status!=CL_SUCCESS) return NULL;


 // Run the reset kernel, to intiialise the data structure...
  size_t work_size[3];
  size_t block_size[3];

  work_size[0] = self->component_cap;
  work_size[1] = self->width;
  work_size[2] = self->height;
  calc_block_size(self->reset_size, 3, work_size, block_size, 1);
  status = clEnqueueNDRangeKernel(self->queue, self->reset, 3, NULL, work_size, block_size, 0, NULL, NULL);
  if (status!=CL_SUCCESS) return NULL;


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
  cl_float deltaMu[3];
  cl_float deltaSigma2[3];
  for (com=0;com<3;com++)
  {
   deltaMu[com]      = mean[com] - self->prior_mu[com];
   deltaSigma2[com]  = var[com]  - self->prior_sigma2[com];
   self->prior_mu[com]     = mean[com];
   self->prior_sigma2[com] = var[com];
  }


 // Send a kernel running to update all the components...
  cl_int status = CL_SUCCESS;
  status |= clSetKernelArg(self->prior_update_mix, 2, sizeof(cl_float4), deltaMu);
  status |= clSetKernelArg(self->prior_update_mix, 3, sizeof(cl_float4), deltaSigma2);
  if (status!=CL_SUCCESS) return NULL;


  size_t work_size[3];
  size_t block_size[3];

  work_size[0] = self->component_cap;
  work_size[1] = self->width;
  work_size[2] = self->height;
  calc_block_size(self->prior_update_mix_size, 3, work_size, block_size, 1);
  status = clEnqueueNDRangeKernel(self->queue, self->prior_update_mix, 3, NULL, work_size, block_size, 0, NULL, NULL);
  if (status!=CL_SUCCESS) return NULL;

  status = clEnqueueBarrier(self->queue);
  if (status!=CL_SUCCESS) return NULL;


 Py_INCREF(Py_None);
 return Py_None;
}



static PyObject * BackSubCoreDP_process(BackSubCoreDP * self, PyObject * args)
{
 // Get the input and output numpy arrays...
  PyArrayObject * image;
  PyArrayObject * pixProb;
  if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &image, &PyArray_Type, &pixProb)) return NULL;


 //struct timeval start, end; // ********************************************
 //gettimeofday(&start, 0); // **********************************************


 // Update the kernel values as needed...
  self->frame += 1;

  cl_int status = CL_SUCCESS;
  
  if (self->lum_only==0)
  {
   status |= clSetKernelArg(self->comp_prob, 2, sizeof(cl_float), &self->prior_count);
   status |= clSetKernelArg(self->comp_prob, 3, sizeof(cl_float4), self->prior_mu);
   status |= clSetKernelArg(self->comp_prob, 4, sizeof(cl_float4), self->prior_sigma2);

   status |= clSetKernelArg(self->new_comp_prob, 1, sizeof(cl_float), &self->prior_count);
   status |= clSetKernelArg(self->new_comp_prob, 2, sizeof(cl_float4), self->prior_mu);
   status |= clSetKernelArg(self->new_comp_prob, 3, sizeof(cl_float4), self->prior_sigma2);
  }
  else
  {
   status |= clSetKernelArg(self->comp_prob_lum, 2, sizeof(cl_float), &self->prior_count);
   status |= clSetKernelArg(self->comp_prob_lum, 3, sizeof(cl_float4), self->prior_mu);
   status |= clSetKernelArg(self->comp_prob_lum, 4, sizeof(cl_float4), self->prior_sigma2);

   status |= clSetKernelArg(self->new_comp_prob_lum, 1, sizeof(cl_float), &self->prior_count);
   status |= clSetKernelArg(self->new_comp_prob_lum, 2, sizeof(cl_float4), self->prior_mu);
   status |= clSetKernelArg(self->new_comp_prob_lum, 3, sizeof(cl_float4), self->prior_sigma2);
  }

  status |= clSetKernelArg(self->update_pixel,  0, sizeof(cl_int), &self->frame);
  status |= clSetKernelArg(self->update_pixel,  4, sizeof(cl_float), &self->prior_count);
  status |= clSetKernelArg(self->update_pixel,  5, sizeof(cl_float4), self->prior_mu);
  status |= clSetKernelArg(self->update_pixel,  6, sizeof(cl_float4), self->prior_sigma2);
  status |= clSetKernelArg(self->update_pixel,  7, sizeof(cl_float), &self->concentration);
  status |= clSetKernelArg(self->update_pixel,  8, sizeof(cl_float), &self->cap);
  status |= clSetKernelArg(self->update_pixel,  9, sizeof(cl_float), &self->weight);
  status |= clSetKernelArg(self->update_pixel, 10, sizeof(cl_float), &self->minWeight);
  status |= clSetKernelArg(self->update_pixel, 14, sizeof(cl_float), &self->varMult);

  if (status!=CL_SUCCESS) return NULL;

 // Move the image to the OpenCL device...
  int y, x, i;
  for (y=0;y<self->height;y++)
  {
   for (x=0;x<self->width;x++)
   {
    float * in = (float*)(image->data + y*image->strides[0] + x*image->strides[1]);
    cl_float * out = self->image_temp + (y*self->width + x)*4;

    for (i=0;i<3;i++) out[i] = in[i];
    out[3] = 0.0;
   }
  }

  status = clEnqueueWriteBuffer(self->queue, self->image, CL_FALSE, 0, self->height*self->width*4*sizeof(cl_float), self->image_temp, 0, NULL, NULL);
  if (status!=CL_SUCCESS) return NULL;

  status = clEnqueueBarrier(self->queue);
  if (status!=CL_SUCCESS) return NULL;

 // Enqueue the work, making sure to wait as needed...
  size_t work_size[3];
  size_t block_size[3];
  
  if (self->lum_only==0)
  {
   work_size[0] = self->component_cap;
   work_size[1] = self->width;
   work_size[2] = self->height;
   calc_block_size(self->comp_prob_size, 3, work_size, block_size, 1);
   status = clEnqueueNDRangeKernel(self->queue, self->comp_prob, 3, NULL, work_size, block_size, 0, NULL, NULL);
   if (status!=CL_SUCCESS) return NULL;


   work_size[0] = self->width;
   work_size[1] = self->height;
   calc_block_size(self->new_comp_prob_size, 2, work_size, block_size, 0);
   status = clEnqueueNDRangeKernel(self->queue, self->new_comp_prob, 2, NULL, work_size, block_size, 0, NULL, NULL);
   if (status!=CL_SUCCESS) return NULL;

   status = clEnqueueBarrier(self->queue);
   if (status!=CL_SUCCESS) return NULL;
  }
  else
  {
   work_size[0] = self->component_cap;
   work_size[1] = self->width;
   work_size[2] = self->height;
   calc_block_size(self->comp_prob_lum_size, 3, work_size, block_size, 1);
   status = clEnqueueNDRangeKernel(self->queue, self->comp_prob_lum, 3, NULL, work_size, block_size, 0, NULL, NULL);
   if (status!=CL_SUCCESS) return NULL;


   work_size[0] = self->width;
   work_size[1] = self->height;
   calc_block_size(self->new_comp_prob_lum_size, 2, work_size, block_size, 0);
   status = clEnqueueNDRangeKernel(self->queue, self->new_comp_prob_lum, 2, NULL, work_size, block_size, 0, NULL, NULL);
   if (status!=CL_SUCCESS) return NULL;

   status = clEnqueueBarrier(self->queue);
   if (status!=CL_SUCCESS) return NULL;
  }
  
  
  work_size[0] = self->width;
  work_size[1] = self->height;
  calc_block_size(self->update_pixel_size, 2, work_size, block_size, 0);
  status = clEnqueueNDRangeKernel(self->queue, self->update_pixel, 2, NULL, work_size, block_size, 0, NULL, NULL);
  if (status!=CL_SUCCESS) return NULL;

  status = clEnqueueBarrier(self->queue);
  if (status!=CL_SUCCESS) return NULL;

 // Extract the probability map...
  status = clEnqueueReadBuffer(self->queue, self->pixel_prob, CL_FALSE, 0, self->height*self->width*sizeof(cl_float), self->pixel_prob_temp, 0, NULL, NULL);
  if (status!=CL_SUCCESS) return NULL;

 // Wait until the work is done...
  status = clFinish(self->queue);
  if (status!=CL_SUCCESS) return NULL;


 // Transfer pixel_prob_temp over so that it can be used...
  for (y=0;y<self->height;y++)
  {
   for (x=0;x<self->width;x++)
   {
    cl_float * in = self->pixel_prob_temp + y*self->width + x;
    float * out = (float*)(pixProb->data + y*pixProb->strides[0] + x*pixProb->strides[1]);

    out[0] = in[0];
   }
  }


 //gettimeofday(&end, 0); // ************************************************************
 //double diff = (end.tv_sec + 1e-6*end.tv_usec) - (start.tv_sec + 1e-6*start.tv_usec);
 //printf("time for model = %.6f\n", diff); // ***********************************


 Py_INCREF(Py_None);
 return Py_None;
}



static PyObject * BackSubCoreDP_light_update(BackSubCoreDP * self, PyObject * args)
{
 // Extract the 3 parameters...
 float mult[3];
 if (!PyArg_ParseTuple(args, "fff", &mult[0], &mult[1], &mult[2])) return NULL;


 // Send a kernel going to update each component in turn...
  cl_int status = CL_SUCCESS;

  status |= clSetKernelArg(self->light_update_mix, 2, sizeof(cl_float4), self->prior_mu);
  status |= clSetKernelArg(self->light_update_mix, 3, sizeof(cl_float4), mult);

  if (status!=CL_SUCCESS) return NULL;

  size_t work_size[3];
  size_t block_size[3];

  work_size[0] = self->component_cap;
  work_size[1] = self->width;
  work_size[2] = self->height;
  calc_block_size(self->light_update_mix_size, 3, work_size, block_size, 1);
  status = clEnqueueNDRangeKernel(self->queue, self->light_update_mix, 3, NULL, work_size, block_size, 0, NULL, NULL);
  if (status!=CL_SUCCESS) return NULL;

  status = clEnqueueBarrier(self->queue);
  if (status!=CL_SUCCESS) return NULL;


 Py_INCREF(Py_None);
 return Py_None;
}



static PyObject * BackSubCoreDP_background(BackSubCoreDP * self, PyObject * args)
{
 // Get the output numpy array...
  PyArrayObject * image;
  if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &image)) return NULL;


 //struct timeval start, end; // ********************************************
 //gettimeofday(&start, 0); // **********************************************


 // Enqueue a task to put the background into the image field...
  cl_int status = clSetKernelArg(self->extract_mode, 2, sizeof(cl_float4), self->prior_mu);
  if (status!=CL_SUCCESS) {open_cl_error(status); return NULL;}

  size_t work_size[2];
  size_t block_size[2];

  work_size[0] = self->width;
  work_size[1] = self->height;
  calc_block_size(self->extract_mode_size, 2, work_size, block_size, 0);
  status = clEnqueueNDRangeKernel(self->queue, self->extract_mode, 2, NULL, work_size, block_size, 0, NULL, NULL);
  if (status!=CL_SUCCESS) {open_cl_error(status); return NULL;}

 // Add a barrier...
  status = clEnqueueBarrier(self->queue);
  if (status!=CL_SUCCESS) {open_cl_error(status); return NULL;}

 // Enqueue extracting the result...
  status = clEnqueueReadBuffer(self->queue, self->image, CL_FALSE, 0, self->height*self->width*4*sizeof(cl_float), self->image_temp, 0, NULL, NULL);
  if (status!=CL_SUCCESS) {open_cl_error(status); return NULL;}

 // Wait till the queue is done...
  status = clFinish(self->queue);
  if (status!=CL_SUCCESS) {open_cl_error(status); return NULL;}

 // Write the result into the output image...
  int y, x, i;
  for (y=0;y<self->height;y++)
  {
   for (x=0;x<self->width;x++)
   {
    cl_float * in = self->image_temp + (y*self->width + x)*4;
    float * out = (float*)(image->data + y*image->strides[0] + x*image->strides[1]);

    for (i=0;i<3;i++) out[i] = in[i];
   }
  }


 //gettimeofday(&end, 0); // ************************************************************
 //double diff = (end.tv_sec + 1e-6*end.tv_usec) - (start.tv_sec + 1e-6*start.tv_usec);
 //printf("time for background fetch = %.6f\n", diff); // ***********************************


 Py_INCREF(Py_None);
 return Py_None;
}



static PyObject * BackSubCoreDP_component_count(BackSubCoreDP * self, PyObject * args)
{
 // Get the output numpy array...
  PyArrayObject * image;
  if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &image)) return NULL;

 // Set parameters as required...
  cl_int status = CL_SUCCESS;
  
  status |= clSetKernelArg(self->extract_component_count, 2, sizeof(cl_float), &self->com_count_mass);
  status |= clSetKernelArg(self->extract_component_count, 3, sizeof(cl_float), &self->cap);
  
  if (status!=CL_SUCCESS) return NULL;

 // Enqueue a task to put the background into the image field...
  size_t work_size[2];
  size_t block_size[2];

  work_size[0] = self->width;
  work_size[1] = self->height;
  calc_block_size(self->extract_component_count_size, 2, work_size, block_size, 0);
  status = clEnqueueNDRangeKernel(self->queue, self->extract_component_count, 2, NULL, work_size, block_size, 0, NULL, NULL);
  if (status!=CL_SUCCESS) {open_cl_error(status); return NULL;}

 // Add a barrier...
  status = clEnqueueBarrier(self->queue);
  if (status!=CL_SUCCESS) {open_cl_error(status); return NULL;}

 // Enqueue extracting the result...
  status = clEnqueueReadBuffer(self->queue, self->image, CL_FALSE, 0, self->height*self->width*4*sizeof(cl_float), self->image_temp, 0, NULL, NULL);
  if (status!=CL_SUCCESS) {open_cl_error(status); return NULL;}

 // Wait till the queue is done...
  status = clFinish(self->queue);
  if (status!=CL_SUCCESS) {open_cl_error(status); return NULL;}

 // Write the result into the output image...
  int y, x, i;
  for (y=0;y<self->height;y++)
  {
   for (x=0;x<self->width;x++)
   {
    cl_float * in = self->image_temp + (y*self->width + x)*4;
    float * out = (float*)(image->data + y*image->strides[0] + x*image->strides[1]);

    for (i=0;i<3;i++) out[i] = in[i];
   }
  }

 Py_INCREF(Py_None);
 return Py_None;
}



static PyObject * BackSubCoreDP_make_mask(BackSubCoreDP * self, PyObject * args)
{
 // Get the input and output numpy arrays...
  PyArrayObject * image;
  PyArrayObject * pixProb;
  PyArrayObject * mask;
  if (!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, &image, &PyArray_Type, &pixProb, &PyArray_Type, &mask)) return NULL;


 //struct timeval start, end; // ********************************************
 //gettimeofday(&start, 0); // **********************************************


 // Some prep required for below - generate some values, set some kernel parameters...
  cl_int status = CL_SUCCESS;

  float priorOffset = log(1.0-self->threshold) - log(self->threshold);
  status |= clSetKernelArg(self->setup_model_bgCost, 1, sizeof(cl_float), &priorOffset);
  status |= clSetKernelArg(self->setup_model_bgCost, 2, sizeof(cl_float), &self->cert_limit);

  float target_dist = (self->half_life / self->min_same_prob) - self->half_life;
  status |= clSetKernelArg(self->setup_model_changeCost, 1, sizeof(cl_float), &self->half_life);
  status |= clSetKernelArg(self->setup_model_changeCost, 2, sizeof(cl_float), &target_dist);
  status |= clSetKernelArg(self->setup_model_changeCost, 3, sizeof(cl_float), &self->change_limit);
  status |= clSetKernelArg(self->setup_model_changeCost, 4, sizeof(cl_float), &self->change_mult);

  if (status!=CL_SUCCESS) {open_cl_error(status); return NULL;}


 // First copy the image and pixProb onto the graphics card - included for completeness, but currently commented out as this data should remain from the immediatly proceding call to calculate the probability map anyway...
  int i,y,x;
//   for (y=0;y<self->height;y++)
//   {
//    for (x=0;x<self->width;x++)
//    {
//     float * in = (float*)(image->data + y*image->strides[0] + x*image->strides[1]);
//     cl_float * out = self->image_temp + (y*self->width + x)*4;
//
//     for (i=0;i<3;i++) out[i] = in[i];
//     out[3] = 0.0;
//
//     in = (float*)(pixProb->data + y*pixProb->strides[0] + x*pixProb->strides[1]);
//     out = self->pixel_prob_temp + y*self->width + x;
//     out[0] = in[0];
//    }
//   }
//
//   status = clEnqueueWriteBuffer(self->queue, self->image, CL_FALSE, 0, self->height*self->width*4*sizeof(cl_float), self->image_temp, 0, NULL, NULL);
//   if (status!=CL_SUCCESS) {open_cl_error(status); return NULL;}
//
//   status = clEnqueueWriteBuffer(self->queue, self->pixel_prob, CL_FALSE, 0, self->height*self->width*sizeof(cl_float), self->pixel_prob_temp, 0, NULL, NULL);
//   if (status!=CL_SUCCESS) {open_cl_error(status); return NULL;}
//
//   status = clEnqueueBarrier(self->queue);
//   if (status!=CL_SUCCESS) {open_cl_error(status); return NULL;}


 // Fill in the costs at the lowest level, plus also zero out some stuff - two steps - bgCost and distances, followed by converting the distances into costs...
  size_t work_offset[2];
  size_t work_size[2];
  size_t block_size[2];

  
  work_offset[0] = 0;
  work_offset[1] = 0;
  work_size[0] = self->width;
  work_size[1] = self->height;
  calc_block_size(self->setup_model_bgCost_size, 2, work_size, block_size, 0);
  status = clEnqueueNDRangeKernel(self->queue, self->setup_model_bgCost, 2, work_offset, work_size, block_size, 0, NULL, NULL);
  if (status!=CL_SUCCESS) {open_cl_error(status); return NULL;}


  work_offset[0] = 0;
  work_offset[1] = 0;
  work_size[0] = self->width - 1;
  work_size[1] = self->height;
  calc_block_size(self->setup_model_dist_pos_x_size, 2, work_size, block_size, 0);
  status = clEnqueueNDRangeKernel(self->queue, self->setup_model_dist_pos_x, 2, work_offset, work_size, block_size, 0, NULL, NULL);
  if (status!=CL_SUCCESS) {open_cl_error(status); return NULL;}

  work_offset[0] = 0;
  work_offset[1] = 0;
  work_size[0] = self->width;
  work_size[1] = self->height - 1;
  calc_block_size(self->setup_model_dist_pos_y_size, 2, work_size, block_size, 0);
  status = clEnqueueNDRangeKernel(self->queue, self->setup_model_dist_pos_y, 2, work_offset, work_size, block_size, 0, NULL, NULL);
  if (status!=CL_SUCCESS) {open_cl_error(status); return NULL;}

  work_offset[0] = 1;
  work_offset[1] = 0;
  work_size[0] = self->width - 1;
  work_size[1] = self->height;
  calc_block_size(self->setup_model_dist_neg_x_size, 2, work_size, block_size, 0);
  status = clEnqueueNDRangeKernel(self->queue, self->setup_model_dist_neg_x, 2, work_offset, work_size, block_size, 0, NULL, NULL);
  if (status!=CL_SUCCESS) {open_cl_error(status); return NULL;}

  work_offset[0] = 0;
  work_offset[1] = 1;
  work_size[0] = self->width;
  work_size[1] = self->height - 1;
  calc_block_size(self->setup_model_dist_neg_y_size, 2, work_size, block_size, 0);
  status = clEnqueueNDRangeKernel(self->queue, self->setup_model_dist_neg_y, 2, work_offset, work_size, block_size, 0, NULL, NULL);
  if (status!=CL_SUCCESS) {open_cl_error(status); return NULL;}


  int d = 0;
  status = clSetKernelArg(self->setup_model_dist_boundary, 1, sizeof(cl_float), &d);
  if (status!=CL_SUCCESS) return NULL;
  work_offset[0] = self->width - 1;
  work_offset[1] = 0;
  work_size[0] = 1;
  work_size[1] = self->height;
  status = clEnqueueNDRangeKernel(self->queue, self->setup_model_dist_boundary, 2, work_offset, work_size, NULL, 0, NULL, NULL);
  if (status!=CL_SUCCESS) {open_cl_error(status); return NULL;}

  d = 1;
  status = clSetKernelArg(self->setup_model_dist_boundary, 1, sizeof(cl_float), &d);
  if (status!=CL_SUCCESS) {open_cl_error(status); return NULL;}
  work_offset[0] = 0;
  work_offset[1] = self->height - 1;
  work_size[0] = self->width;
  work_size[1] = 1;
  status = clEnqueueNDRangeKernel(self->queue, self->setup_model_dist_boundary, 2, work_offset, work_size, NULL, 0, NULL, NULL);
  if (status!=CL_SUCCESS) {open_cl_error(status); return NULL;}

  d = 2;
  status = clSetKernelArg(self->setup_model_dist_boundary, 1, sizeof(cl_float), &d);
  if (status!=CL_SUCCESS) {open_cl_error(status); return NULL;}
  work_offset[0] = 0;
  work_offset[1] = 0;
  work_size[0] = 1;
  work_size[1] = self->height;
  status = clEnqueueNDRangeKernel(self->queue, self->setup_model_dist_boundary, 2, work_offset, work_size, NULL, 0, NULL, NULL);
  if (status!=CL_SUCCESS) {open_cl_error(status); return NULL;}

  d = 3;
  status = clSetKernelArg(self->setup_model_dist_boundary, 1, sizeof(cl_float), &d);
  if (status!=CL_SUCCESS) {open_cl_error(status); return NULL;}
  work_offset[0] = 0;
  work_offset[1] = 0;
  work_size[0] = self->width;
  work_size[1] = 1;
  status = clEnqueueNDRangeKernel(self->queue, self->setup_model_dist_boundary, 2, work_offset, work_size, NULL, 0, NULL, NULL);
  if (status!=CL_SUCCESS) {open_cl_error(status); return NULL;}


  status = clEnqueueBarrier(self->queue);
  if (status!=CL_SUCCESS) {open_cl_error(status); return NULL;}


  work_size[0] = self->width;
  work_size[1] = self->height;
  calc_block_size(self->setup_model_changeCost_size, 2, work_size, block_size, 0);
  status = clEnqueueNDRangeKernel(self->queue, self->setup_model_changeCost, 2, NULL, work_size, block_size, 0, NULL, NULL);
  if (status!=CL_SUCCESS) {open_cl_error(status); return NULL;}


  int l;
  work_offset[0] = 0;
  work_offset[1] = 0;
  for (l=0;l<self->layers;l++)
  {
   work_size[0] = self->widthBL[l];
   work_size[1] = self->heightBL[l];

   status |= clSetKernelArg(self->reset_in, 0, sizeof(cl_int), &self->widthBL[l]);
   status |= clSetKernelArg(self->reset_in, 1, sizeof(cl_mem), &self->msgIn[l]);

   if (status!=CL_SUCCESS) {open_cl_error(status); return NULL;}

   calc_block_size(self->reset_in_size, 2, work_size, block_size, 0);
   status = clEnqueueNDRangeKernel(self->queue, self->reset_in, 2, work_offset, work_size, block_size, 0, NULL, NULL);
   if (status!=CL_SUCCESS) {open_cl_error(status); return NULL;}
  }


  status = clEnqueueBarrier(self->queue);
  if (status!=CL_SUCCESS) {open_cl_error(status); return NULL;}


 // Downsample the costs to the highest level of the processing hierarchy...
  for (l=1;l<self->layers;l++)
  {
   status |= clSetKernelArg(self->downsample_model, 0, sizeof(cl_int), &self->widthBL[l-1]);
   status |= clSetKernelArg(self->downsample_model, 1, sizeof(cl_mem), &self->bgCost[l-1]);
   status |= clSetKernelArg(self->downsample_model, 2, sizeof(cl_mem), &self->changeCost[l-1]);
   status |= clSetKernelArg(self->downsample_model, 3, sizeof(cl_int), &self->widthBL[l]);
   status |= clSetKernelArg(self->downsample_model, 4, sizeof(cl_mem), &self->bgCost[l]);
   status |= clSetKernelArg(self->downsample_model, 5, sizeof(cl_mem), &self->changeCost[l]);

   if (status!=CL_SUCCESS) {open_cl_error(status); return NULL;}

   work_size[0] = self->widthBL[l];
   work_size[1] = self->heightBL[l];
   calc_block_size(self->downsample_model_size, 2, work_size, block_size, 0);
   status = clEnqueueNDRangeKernel(self->queue, self->downsample_model, 2, NULL, work_size, block_size, 0, NULL, NULL);
   if (status!=CL_SUCCESS) {open_cl_error(status); return NULL;}

   status = clEnqueueBarrier(self->queue);
   if (status!=CL_SUCCESS) {open_cl_error(status); return NULL;}
  }


 // Iterate over each level of the hierarchy in turn, doing the iterations and then upsampling to the next layer down...
  for (l=self->layers-1;l>=0;l--)
  {
   //printf("layer = %i; width = %i; height = %il; widthBL = %i; heightBL = %i\n", l, self->width, self->height, self->widthBL[l], self->heightBL[l]); 
    
   // Prepare the kernels for the iterations...
    status |= clSetKernelArg(self->send_pos_x, 0, sizeof(cl_int), &self->widthBL[l]);
    status |= clSetKernelArg(self->send_pos_x, 1, sizeof(cl_mem), &self->bgCost[l]);
    status |= clSetKernelArg(self->send_pos_x, 2, sizeof(cl_mem), &self->changeCost[l]);
    status |= clSetKernelArg(self->send_pos_x, 3, sizeof(cl_mem), &self->msgIn[l]);

    status |= clSetKernelArg(self->send_pos_y, 0, sizeof(cl_int), &self->widthBL[l]);
    status |= clSetKernelArg(self->send_pos_y, 1, sizeof(cl_mem), &self->bgCost[l]);
    status |= clSetKernelArg(self->send_pos_y, 2, sizeof(cl_mem), &self->changeCost[l]);
    status |= clSetKernelArg(self->send_pos_y, 3, sizeof(cl_mem), &self->msgIn[l]);

    status |= clSetKernelArg(self->send_neg_x, 0, sizeof(cl_int), &self->widthBL[l]);
    status |= clSetKernelArg(self->send_neg_x, 1, sizeof(cl_mem), &self->bgCost[l]);
    status |= clSetKernelArg(self->send_neg_x, 2, sizeof(cl_mem), &self->changeCost[l]);
    status |= clSetKernelArg(self->send_neg_x, 3, sizeof(cl_mem), &self->msgIn[l]);

    status |= clSetKernelArg(self->send_neg_y, 0, sizeof(cl_int), &self->widthBL[l]);
    status |= clSetKernelArg(self->send_neg_y, 1, sizeof(cl_mem), &self->bgCost[l]);
    status |= clSetKernelArg(self->send_neg_y, 2, sizeof(cl_mem), &self->changeCost[l]);
    status |= clSetKernelArg(self->send_neg_y, 3, sizeof(cl_mem), &self->msgIn[l]);

    if (status!=CL_SUCCESS) {open_cl_error(status); return NULL;}

   // Do the iterations...
    int iters = self->itersPerLevel;
    if (l==0) iters = self->iterations;
    for (i=0;i<iters;i++)
    {
     status |= clSetKernelArg(self->send_pos_x, 4, sizeof(cl_int), &i);
     status |= clSetKernelArg(self->send_pos_y, 4, sizeof(cl_int), &i);
     status |= clSetKernelArg(self->send_neg_x, 4, sizeof(cl_int), &i);
     status |= clSetKernelArg(self->send_neg_y, 4, sizeof(cl_int), &i);
     if (status!=CL_SUCCESS) {open_cl_error(status); return NULL;}
     
     work_offset[0] = 0;
     work_offset[1] = 0;
     work_size[0] = self->widthBL[l] - 1;
     work_size[1] = self->heightBL[l];
     calc_block_size(self->send_pos_x_size, 2, work_size, block_size, 0);
     status = clEnqueueNDRangeKernel(self->queue, self->send_pos_x, 2, work_offset, work_size, block_size, 0, NULL, NULL);
     if (status!=CL_SUCCESS) {open_cl_error(status); return NULL;}

     work_offset[0] = 0;
     work_offset[1] = 0;
     work_size[0] = self->widthBL[l];
     work_size[1] = self->heightBL[l] - 1;
     calc_block_size(self->send_pos_y_size, 2, work_size, block_size, 0);
     status = clEnqueueNDRangeKernel(self->queue, self->send_pos_y, 2, work_offset, work_size, block_size, 0, NULL, NULL);
     if (status!=CL_SUCCESS) {open_cl_error(status); return NULL;}

     work_offset[0] = 1;
     work_offset[1] = 0;
     work_size[0] = self->widthBL[l] - 1;
     work_size[1] = self->heightBL[l];
     calc_block_size(self->send_neg_x_size, 2, work_size, block_size, 0);
     status = clEnqueueNDRangeKernel(self->queue, self->send_neg_x, 2, work_offset, work_size, block_size, 0, NULL, NULL);
     if (status!=CL_SUCCESS) {open_cl_error(status); return NULL;}

     work_offset[0] = 0;
     work_offset[1] = 1;
     work_size[0] = self->widthBL[l];
     work_size[1] = self->heightBL[l] - 1;
     calc_block_size(self->send_neg_y_size, 2, work_size, block_size, 0);
     status = clEnqueueNDRangeKernel(self->queue, self->send_neg_y, 2, work_offset, work_size, block_size, 0, NULL, NULL);
     if (status!=CL_SUCCESS) {open_cl_error(status); return NULL;}


     status = clEnqueueBarrier(self->queue);
     if (status!=CL_SUCCESS) {open_cl_error(status); return NULL;}
    }

   // Upsample the messages to the higher resolution...
    if (l!=0)
    {
     status |= clSetKernelArg(self->upsample_messages, 0, sizeof(cl_int), &self->widthBL[l]);
     status |= clSetKernelArg(self->upsample_messages, 1, sizeof(cl_mem), &self->msgIn[l]);
     status |= clSetKernelArg(self->upsample_messages, 2, sizeof(cl_int), &self->widthBL[l-1]);
     status |= clSetKernelArg(self->upsample_messages, 3, sizeof(cl_mem), &self->msgIn[l-1]);

     if (status!=CL_SUCCESS) {open_cl_error(status); return NULL;}

     work_size[0] = self->widthBL[l-1];
     work_size[1] = self->heightBL[l-1];
     calc_block_size(self->upsample_messages_size, 2, work_size, block_size, 0);
     status = clEnqueueNDRangeKernel(self->queue, self->upsample_messages, 2, NULL, work_size, block_size, 0, NULL, NULL);
     if (status!=CL_SUCCESS) return NULL;

     status = clEnqueueBarrier(self->queue);
     if (status!=CL_SUCCESS) {open_cl_error(status); return NULL;}
    }
  }


 // Generate the mask and copy it off the device, to be stuck in the return...
  work_offset[0] = 0;
  work_offset[1] = 0;
  work_size[0] = self->width;
  work_size[1] = self->height;
  calc_block_size(self->calc_mask_size, 2, work_size, block_size, 0);
  status = clEnqueueNDRangeKernel(self->queue, self->calc_mask, 2, work_offset, work_size, block_size, 0, NULL, NULL);
  if (status!=CL_SUCCESS) {open_cl_error(status); return NULL;}


  status = clEnqueueBarrier(self->queue);
  if (status!=CL_SUCCESS) {open_cl_error(status); return NULL;}


  status = clEnqueueReadBuffer(self->queue, self->mask, CL_FALSE, 0, self->height*self->width*sizeof(cl_char), self->mask_temp, 0, NULL, NULL);
  if (status!=CL_SUCCESS) {open_cl_error(status); return NULL;}


  status = clFinish(self->queue);
  if (status!=CL_SUCCESS) {open_cl_error(status); return NULL;}


  for (y=0;y<self->height;y++)
  {
   for (x=0;x<self->width;x++)
   {
    cl_char * in = self->mask_temp + y*self->width + x;
    char * out = (char*)(mask->data + y*mask->strides[0] + x*mask->strides[1]);

    out[0] = in[0];
   }
  }


 //gettimeofday(&end, 0); // ************************************************************
 //double diff = (end.tv_sec + 1e-6*end.tv_usec) - (start.tv_sec + 1e-6*start.tv_usec);
 //printf("time for post process = %.6f\n", diff); // ***********************************


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
 {"component_count", (PyCFunction)BackSubCoreDP_component_count, METH_VARARGS, "Given an output float32 rgb array this fills it with an estimate of the number of components for each pixel, or at least how many it takes to get to com_count_mass of the probability. Note that it assumes they are ordered largest to smallest, which is statisticaly likelly but hardly guaranteed."},
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



static PyMethodDef backsub_dp_cl_methods[] =
{
 {NULL}
};



#ifndef PyMODINIT_FUNC
#define PyMODINIT_FUNC void
#endif

PyMODINIT_FUNC initbacksub_dp_cl(void)
{
 PyObject * mod = Py_InitModule3("backsub_dp_cl", backsub_dp_cl_methods, "Provides a background subtraction algorithm based on a Dirichlet process Gaussian mixture model for each pixel, OpenCL version.");
 import_array();

 if (PyType_Ready(&BackSubCoreDPType) < 0) return;

 Py_INCREF(&BackSubCoreDPType);
 PyModule_AddObject(mod, "BackSubCoreDP", (PyObject*)&BackSubCoreDPType);
}
