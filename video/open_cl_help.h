#ifndef OPENCL_HELP
#define OPENCL_HELP

#include <fcntl.h>
#include <string.h>
#include <sys/time.h>

#include <Python.h>
#include <CL/cl.h>


// Sets up python to give a reasonable exception given a OpenCL error...
void open_cl_error(cl_int status)
{
 const char * msg = "OpenCL: Unknown error";
 switch(status)
 {
  case CL_INVALID_COMMAND_QUEUE:
   msg = "OpenCL: Invalid command queue";
  break;

  case CL_INVALID_CONTEXT:
   msg = "OpenCL: Invalid context";
  break;

  case CL_INVALID_MEM_OBJECT:
   msg = "OpenCL: Invalid memory object";
  break;

  case CL_INVALID_VALUE:
   msg = "OpenCL: Invalid value";
  break;

  case CL_INVALID_EVENT_WAIT_LIST:
   msg = "OpenCL: Invalid event wait list";
  break;

  case CL_MEM_OBJECT_ALLOCATION_FAILURE:
   msg = "OpenCL: Memory object allocation failure";
  break;

  case CL_OUT_OF_HOST_MEMORY:
   msg = "OpenCL: Out of host memory";
  break;

  case CL_INVALID_PROGRAM_EXECUTABLE:
   msg = "OpenCL: Invalid program executable";
  break;

  case CL_INVALID_KERNEL:
   msg = "OpenCL: Invalid kernel";
  break;

  case CL_INVALID_KERNEL_ARGS:
   msg = "OpenCL: Invalid kernel arguments";
  break;

  case CL_INVALID_WORK_DIMENSION:
   msg = "OpenCL: Invalid work dimension";
  break;

  case CL_INVALID_GLOBAL_WORK_SIZE:
   msg = "OpenCL: Invalid global work size";
  break;

  case CL_INVALID_GLOBAL_OFFSET:
   msg = "OpenCL: Invalid global offset";
  break;

  case CL_INVALID_WORK_GROUP_SIZE:
   msg = "OpenCL: Invalid work group size";
  break;

  case CL_INVALID_WORK_ITEM_SIZE:
   msg = "OpenCL: Invalid work item size";
  break;

  case CL_MISALIGNED_SUB_BUFFER_OFFSET:
   msg = "OpenCL: Misaligned sub-buffer offset";
  break;

  case CL_INVALID_IMAGE_SIZE:
   msg = "OpenCL: Invalid image size";
  break;

  case CL_OUT_OF_RESOURCES:
   msg = "OpenCL: Out of resources";
  break;
 }

 PyErr_SetString(PyExc_RuntimeError, msg);
}



// Builds and compiles an openCL program from a file - returns NULL if their is an error, a compiled program if not...
inline cl_program prep_program(const char * path, const char * file, cl_context context, cl_device_id device)
{
 cl_int status;

 // Generate full filename...
  char * fn = malloc(strlen(path)+strlen(file)+2);
  strcpy(fn, path);
  strcat(fn, "/");
  strcat(fn, file);

 // Open file...
  int fin = open(fn, O_RDONLY);
  free(fn);
  if (fin<0) return NULL;

 // Get its length...
  int cl_size = lseek(fin, 0, SEEK_END);
  lseek(fin, 0, SEEK_SET);
  if (cl_size<1)
  {
   close(fin);
   return NULL;
  }

 // Load the data...
  char * cl_code = malloc((cl_size+1) * sizeof(char));
  if (read(fin, cl_code, cl_size)!=cl_size)
  {
   close(fin);
   free(cl_code);
   return NULL;
  }
  cl_code[cl_size] = 0; // Null terminator.
  close(fin);

 // Create the program...
  cl_program program = clCreateProgramWithSource(context, 1, (const char**)&cl_code, NULL, &status);
  free(cl_code);
  if (status!=CL_SUCCESS) return NULL;

 // Build...
  status = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (status!=CL_SUCCESS)
  {
   size_t errSize;
   if (clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &errSize)==CL_SUCCESS)
   {
    char * errBuf = (char*)malloc(errSize);
    if (clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, errSize, errBuf, NULL)==CL_SUCCESS)
    {
     printf(errBuf);
    }
    free(errBuf);
   }

   clReleaseProgram(program);
   return NULL;
  }

 return program;
}



// Little helper function - given a block size and a width it returns the width increased so it is a multiple of the block size...
inline int make_multiple(int width, int block_size)
{
 if ((width%block_size)==0) return width;
 int mult = width/block_size;
 mult += 1;
 return mult*block_size;
}

// Helper function - given the number of dimensions and two pointers, to the global problem size, which is filled in, and to the local block size, it attempts to fill in the latter so the multiplication of terms equals scale (Which must be 2^n), following the rules of OpenCL. It has a preference towards earlier entrys for memory coherance. Additionally if one of the dimensions in the global size can be increased without problem that can be indicated by overrun_safe being set to the relevant index...
inline void calc_block_size(int scale, int dims, size_t * global, size_t * local, int overrun_safe)
{
 int i;
 // Set all entrys in local to 1...
  for (i=0;i<dims;i++) local[i] = 1;

 // Find out how high we are going in the next step...
  int cap = dims;
  if (overrun_safe>=0) cap = overrun_safe;

 // For each dimension set their local value to the largest 2^n value which divides through...
  for (i=0;i<cap;i++)
  {
   while ((local[i]&global[i])==0) local[i] *= 2;
  }

 // Eat up the scale...
  for (i=0;i<cap;i++)
  {
   if (scale>local[i]) scale /= local[i];
   else
   {
    local[i] = scale;
    scale = 1;
   }
  }

 // If there is any left and we are at the overun use it to acheive scale...
  if ((scale>1)&&(overrun_safe>=0))
  {
   local[overrun_safe] = scale;
   global[overrun_safe] = make_multiple(global[overrun_safe], scale);
  }
}

#endif
