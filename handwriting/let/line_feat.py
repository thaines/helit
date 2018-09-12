# Copyright 2016 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import numpy

try:
  from scipy import weave
except ImportError:
  import weave

from utils.start_cpp import start_cpp



def calc_radius(mask, thin_mask, max_radius = 48.0, on_true = 1.0, on_false = -128.0):
  """Given two 2D boolean numpy arrays - the first represents a mask of line regions, the second the thined version of this mask, which is the locations where a radius needs to be calculated. It returns an identically sized array of floats, with a radius at each location where thin_mask is True. The radius is selected to fill the mask region as best as possible - essentially finds the maxima on a graph which plots the score of each radius - plus 'on_true' for each pixel that is True in the mask, plus 'on_false' for each that is False. A maixmum radius to consider when plotting this graph is provided."""
  
  radius = numpy.zeros(mask.shape, dtype=numpy.float32)
  
  support_code = start_cpp() + """
  int compare_int(const void * lhs, const void * rhs)
  {
   int l = *(const int*)lhs;
   int r = *(const int*)rhs;
   
   if (l<r) return -1;
   else return 1;
  }
  """
  
  code = start_cpp(support_code) + """
  // Create some variables we need...
   int max_rad = int(ceil(max_radius));
   float max_rad_sqr = max_radius * max_radius;
   
  // Create an array of offsets from a point that is sorted by distance....
   int * offsets = (int*)malloc((max_rad*2+1) * (max_rad*2+1) * 3 * sizeof(int));
   int count = 0;
   
   int v, u;
   for (int v=-max_rad; v<=max_rad; v++)
   {
    for (int u=-max_rad; u<=max_rad; u++)
    {
     int dist = v*v + u*u;
     if (dist < max_rad_sqr)
     {
      offsets[3*count+0] = dist;
      offsets[3*count+1] = v;
      offsets[3*count+2] = u;
      count += 1;
     }
    }
   }
   
   qsort(offsets, count, 3*sizeof(int), &compare_int);
  
  // Iterate and process every pixel in the thin_mask...
   int y, x;
   for (y=0; y<Nmask[0]; y++)
   {
    for (x=0; x<Nmask[1]; x++)
    {
     if (THIN_MASK2(y,x)!=0)
     {
      // Go through the offsets in increasing distance, keeping track of the cost of the radius at each point, and its maximum...
       float best = 0.0;
       float best_r_sqr = 0.25;
      
       int i;
       float cum = 0.0;
       for (i=0; i<count; i++)
       {
        int ly = y + offsets[3*i+1];
        int lx = x + offsets[3*i+2];
        
        if ((ly>=0)&&(ly<Nmask[0])&&(lx>=0)&&(lx<Nmask[1]))
        {
         cum += (MASK2(ly,lx)!=0) ? on_true : on_false;
         
         if (cum>best)
         {
          best = cum;
          best_r_sqr = offsets[3*i+0];
         }
        }
       }
      
      // Set the radius as the one that gets the highest score...
       RADIUS2(y,x) = sqrt(best_r_sqr);
     }
    }
   }
  
  // Clean up...
   free(offsets);
  """
  
  weave.inline(code, ['mask', 'thin_mask', 'radius', 'max_radius', 'on_true', 'on_false'], support_code=support_code)
  
  return radius



def global_colour_histogram(image, mask, size=256):
  """Given an image ([0:height, 0:width, 0:3 - r,g,b] -> uint8) and a mask ([0:height, 0:width] -> bool.) this returns a histogram of how many in each colour bin belongs to the foreground and background. Return is [0:size - r, 0:size - g, 0:size - b, 0:2 - bg,fg] -> float32. For the default size of 256 this is 128 meg, so be nice! Note that if pixels don't land exactly into a bin they will be linearly interpolated, hence the use of float."""
  
  ret = numpy.zeros((size, size, size, 2), dtype=numpy.float32)
  
  code = start_cpp() + """
  float mult_r = (size-1) / 255.0;
  float mult_g = (size-1) / 255.0;
  float mult_b = (size-1) / 255.0;
  
  int y, x;
  for (y=0; y<Nimage[0]; y++)
  {
   for (x=0; x<Nimage[1]; x++)
   {
    float r = mult_r * IMAGE3(y, x, 0);
    float g = mult_g * IMAGE3(y, x, 1);
    float b = mult_b * IMAGE3(y, x, 2);
    
    int lr = int(floor(r));
    int lg = int(floor(g));
    int lb = int(floor(b));
    
    float tr = r - lr;
    float tg = g - lg;
    float tb = b - lb;
    
    int hr = int(ceil(r));
    int hg = int(ceil(g));
    int hb = int(ceil(b));
    
    if (hr>=size) hr = size-1;
    if (hg>=size) hg = size-1;
    if (hb>=size) hb = size-1;
    
    char ci = (MASK2(y, x)!=0) ? 1 : 0;
    
    RET4(lr, lg, lb, ci) += (1.0-tr) * (1.0-tg) * (1.0-tb);
    
    RET4(hr, lg, lb, ci) += tr * (1.0-tg) * (1.0-tb);
    RET4(lr, hg, lb, ci) += (1.0-tr) * tg * (1.0-tb);
    RET4(lr, lg, hb, ci) += (1.0-tr) * (1.0-tg) * tb;
    
    RET4(lr, hg, hb, ci) += (1.0-tr) * tg * tb;
    RET4(hr, lg, hb, ci) += tr * (1.0-tg) * tb;
    RET4(hr, hg, lb, ci) += tr * tg * (1.0-tb);
    
    RET4(hr, hg, hb, ci) += tr * tg * tb;
   }
  }
  """
  
  weave.inline(code, ['ret', 'image', 'mask'])
  
  return ret



def calc_average(image, thin_mask, radius):
  """Calculates the average colour for all pixels in a mask (thin_mask), outputing a new image of average colours, at the mask points only. Also requires a radius mask, that defines the radius of the circle to average over for each pixel."""
  
  ret = numpy.zeros(image.shape, dtype=numpy.float32)
  
  code = start_cpp() + """
  // Loop and process each pixel in turn...
   int y, x;
   for (y=0; y<Nimage[0]; y++)
   {
    for (x=0; x<Nimage[1]; x++)
    {
     if (THIN_MASK2(y,x)!=0)
     {
      // Work out the range of values to consider...
       float radius = RADIUS2(y,x);
       int rad = int(ceil(radius));
       float radius_sqr = radius * radius;
       
       int low_v  = y - rad;
       int high_v = y + rad;
       int low_u  = x - rad;
       int high_u = x + rad;
       
       if (low_v<0) low_v = 0;
       if (high_v>=Nimage[0]) high_v = Nimage[0] - 1;
       if (low_u<0) low_u = 0;
       if (high_u>=Nimage[1]) high_u = Nimage[1] - 1;
       
      // Loop the range and average the relevant pixels...
       float col[3] = {0.0, 0.0, 0.0};
       float weight = 0.0;
       
       int v, u;
       for (v=low_v; v<=high_v; v++)
       {
        for (u=low_u; u<=high_u; u++)
        {
         int dy = v - y;
         int dx = u - x;
         int dist_sqr = dy*dy + dx*dx;
         
         if (dist_sqr < radius_sqr)
         {
          weight += 1.0;
          col[0] += (IMAGE3(v,u,0) - col[0]) / weight;
          col[1] += (IMAGE3(v,u,1) - col[1]) / weight;
          col[2] += (IMAGE3(v,u,2) - col[2]) / weight;
         }
        }
       }
       
      // Record the final value...
       RET3(y,x,0) = col[0];
       RET3(y,x,1) = col[1];
       RET3(y,x,2) = col[2];
     }
    }
   }
  """
  
  weave.inline(code, ['ret', 'image', 'thin_mask', 'radius'])
  
  return ret



def apply_tps(average_image, thin_mask, tps):
  """Given an image of average colours and a thin plate spline (tps) this returns a floating point map aligned with the image where the thin plate spline has been applied to every pixel that is in the thin_mask variable."""
  index = thin_mask==True
  dm = average_image[index,:]
  
  values = tps(dm)
  
  ret = numpy.zeros(average_image.shape[:2], dtype=numpy.float32)
  ret[index] = values
  return ret



def apply_tps_all(image, tps):
  """Given an image this applies the thin plate spline to every pixels colour, returning an answer map."""
  
  # Convert the image into a data matrix...
  dm = image.reshape((-1,3))
  
  # Compress the data matrix down, to remove duplicates...
  index = numpy.lexsort(dm.T)
  dm = dm[index,:]
  keep = numpy.ones(dm.shape[0], dtype=numpy.bool)
  keep[1:] = (numpy.diff(dm, axis=0)!=0).any(axis=1)
  dm = dm[keep]

  # Run for all unique colours...
  out = numpy.empty(dm.shape[0], dtype=numpy.float32)
  step = 1024 * 1024
  for i in xrange(0,out.shape[0],step):
    out[i:i+step] = tps(dm[i:i+step,:].astype(numpy.float32))
  
  # Blow the data matrix up to its original size...
  source = numpy.cumsum(keep) - 1
  out = out[source]
  out = out[numpy.argsort(index)]
  
  # Return, image shaped...
  return out.reshape(image.shape[:2])

