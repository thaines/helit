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



class ThresholdLine:
  """Given a probability of belonging to a line for each pixel in an image this returns a mask of line/not line. Actual inference is quite sophisticated - it works in signed distance to the edge of a line space, and optimises for consistancy with a regularisation cost that assumes a shared line being the closest for each small neighbourhood. It minimises this cost using Nesterov's method, initialised based on a simple threshold of the input probabilites. Has a bunch of parameters."""
  def __init__(self):
    self.limit = 0.1 # Limit of how extreme the probability map can get.
    self.threshold = 0.25 # Threshold for initial assignment to line.
    
    self.iters = 64
    self.radius = 8.0 # Only analyse pixels within this radius of a line.
    self.step_size = 0.1
    self.unary_sharpness = 16.0
    self.tertiary_grad = 16.0
  
  def mask_to_sdf(self, mask):
    # Initalise with effective infinities...
    sdf = numpy.empty(mask.shape, dtype=numpy.float32)
    sdf[:,:] = 1e64
  
    # Mark all pixels that are at a transition boundary with the relevant cost - first the diagonals, then the halfs, as half is less than sqrt(2)...
    tran_sqrt2 = numpy.zeros(sdf.shape, dtype=numpy.bool)
    numpy.logical_or(mask[1:,1:]!=mask[:-1,:-1], tran_sqrt2[:-1,:-1], tran_sqrt2[:-1,:-1])
    numpy.logical_or(mask[1:,:-1]!=mask[:-1,1:], tran_sqrt2[:-1,1:], tran_sqrt2[:-1,1:])
    numpy.logical_or(mask[:-1,1:]!=mask[1:,:-1], tran_sqrt2[1:,:-1], tran_sqrt2[1:,:-1])
    numpy.logical_or(mask[:-1,:-1]!=mask[1:,1:], tran_sqrt2[1:,1:], tran_sqrt2[1:,1:])
    sdf[tran_sqrt2] = numpy.sqrt(2.0)
  
    tran_half = numpy.zeros(sdf.shape, dtype=numpy.bool)
    numpy.logical_or(mask[1:,:]!=mask[:-1,:], tran_half[:-1,:], tran_half[:-1,:])
    numpy.logical_or(mask[:-1,:]!=mask[1:,:], tran_half[1:,:], tran_half[1:,:])
    numpy.logical_or(mask[:,1:]!=mask[:,:-1], tran_half[:,:-1], tran_half[:,:-1])
    numpy.logical_or(mask[:,:-1]!=mask[:,1:], tran_half[:,1:], tran_half[:,1:])
    sdf[tran_half] = 0.5
  
    # Do all 8 directions of sweep iterativly until distances stop getting smaller...
    stop = False
    while not stop:
      stop = True
    
      code = start_cpp() + """
      float sqrt2 = sqrt(2.0);
    
      // Forwards pass...
       for (int y=0; y<Nsdf[0]; y++)
       {
        for (int x=0; x<Nsdf[1]; x++)
        {
         bool negx = x!=0;
         bool negy = y!=0;
       
         if ((negx)&&((SDF2(y, x-1)+1.0)<SDF2(y, x)))
         {
          SDF2(y, x) = SDF2(y, x-1) + 1.0;
          stop = false;
         }
       
         if ((negy)&&((SDF2(y-1, x)+1.0)<SDF2(y, x)))
         {
          SDF2(y, x) = SDF2(y-1, x) + 1.0;
          stop = false;
         }
       
         if ((negx)&&(negy)&&((SDF2(y-1, x-1)+sqrt2)<SDF2(y, x)))
         {
          SDF2(y, x) = SDF2(y-1, x-1) + sqrt2;
          stop = false;
         }
        }
       }
    
      // Backwards pass...
       for (int y=Nsdf[0]-1; y>=0; y--)
       {
        for (int x=Nsdf[1]-1; x>=0; x--)
        {
         bool posx = (x+1)!=Nsdf[1];
         bool posy = (y+1)!=Nsdf[0];
       
         if ((posx)&&((SDF2(y, x+1)+1.0)<SDF2(y, x)))
         {
          SDF2(y, x) = SDF2(y, x+1) + 1.0;
          stop = false;
         }
       
         if ((posy)&&((SDF2(y+1, x)+1.0)<SDF2(y, x)))
         {
          SDF2(y, x) = SDF2(y+1, x) + 1.0;
          stop = false;
         }
       
         if ((posx)&&(posy)&&((SDF2(y+1, x+1)+sqrt2)<SDF2(y, x)))
         {
          SDF2(y, x) = SDF2(y+1, x+1) + sqrt2;
          stop = false;
         }
        }
       }
      """

      weave.inline(code, ['sdf', 'stop'])
  
    # Add in the sign - negate all pixels that are within the mask, and return...
    sdf[mask] *= -1.0
    return sdf
  
  
  def cost(self, cost, sdf, analyse, grad):
    """Given the cost difference (negative log probability) of choosing the pixel belongs to a line over does not belong to a line, a signed distance function and an output into which it will write the gradient of the cost this returns the cost of the curretn sdf, with the gradient of the cost written into grad."""
    support = start_cpp() + """
    int comp_float(const void * a, const void * b)
    {
     float fa = *(float*)a;
     float fb = *(float*)b;
     
     if (fa<fb) return -1;
     if (fb<fa) return 1;
     return 0;
    }
    """
    
    code = start_cpp() + """
    float ret = 0.0;
    
    // Zero out the gradiant output...
     for (int y=0; y<Nsdf[0]; y++)
     {
      for (int x=0; x<Nsdf[1]; x++)
      {
       GRAD2(y, x) = 0.0;
      }
     }
    
    // First go through and add in the unary cost for each pixel - we smooth it at the boundary using a sigmoid, as that helps the solver, represents the fact we expect there to be a little noise and keeps it differentiable...
     for (int y=0; y<Nsdf[0]; y++)
     {
      for (int x=0; x<Nsdf[1]; x++)
      {
       if (ANALYSE2(y, x)==0) continue;
       
       float t = -unary_sharpness * SDF2(y, x);
       if (t<-16.0) t = -16.0; // Avoid really extreme values.
       if (t>16.0)  t = 16.0;  // "
       
       float val = 1.0 / (1.0 + exp(-t));
       ret += COST2(y, x) * val;
       GRAD2(y, x) += -COST2(y, x) * unary_sharpness * val * (1.0-val);
      }
     }
     
    // Now do the neighbour costs - use a robust mechanism to infer the correct orientation for the line associated (e.g. tangent at nearest point) with each pixel and its distance, and then peanalise deviation from that distance...
     static const char dx[8] = {-1,  0,  1, 1, 1, 0, -1, -1};
     static const char dy[8] = {-1, -1, -1, 0, 1, 1,  1,  0};
     static const float div[8] = {sqrt(2), 1, sqrt(2), 1, sqrt(2), 1, sqrt(2), 1};
     const float tgs = tertiary_grad * tertiary_grad;
    
     for (int y=1; y<Nsdf[0]-1; y++)
     {
      for (int x=1; x<Nsdf[1]-1; x++)
      {
       // Skip pixels that are not close enough to a line to matter...
        if (ANALYSE2(y, x)==0) continue;
        
       // Estimate the 'correct' direction and then distance to the closest line of this pixel, in a super-robust, if expensive, way...
        float bestMedian = SDF2(y, x);
        float bestMAD = 1e64;
       
        for (int ni=0; ni<8; ni++)
        {
         // Estimate the perpendicular to the line direction from the 3 neighbours under consideration - a maximum liklihood mean direction of a Fisher distribution (Which for two pixels would be the maximum liklihood fit)...
          float nx = 0.0;
          float ny = 0.0;

          bool oob = false; // Skip set if one of the pixels is not of interest.
          for (int oi=0; oi<3; oi++)
          {
           int i = (ni+oi) % 8;
           
           if (ANALYSE2(y+dy[i], x+dx[i])==0)
           {
            oob = true;
            break;
           }
          
           float l = SDF2(y + dy[i], x + dx[i]) - SDF2(y, x);
           l /= div[i];
          
           nx += l * dx[i];
           ny += l * dy[i];
          }
          if (oob) continue;
         
         // Normalise the perpendicular to the projection line...
          float len = sqrt(nx*nx + ny*ny);
          if (len<1e-3) continue;
          nx /= len;
          ny /= len;
          
         // Use the proposed line to calculate all 8 estimates...
          float e[8];
          for (int i=0; i<8; i++)
          {
           float dot = nx * dx[i] + ny * dy[i];
           e[i] = SDF2(y + dy[i], x + dx[i]) + dot;
          }
          
         // Use the estimates to calculate the median...
          qsort(e, 8, sizeof(float), comp_float);
          float median = 0.5 * (e[3] + e[4]);
         
         // Mess around and then calculate the MAD...
          for (int i=0; i<8; i++)
          {
           e[i] = fabs(e[i] - median);
          }
          
          qsort(e, 8, sizeof(float), comp_float);
          float MAD = 0.5 * (e[3] + e[4]);
         
         // If its the best MAD thus far record it...
          if (MAD<bestMAD)
          {
           bestMedian = median;
           bestMAD = MAD;
          }
        }

       // Calculate the difference between the robust estimate of the correct distance and the current distance, and use that in a psuedo Huber to update the cost and gradiant...
        float err = SDF2(y, x) - bestMedian;
        double inner = sqrt(1.0 + err*err/tgs);
        ret += tgs * (inner - 1.0);
        GRAD2(y, x) += err / inner;
      }
     }
    
    // Handle the return...
     return_val = ret;
    """
    
    unary_sharpness = self.unary_sharpness
    tertiary_grad = self.tertiary_grad
    
    return weave.inline(code, ['cost', 'sdf', 'analyse', 'grad', 'unary_sharpness', 'tertiary_grad'], support_code=support)


  def __call__(self, prob):
    """Given as input a probability map (Or something that behaves like one! Will be clamped) of the probability of being part of the line this returns a mask indicating line/not line."""
    # Convert the probability map into a cost of being a line, with 0 the cost of being background...
    clip_prob = numpy.clip(prob, self.limit, 1.0-self.limit)
    cost = numpy.log(1.0-clip_prob) - numpy.log(clip_prob)
    
    # Threshold the given probabilities to create an initial mask...
    mask = numpy.empty(prob.shape, dtype=numpy.bool)
    mask[:,:]  = prob>self.threshold
    
    # Create a signed distance function from the mask...
    sdf = self.mask_to_sdf(mask)
    
    # We only analyse pixels within a certain radius of the line, to speed stuff up...
    analyse = sdf < self.radius
    
    # Optimise the sdf using the cost function to find the lowest cost assignment...
    # (Using Nesterov's method - Gradient decent with fast convergance)
    lamb = 1.0
    grad = sdf.copy()
    sdf_gd_old = sdf.copy()
    sdf_gd = sdf.copy()
    
    for _ in xrange(self.iters):
      # Calculate the gradient - break if its close enough to zero for all values...
      c = self.cost(cost, sdf, analyse, grad)
      if numpy.all(numpy.fabs(grad)<1e-5):
        break
      
      # Calculate the normal gradient descent update...
      sdf_gd[:,:] = sdf - self.step_size * grad
      
      # Calculate the gradient update with weird momentum term...
      new_lamb = 0.5 * (1.0 + numpy.sqrt(1.0 + 4.0 * lamb * lamb))
      gamma = (1.0 - lamb) / new_lamb
      
      sdf[:,:] = (1.0 - gamma) * sdf_gd + gamma * sdf_gd_old
      
      # Prepare for next iteration...
      sdf_gd_old[:,:] = sdf_gd
      lamb = new_lamb
    
    # Convert the signed distance function back to a mask and return it...
    mask[:,:] = sdf<0.0
    return mask
