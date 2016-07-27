# Copyright 2016 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import numpy

from graph_cuts.binary_label import *
from ms.ms import MeanShift
from misc.tps import TPS

from scipy import weave
from utils.start_cpp import start_cpp



def tight_total(data, total):
  """Given a 1D array and a total this returns the smallest interval (as (start (inclusive), end (exclusive))) that contains the given amount of the total."""
  
  # Create matrix containing the total for every range...
  cs = numpy.append(numpy.array([0], dtype=data.dtype), numpy.cumsum(data))
  assert(total<=cs[-1])
  scale = cs.reshape((1,-1)) - cs.reshape((-1,1))
  
  # Fetch those that are over the total, and find the one with the shortest range...
  valid = numpy.argwhere(scale>=total)
  index = numpy.argsort(valid[:,1]-valid[:,0])[0]
  
  # Return the range tuple...
  return (valid[index,0], valid[index,1])



def cuboid_bg_model(image, percentage = 50.0):
  """Given an image, as a numpy array [y,x,c] where c is 0=r, 1=g, 2=b, of type numpy.uint8, and a total percentage, this selects a cuboid of colour space to be the background, everything else foreground. The cuboid is selected by, for each colour channel, calculating the smallest range that includes the given percentage of the pixels. Returns a boolean array, indexed [r,g,b] - this is obviously quite large, as its a 256x256x256 array (16 meg), but it makes applying the threshold easy and supports more complex models for if needed."""
  
  # Convert to a total and flatten the image...
  total = numpy.ceil((percentage/100.0)*image.shape[0]*image.shape[1])
  data = image.reshape((-1,3))
  
  # For each channel in turn calculate a range...
  col_ran = []
  for c in xrange(3):
    histo = numpy.bincount(data[:,c], minlength=256)
    cr = tight_total(histo, total)
    col_ran.append(cr)
  
  # Create the region and mark the ranges selected as being background...
  ret = numpy.zeros((256,256,256), dtype=numpy.bool)
  
  ret[:col_ran[0][0],:,:] = True
  ret[col_ran[0][1]:,:,:] = True
  
  ret[:,:col_ran[1][0],:] = True
  ret[:,col_ran[1][1]:,:] = True
  
  ret[:,:,:col_ran[2][0]] = True
  ret[:,:,col_ran[2][1]:] = True
  
  return ret



def threshold(image, model):
  """Given an image (numpy array, [y,x,c] where c is the channel) and a model (boolean array[256,256,256] - True for foreground, False for background.)"""
  flat = image.reshape((-1,3))
  return model[flat[:,0], flat[:,1], flat[:,2]].reshape((image.shape[0],image.shape[1]))

  
  
def threshold_reg(image, model, data_diff = 1.0, smooth_max = 4.0, half_life = 32.0):
  """Same as threshold, except it uses graph cuts to regularise the output. You provide the data difference, which is the bias towards the original thresholding label of each pixel, and a smooth max, which is the maximum cost of two labels being different. half_life then modulates the smoothing, by defining the colourmetric distance at which the cost of difference becomes 0."""
  
  # Create a binary labelling object...
  bl = BinaryLabel((image.shape[0], image.shape[1]))
  
  # Add in the basic thresholding result, with the given difference...
  initial = threshold(image, model)
  bl.addCostFalse(initial * data_diff)
  bl.addCostTrue((1.0-initial) * data_diff)
  
  # Add in the smoothing terms...
  ## Dimension 0...
  dist = numpy.square(image[1:,:].astype(numpy.float32) - image[:-1,:].astype(numpy.float32)).sum(axis=2)
  bl.addCostDifferent(0, (smooth_max * half_life) / (half_life + dist))
  
  ## Dimension 1...
  dist = numpy.square(image[:,1:].astype(numpy.float32) - image[:,:-1].astype(numpy.float32)).sum(axis=2)
  bl.addCostDifferent(1, (smooth_max * half_life) / (half_life + dist))
  
  # Solve the binary labelling to get the final output, return it...
  return bl.solve()[0]



def density_median(density, radius = 2, exp_weight = 0.0):
  """Performs a weighted median for each pixel in a density map - pixels are weighted by 1 minus their distance to the input values density. Uses the given radius to define the window for each pixel. Returns a new modified density map. exp_weight is a weight given to a location based on its value - used to bias towards larger values if set positive for instance."""
  ret = density.copy()
  
  support = start_cpp() + """
  struct Sam
  {
   float value;
   float weight;
  };
  
  int comp_sam(const void * a, const void * b)
  {
   Sam & fa = *(Sam*)a;
   Sam & fb = *(Sam*)b;
     
   if (fa.value<fb.value) return -1;
   if (fb.value<fa.value) return 1;
   return 0;
  }
  """
  
  code = start_cpp() + """
  int rad = radius;
  float ew = exp_weight;
  
  Sam * sam = new Sam[(radius * 2 + 1) * (radius * 2 + 1)];
  
  for (int y=radius; y<Ndensity[0]-radius; y++)
  {
   for (int x=radius; x<Ndensity[1]-radius; x++)
   {
    // Collect the samples required...
     float centre_value = DENSITY2(y, x);
    
     int total = 0;
     float total_weight = 0.0;
     for (int dy=-radius; dy<=radius; dy++)
     {
      const int range = radius - abs(dy);
      for (int dx=-range; dx<=range; dx++)
      {
       if ((dx!=0)||(dy!=0)) // Middle pixel doesn't get a vote!
       {
        float cv = sam[total].value;
        if (cv<1e-3) cv = 1e-3;
        if (cv>1.0) cv = 1.0;
        
        sam[total].value = DENSITY2(y+dy, x+dx);
        sam[total].weight = (1.0 - fabs(centre_value - cv)) * pow(cv, ew);
        
        total_weight += sam[total].weight;
        total += 1;
       }
      }
     }
     
    // Sort them...
     qsort(sam, total, sizeof(Sam), comp_sam);
    
    // Find the median, assign the pixel...
     float remain = 0.5 * total_weight;
     
     for (int i=0; i<total; i++)
     {
      if (remain>sam[i].weight)
      {
       remain -= sam[i].weight;
      }
      else
      {
       if (i==0) RET2(y, x) = sam[0].value;
       else
       {
        float t = remain / sam[i].weight;
        RET2(y, x) = (1-t) * sam[i-1].value + t * sam[i].value;
       }
       
       break;
      }
     }
     
    // Only use it if its larger than - anotehr bias term...
     if (DENSITY2(y, x)>RET2(y, x)) RET2(y, x) = DENSITY2(y, x);
   }
  }
  
  delete[] sam;
  """
  
  weave.inline(code, ['density', 'ret', 'radius', 'exp_weight'], support_code=support)
  
  return ret



def threshold_density(image, density, bg_cost = 0.5, data_mult = 32.0, smooth_max = 16.0, lonely = 0.75, half_life = 32.0, force = None):
  """Same as threshold, except it uses graph cuts to regularise the output. You provide the data difference, which is the bias towards the original thresholding label of each pixel, and a smooth max, which is the maximum cost of two labels being different. half_life then modulates the smoothing, by defining the colourmetric distance at which the cost of difference becomes 0. You can optionally provide a force array, same shape and size as the image, where 0 means to process as normal, 1 means fix to background, 2 means fix to foreground."""
  
  # Create a binary labelling object...
  bl = BinaryLabel((image.shape[0], image.shape[1]))
  
  # Use the density estimate to set the costs......
  bl.addCostTrue((density<1e-3).astype(numpy.float32) * bg_cost)
  bl.addCostFalse(numpy.clip(density, 0.0, 1.0) * data_mult)
  
  # Add in the smoothing terms...
  ## Dimension 0...
  dist = numpy.square(image[1:,:].astype(numpy.float32) - image[:-1,:].astype(numpy.float32)).sum(axis=2)
  bl.addCostDifferent(0, (smooth_max * half_life) / (half_life + dist))
  
  ## Dimension 1...
  dist = numpy.square(image[:,1:].astype(numpy.float32) - image[:,:-1].astype(numpy.float32)).sum(axis=2)
  bl.addCostDifferent(1, (smooth_max * half_life) / (half_life + dist))
  
  # Avoid noise!..
  bl.setLonelyCost(lonely)
  
  # Lock some pixels, at the users request...
  if force!=None:
    fix = numpy.zeros(force.shape, dtype=numpy.int8)
    fix[force==1] = -1
    fix[force==2] = 1
    
    bl.fix(fix)
  
  # Solve the binary labelling to get the final output, return it...
  return bl.solve()[0]



def cluster_colour(image, size = 16.0, kernel = 'epanechnikov', halves = 3):
  """Builds a colour cube, and applies mean shift to it (Each location is weighted by the number of pixels that have that colour), to cluster the colours. Each cluster is then assigned to be foreground or background. The return value is a tuple: (boolean array, indexed by [r,g,b] of False for background, True for foreground, a TPS thin plate spline object, that assigns a density value to every location in the colour cube.)."""
  
  # Prepare the data, applying the halving...
  data = image.reshape((-1,3)).copy()
  dim = 256
  
  scale = 1
  for _ in xrange(halves):
    data /= 2
    dim /= 2
    size *= 0.5
    scale *= 2
  
  # Count how many instances of each colour exist, building a 3D colour cube...
  exploded = data[:,0].astype(numpy.uint32) * dim * dim + data[:,1].astype(numpy.uint32) * dim + data[:,2].astype(numpy.uint32)
  
  c_cube = numpy.bincount(exploded, minlength=dim*dim*dim)
  c_cube = c_cube.reshape((dim,dim,dim))
  
  del exploded
  del data
  
  # Setup mean shift...
  ms = MeanShift()
  ms.set_data(c_cube, 'bbb', 3)

  ms.set_kernel(kernel)
  ms.set_spatial('iter_dual')
  ms.set_balls('hash')
  
  ms.set_scale(numpy.array([1.0/size, 1.0/size, 1.0/size]))
  
  # Run it, to get cluster information for every pixel...
  modes, indices = ms.cluster()
  
  # Calculate the size of each cluster...
  index = indices.flatten() >= 0
  sizes = numpy.bincount(indices.flatten()[index], c_cube.flatten()[index])
  
  # Select the largest cluster as the background - everything else is foreground...
  bg = numpy.argmax(sizes)
  
  # Create the bg/fg colour cube boolean map...
  bg_cube = indices!=bg
  
  # Create a data matrix to train a thin plate spline from - iterate the foreground clusters, and for each take a slice and linearise the cumulative probability over this range...
  points = []
  
  samples = 1024
  cum_samples = 64
  
  main_fg = numpy.concatenate((sizes[:bg], numpy.array([-1.0]), sizes[bg+1:])).argmax()
  threshold = 0.5 * sizes[main_fg]
  
  for i in [main_fg]: #xrange(modes.shape[0]):
    if i==bg: continue
    if sizes[i]<threshold: continue
    
    # We have the index of the centre of a valid cluster - calculate evenly spaced coordinates along the line connecting its centre to the background centre...
    delta = modes[i,:] - modes[bg,:]
    
    best_dot = 1e64
    best_vec = None
    for i in xrange(3):
      vec = numpy.zeros(3)
      vec[i] = 1.0
      dot = delta.dot(vec)
      if dot<best_dot:
        best_dot = dot
        best_vec = vec
      
    perpA = numpy.cross(delta, best_vec)
    perpB = numpy.cross(delta, perpA)
    perpA *= 0.5 / numpy.sqrt(perpA.dot(perpA))
    perpB *= 0.5 / numpy.sqrt(perpB.dot(perpB))
    
    r = numpy.linspace(modes[bg,0], modes[bg,0] + 2.0*delta[0], samples)
    g = numpy.linspace(modes[bg,1], modes[bg,1] + 2.0*delta[1], samples)
    b = numpy.linspace(modes[bg,2], modes[bg,2] + 2.0*delta[2], samples)
    sams = numpy.concatenate((r.reshape((-1,1)), g.reshape((-1,1)), b.reshape((-1,1))), axis=1)
    
    # Calculate a probability for each entry in sams...
    probs = ms.probs(sams)
    
    # Calculate a cluster assignment for each entry in sams...
    clusters = ms.assign_clusters(sams)
    
    # Zero all entries that belong to the background...
    probs[clusters==bg] = 0.0
    
    # Make the probabilities cumulative, smooth them and normalise...
    probs = numpy.cumsum(probs)
    probs /= probs[probs.shape[0]//2]
    
    # Find the requested sample points and add them to the points array...
    cs = numpy.ceil(cum_samples * float(sizes[i]) / float(sizes[main_fg]))
    
    for val in numpy.linspace(1e-6, 2.0, cum_samples):
      high = numpy.searchsorted(probs, val)
      if high==probs.shape[0]: break
      
      low = high - 1
      if low<0:
        low = 0
        high = 1
        
      t = (val - probs[low]) / (probs[high] - probs[low])
      
      loc = (1.0-t) * sams[low,:] + t * sams[high,:]
      
      points.append((loc+perpA+perpB, val))
      points.append((loc+perpA-perpB, val))
      points.append((loc-perpA+perpB, val))
      points.append((loc-perpA-perpB, val))
  
  dm_x = numpy.concatenate(map(lambda s: s[0].reshape((1,-1)), points), axis=0)
  dm_y = numpy.array(map(lambda s: s[1], points))
  
  # Fit the thin plate spline to the points...
  tps = TPS(3)
  tps.learn(dm_x * scale, dm_y)
  
  # Scale up the bg cube, to undo the halving...
  bg_cube = bg_cube.repeat(scale, axis=0)
  bg_cube = bg_cube.repeat(scale, axis=1)
  bg_cube = bg_cube.repeat(scale, axis=2)
  
  # Do the return...
  return (bg_cube, tps)



def dilate(mask, repeat = 1):
  """Given a mask this dilates it repeat times, and returns the new mask. Uses a simple diamond mask, which is the 4 neighbours of each pixel."""
  ret = mask.copy()
  
  for _ in xrange(repeat):
    prev = ret.copy()
    numpy.logical_or(ret[1:,:], prev[:-1,:], ret[1:,:])
    numpy.logical_or(ret[:-1,:], prev[1:,:], ret[:-1,:])
    numpy.logical_or(ret[:,1:], prev[:,:-1], ret[:,1:])
    numpy.logical_or(ret[:,:-1], prev[:,1:], ret[:,:-1])
  
  return ret



def erode(mask, repeat = 1):
  """Given a mask this erodes it repeat times, and returns the new mask. Uses a simple diamond mask, which is the 4 neighbours of each pixel."""
  ret = mask.copy()
  
  for _ in xrange(repeat):
    prev = ret.copy()
    numpy.logical_and(ret[1:,:], prev[:-1,:], ret[1:,:])
    numpy.logical_and(ret[:-1,:], prev[1:,:], ret[:-1,:])
    numpy.logical_and(ret[:,1:], prev[:,:-1], ret[:,1:])
    numpy.logical_and(ret[:,:-1], prev[:,1:], ret[:,:-1])
  
  return ret



def smooth(mask, repeat = 1):
  """Smooths a mask by repeatedly dilating and then eroding it, the given number of times."""
  for _ in xrange(repeat): mask = dilate(mask)
  for _ in xrange(repeat): mask = erode(mask)
  
  return mask



def smooth_signed_distance(mask, iters = 1):
  """Given a mask this smooths it using a vaugly-not-stupid techneque based on signed distance to the edge of the line - converts the mask, then smoothes it using an oriented filter, that strongly enforces smooth contours and applies sub-pixel estimation. It then converts it back to a mask using the sign of the resulting distance. Should smoooth out bumps and sharpen small angle intersections."""

  # Convert to signed distance, using the 8 way neighbourhood (With sqrt(2) for the diagonals)...
  ## Initalise with effective infinities...
  sigdist = numpy.empty(mask.shape, dtype=numpy.float32)
  sigdist[:,:] = 1e64
  
  ## Mark all pixels that are at a transition boundary with the relevant cost - first the diagonals, then the halfs, as half is less than sqrt(2)...
  tran_sqrt2 = numpy.zeros(sigdist.shape, dtype=numpy.bool)
  numpy.logical_or(mask[1:,1:]!=mask[:-1,:-1], tran_sqrt2[:-1,:-1], tran_sqrt2[:-1,:-1])
  numpy.logical_or(mask[1:,:-1]!=mask[:-1,1:], tran_sqrt2[:-1,1:], tran_sqrt2[:-1,1:])
  numpy.logical_or(mask[:-1,1:]!=mask[1:,:-1], tran_sqrt2[1:,:-1], tran_sqrt2[1:,:-1])
  numpy.logical_or(mask[:-1,:-1]!=mask[1:,1:], tran_sqrt2[1:,1:], tran_sqrt2[1:,1:])
  sigdist[tran_sqrt2] = numpy.sqrt(2.0)
  
  tran_half = numpy.zeros(sigdist.shape, dtype=numpy.bool)
  numpy.logical_or(mask[1:,:]!=mask[:-1,:], tran_half[:-1,:], tran_half[:-1,:])
  numpy.logical_or(mask[:-1,:]!=mask[1:,:], tran_half[1:,:], tran_half[1:,:])
  numpy.logical_or(mask[:,1:]!=mask[:,:-1], tran_half[:,:-1], tran_half[:,:-1])
  numpy.logical_or(mask[:,:-1]!=mask[:,1:], tran_half[:,1:], tran_half[:,1:])
  sigdist[tran_half] = 0.5
  
  ## Do all 8 directions of sweep iterativly until distances stop getting smaller...
  stop = False
  while not stop:
    stop = True
    
    code = start_cpp() + """
    float sqrt2 = sqrt(2.0);
    
    // Forwards pass...
     for (int y=0; y<Nsigdist[0]; y++)
     {
      for (int x=0; x<Nsigdist[1]; x++)
      {
       bool negx = x!=0;
       bool negy = y!=0;
       
       if ((negx)&&((SIGDIST2(y, x-1)+1.0)<SIGDIST2(y, x)))
       {
        SIGDIST2(y, x) = SIGDIST2(y, x-1) + 1.0;
        stop = false;
       }
       
       if ((negy)&&((SIGDIST2(y-1, x)+1.0)<SIGDIST2(y, x)))
       {
        SIGDIST2(y, x) = SIGDIST2(y-1, x) + 1.0;
        stop = false;
       }
       
       if ((negx)&&(negy)&&((SIGDIST2(y-1, x-1)+sqrt2)<SIGDIST2(y, x)))
       {
        SIGDIST2(y, x) = SIGDIST2(y-1, x-1) + sqrt2;
        stop = false;
       }
      }
     }
    
    // Backwards pass...
     for (int y=Nsigdist[0]-1; y>=0; y--)
     {
      for (int x=Nsigdist[1]-1; x>=0; x--)
      {
       bool posx = (x+1)!=Nsigdist[1];
       bool posy = (y+1)!=Nsigdist[0];
       
       if ((posx)&&((SIGDIST2(y, x+1)+1.0)<SIGDIST2(y, x)))
       {
        SIGDIST2(y, x) = SIGDIST2(y, x+1) + 1.0;
        stop = false;
       }
       
       if ((posy)&&((SIGDIST2(y+1, x)+1.0)<SIGDIST2(y, x)))
       {
        SIGDIST2(y, x) = SIGDIST2(y+1, x) + 1.0;
        stop = false;
       }
       
       if ((posx)&&(posy)&&((SIGDIST2(y+1, x+1)+sqrt2)<SIGDIST2(y, x)))
       {
        SIGDIST2(y, x) = SIGDIST2(y+1, x+1) + sqrt2;
        stop = false;
       }
      }
     }
    """
    
    weave.inline(code, ['sigdist', 'stop'])
  
  ## Add in the sign - negate all pixels that are within the mask...
  sigdist[mask] *= -1.0
  
  # Apply a funky smoothing function...
  temp = sigdist.copy()
  use = sigdist<16.0 # Don't bother with pixels that are too far from the text.
  
  for _ in xrange(iters):
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
    
    code = start_cpp(support) + """
    // Calculate and store the smoothed version into temp...
     for (int y=1; y<Nsigdist[0]-1; y++)
     {
      for (int x=1; x<Nsigdist[1]-1; x++)
      {
       if (USE2(y, x)==0) continue; // Skip pixels that are too far away to care about.
       
       static const char dx[8] = {-1,  0,  1, 1, 1, 0, -1, -1};
       static const char dy[8] = {-1, -1, -1, 0, 1, 1,  1,  0};
       static const float div[8] = {sqrt(2), 1, sqrt(2), 1, sqrt(2), 1, sqrt(2), 1};
       
       // Loop through using a line direction estimated from each set of 3 adjacent neigbours in the 8-way neighbourhood - select the line direction that results in the lowest MAD, using the median of the estimates. The estimates are based on the signed distance of the neighbour offset by the projection distance to the line...
       
        float bestMedian = SIGDIST2(y, x);
        float bestMAD = 1e64;
       
        for (int ni=0; ni<8; ni++)
        {
         // Estimate the perpendicular to the line direction from the 3 neighbours under consideration - a maximum liklihood mean direction of a Fisher distribution...
          float nx = 0.0;
          float ny = 0.0;
          
          bool skip = false;
          for (int oi=0; oi<3; oi++)
          {
           int i = (ni+oi) % 8;
           
           if (USE2(y + dy[i], x + dx[i])==0)
           {
            skip = true;
            break;
           }
          
           float l = SIGDIST2(y + dy[i], x + dx[i]) - SIGDIST2(y, x);
           l /= div[i];
          
           nx += l * dx[i];
           ny += l * dy[i];
          }
          if (skip) continue;
         
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
           e[i] = SIGDIST2(y + dy[i], x + dx[i]) + dot;
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
        
        TEMP2(y, x) = bestMedian;
      }
     }
    
    // Copy from temp to the actual signed distance field (Yeah, pointer flipping would make more sense, but no idea how to make a numpy object dance that)...
     for (int y=1; y<Nsigdist[0]-1; y++)
     {
      for (int x=1; x<Nsigdist[1]-1; x++)
      {
       SIGDIST2(y, x) = TEMP2(y, x);
      }
     }
    """
    
    weave.inline(code, ['sigdist', 'temp', 'use'], support_code=support)
  
  # Convert back to a mask and return...
  return sigdist<=0.0



def nuke_islands(mask, size = 1):
  """Removes all islands in the mask that are less than or equal to the given size by flipping their state. Good for toasting salt and pepper noise."""
  support = start_cpp() + """
  struct Tree
  {
   Tree * parent;
   int size;
  };
  
  Tree * Parent(Tree * tree)
  {
   if (tree->parent==NULL) return tree;
   Tree * ret = Parent(tree->parent);
   tree->parent = ret;
   return ret;
  }
  
  void Merge(Tree * a, Tree * b)
  {
   a = Parent(a);
   b = Parent(b);
   
   if (a!=b)
   {
    b->parent = a;
    a->size += b->size;
   }
  }

  """
  
  code = start_cpp(support) + """
  // Malloc and plant a forest...
   int trees = Nmask[0] * Nmask[1];
   Tree * forest = (Tree*)malloc(trees * sizeof(Tree));
   
   for (int i=0; i<trees; i++)
   {
    forest[i].parent = NULL;
    forest[i].size = 1;
   }
  
  // Merge it to create the islands...
   for (int y=0; y<Nmask[0]; y++)
   {
    for (int x=0; x<Nmask[1]; x++)
    {
     int fi = y*Nmask[1] + x;
     
     if ((x!=0)&&(MASK2(y, x)==MASK2(y, x-1)))
     {
      Merge(forest + fi, forest + fi - 1);
     }
     
     if ((y!=0)&&(MASK2(y, x)==MASK2(y-1, x)))
     {
      Merge(forest + fi, forest + fi - Nmask[1]);
     }
    }
   }
   
  // Flip the state of all pixels that are in too small an island...
   for (int y=0; y<Nmask[0]; y++)
   {
    for (int x=0; x<Nmask[1]; x++)
    {
     int fi = y*Nmask[1] + x;
     Tree * parent = Parent(forest + fi);
     
     if (parent->size <= size)
     {
      MASK2(y, x) = (MASK2(y, x) + 1) % 2;
     }
    }
   } 
  
  // Clean up...
   free(forest);
  """
  
  mask = mask.copy()
  weave.inline(code, ['mask', 'size'], support_code=support)
  
  return mask
