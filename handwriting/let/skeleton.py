# Copyright 2016 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import numpy
import scipy.weave as weave

from ms.ms import MeanShift

from utils.start_cpp import start_cpp



def zhang_suen(mask):
  """Given a mask this thins it to get a skeleton of the object, returning a replacement mask that is only one pixel wide."""
  assert(len(mask.shape)==2)
  assert(mask.dtype==numpy.bool)

  # Variables that are needed...
  mask = mask.copy()
  count = numpy.empty(mask.shape, dtype=numpy.int32)
  to_die = numpy.empty(mask.shape, dtype=numpy.bool)

  # A version performed in c, to make it fast...
  try:
    code = start_cpp() + """
    // Setup...
     int iteration = -1;
     const int size = Nmask[0] * Nmask[1];
     for (int y=0; y<Ncount[0]; y++)
     {
      for (int x=0; x<Ncount[1]; x++) COUNT2(y,x) = 1;
     }

    // Loop until done...
     while (true)
     {
      iteration += 1;

      // Reset the to die array...
       for (int y=0; y<Nto_die[0]; y++)
       {
        for (int x=0; x<Nto_die[1]; x++) TO_DIE2(y,x) = 0;
       }

      // Iterate and process every pixel in turn, recording if it is to die or not - we use count as a skip array to skip over pixels that are already marked as false...
       int prev = -1;
       for (int i=0; i<size;)
       {
        // Get coordinates...
         int x = i % Nmask[1];
         int y = i / Nmask[1];

        // Only process if its curently true...
         if (MASK2(y,x)!=0)
         {
          // Shorten the skip array...
           if ((i!=0)&&(prev==-1)) prev = 0;
           if (prev!=-1) COUNT2(prev/Nmask[1],prev%Nmask[1]) = i - prev;
           prev = i;

          // Extract the neighbours - 1 for true, 0 for false...
          // (Means we only handle boundary conditions once)
           char p[8];
           p[0] = (y!=0) ? (MASK2(y-1,x)) : (0);
           p[1] = (y!=0 && x+1<Nmask[1]) ? (MASK2(y-1,x+1)) : (0);
           p[2] = (x+1<Nmask[1]) ? (MASK2(y,x+1)) : (0);
           p[3] = (x+1<Nmask[1] && y+1<Nmask[0]) ? (MASK2(y+1,x+1)) : (0);
           p[4] = (y+1<Nmask[0]) ? (MASK2(y+1,x)) : (0);
           p[5] = (y+1<Nmask[0] && x!=0) ? (MASK2(y+1,x-1)) : (0);
           p[6] = (x!=0) ? (MASK2(y,x-1)) : (0);
           p[7] = (x!=0 && y!=0) ? (MASK2(y-1,x-1)) : (0);

          // Do the neighbour count test...
           int c = p[0] + p[1] + p[2] + p[3] + p[4] + p[5] + p[6] + p[7];
           if (c>=2 && c<=6)
           {
            // Do the change count test...
             c = (p[0] ^ p[1]) + (p[1] ^ p[2]) + (p[2] ^ p[3]) + (p[3] ^ p[4]) + (p[4] ^ p[5]) + (p[5] ^ p[6]) + (p[6] ^ p[7]) + (p[7] ^ p[0]);
             if (c==2)
             {
              // Do the two corner tests...
               bool passed;
               if ((iteration%2)==0)
               {
                passed = ((p[0]&p[2]&p[4])==0) && ((p[2]&p[4]&p[6])==0);
               }
               else
               {
                passed = ((p[0]&p[2]&p[6])==0) && ((p[0]&p[4]&p[6])==0);
               }

               if (passed)
               {
                // Its passed all tests - off with its head!..
                 TO_DIE2(y,x) = 1;
               }
             }
           }
         }

        // Move to next...
         i += COUNT2(y, x);
       }

      // Apply the to_die array...
       int changes = 0;
       for (int i=0; i<size;)
       {
        int x = i % Nmask[1];
        int y = i / Nmask[1];

        if (TO_DIE2(y,x)!=0)
        {
         MASK2(y,x) = 0;
         changes += 1;
        }
        i += COUNT2(y, x);
       }

      // Finish if we are done...
       if (changes==0) break;
     }
    """

    weave.inline(code, ['mask', 'count', 'to_die'])

    return mask
  except:
    print 'Inline code failed - slow python version being used instead'


  # Create assorted intermediates...
  changes = numpy.empty(mask.shape, dtype=numpy.int32)
  corner1 = numpy.empty(mask.shape, dtype=numpy.bool)
  corner2 = numpy.empty(mask.shape, dtype=numpy.bool)

  # Iterate until convergance...
  iteration = -1
  while True:
    iteration += 1

    # Count how many neighbours each pixel has...
    count[:,:] = 0

    count[1:,:] += mask[:-1,:]
    count[:-1,:] += mask[1:,:]
    count[:,1:] += mask[:,:-1]
    count[:,:-1] += mask[:,1:]

    count[1:,1:] += mask[:-1,:-1]
    count[1:,:-1] += mask[:-1,1:]
    count[:-1,1:] += mask[1:,:-1]
    count[:-1,:-1] += mask[1:,1:]

    # Count how many zero-one changes occur in the ring around each pixel...
    # (We do it both ways, unlike the paper - easier, and it just means the condition is ==2.)
    changes[:,:] = 0

    changes[1:,1:] += numpy.logical_xor(mask[:-1,:-1], mask[:-1,1:])
    changes[1:,1:] += numpy.logical_xor(mask[:-1,:-1], mask[1:,:-1])

    changes[:-1,1:] += numpy.logical_xor(mask[1:,:-1], mask[1:,1:])
    changes[:-1,1:] += numpy.logical_xor(mask[1:,:-1], mask[:-1,:-1])

    changes[1:,:-1] += numpy.logical_xor(mask[:-1,1:], mask[:-1,:-1])
    changes[1:,:-1] += numpy.logical_xor(mask[:-1,1:], mask[1:,1:])

    changes[:-1,:-1] += numpy.logical_xor(mask[1:,1:], mask[1:,:-1])
    changes[:-1,:-1] += numpy.logical_xor(mask[1:,1:], mask[:-1,1:])

    # Do the last two conditions, depending on which subiteration we are on...
    corner1[:,:] = True
    corner2[:,:] = True

    if (iteration%2)==0:
      corner1[0,:] = False
      corner1[1:,:] = numpy.logical_and(corner1[1:,:], mask[:-1,:]) # P2

      corner1[:,-1] = False
      corner1[:,:-1] = numpy.logical_and(corner1[:,:-1], mask[:,1:]) # P4

      corner1[-1,:] = False
      corner1[:-1,:] = numpy.logical_and(corner1[:-1,:], mask[1:,:]) # P6

      corner2[:,-1] = False
      corner2[:,:-1] = numpy.logical_and(corner2[:,:-1], mask[:,1:]) # P4

      corner2[-1,:] = False
      corner2[:-1,:] = numpy.logical_and(corner2[:-1,:], mask[1:,:]) # P6

      corner2[:,0] = False
      corner2[:,1:] = numpy.logical_and(corner2[:,1:], mask[:,:-1]) # P8
    else:
      corner1[:,0] = False
      corner1[:,1:] = numpy.logical_and(corner1[:,1:], mask[:,:-1]) # P8

      corner1[0,:] = False
      corner1[1:,:] = numpy.logical_and(corner1[1:,:], mask[:-1,:]) # P2

      corner1[:,-1] = False
      corner1[:,:-1] = numpy.logical_and(corner1[:,:-1], mask[:,1:]) # P4

      corner2[-1,:] = False
      corner2[:-1,:] = numpy.logical_and(corner2[:-1,:], mask[1:,:]) # P6

      corner2[:,0] = False
      corner2[:,1:] = numpy.logical_and(corner2[:,1:], mask[:,:-1]) # P8

      corner2[0,:] = False
      corner2[1:,:] = numpy.logical_and(corner2[1:,:], mask[:-1,:]) # P2

    # Merge the conditions...
    to_die[:,:] = mask

    to_die[:,:] = numpy.logical_and(to_die, count >= 2)
    to_die[:,:] = numpy.logical_and(to_die, count <= 6)

    to_die[:,:] = numpy.logical_and(to_die, changes == 2)

    to_die[:,:] = numpy.logical_and(to_die, corner1 == False)
    to_die[:,:] = numpy.logical_and(to_die, corner2 == False)

    # Apply the update...
    mask[numpy.where(to_die)] = False

    # Exit if nothing happened...
    if to_die.sum()==0: break

  # Return...
  return mask



def refine_mask(density, line_mask, radius, threshold = 1e-3):
  """Given a mask of lines extracted from an image this refines it, using subspace constrained mean shift. For every pixel it runs it to refine the location, noting that it might end up at another pixel. It also iterativly converges from adjacent pixels in the 8 neighbourhood, to see if it can grow the line any further. When multiple pixels conmverge to the same pixel it takes the subpixel position closest to the centre, to maintain an even sampling of the position. Returns the tuple (new mask, subpixel coordinates, float32 array of offsets from pixel centre, as [y,x,0] for y offset and [y,x,1] for x offset. The density estimate is created using a dense assignment of density to every pixel, with a radius parameter to control the size of the Gaussian kernel."""
  
  # First setup the mean shift object...
  ms = MeanShift()
  ms.set_data(density, 'bb', 2)
  ms.set_spatial('iter_dual')
  ms.set_scale(numpy.array([1.0/radius, 1.0/radius]))
  ms.epsilon = 1e-2
  
  # Setup the three buffers - the mask of pixels to checkout, the mask of confirmed pixels, and the subpixel refinement array...
  check_mask = line_mask.copy()
  done_mask = numpy.zeros(line_mask.shape, dtype=numpy.bool)
  keep_mask = numpy.zeros(line_mask.shape, dtype=numpy.bool)
  subpixel = numpy.ones((line_mask.shape[0], line_mask.shape[1], 2), dtype=numpy.float32)
  
  # Loop until the check mask is empty...
  while True:
    # Apply and update the done mask, check for convergance...
    numpy.logical_or(keep_mask, done_mask, done_mask)
    check_mask[done_mask] = False
    numpy.logical_or(check_mask, done_mask, done_mask)
    check_mask[density <= threshold] = False
    if not numpy.any(check_mask): break
    
    # Convert the check mask into a list of coordinates; zero it out ready for the next iteration...
    source = numpy.transpose(numpy.nonzero(check_mask!=0))
    check_mask[:,:] = False

    # Run subspace constrained mean shift on each coordinate...
    if source.shape[0]>1000: print 'scms %i points...'%source.shape[0]
    dest = ms.manifolds(source, 1)
    if source.shape[0]>1000: print '...done'
    
    # Update the keep_mask with the converged pixels, checking that the locations are valid and recording subpixel information...
    for i in xrange(dest.shape[0]):
      source_i = numpy.round(source[i,:]).astype(numpy.int32)
      dest_i = numpy.round(dest[i,:]).astype(numpy.int32)
      offset = dest[i,:] - dest_i.astype(numpy.float32)
            
      if keep_mask[dest_i[0], dest_i[1]]:
        cd = subpixel[dest_i[0], dest_i[1]]
        cd = cd.dot(cd)
        nd = offset.dot(offset)
        
        if nd<cd:
          subpixel[dest_i[0], dest_i[1]] = offset
          
      else:
        keep_mask[dest_i[0], dest_i[1]] = True
        subpixel[dest_i[0], dest_i[1]] = offset
        
        lowY  = dest_i[0]-1 if dest_i[0]>0 else 0
        highY = dest_i[0]+2 if dest_i[0]+2<=check_mask.shape[0] else check_mask.shape[0]
        lowX  = dest_i[1]-1 if dest_i[1]>0 else 0
        highX = dest_i[1]+2 if dest_i[1]+2<=check_mask.shape[1] else check_mask.shape[1]
        check_mask[lowY:highY, lowX:highX] = True
      
      # Check if we need to add a line in to handle the gaps between dominant and minor lines that can occur - first test is if we have actually moved anywhere...
      if abs(source_i[0]-dest_i[0])>1 or abs(source_i[1]-dest_i[1])>1:
        # Only fill in the line if the start point was next to an already established point...
        established = False
        
        lowY  = source_i[0]>0
        highY = source_i[0]+1 < check_mask.shape[0]
        lowX  = source_i[1]>0
        highX = source_i[1]+1 < check_mask.shape[1]
        
        if lowY: established = established or keep_mask[source_i[0]-1, source_i[1]]
        if highY: established = established or keep_mask[source_i[0]+1, source_i[1]]
        if lowX: established = established or keep_mask[source_i[0], source_i[1]-1]
        if highX: established = established or keep_mask[source_i[0], source_i[1]+1]
        
        if lowY and lowX: established = established or keep_mask[source_i[0]-1, source_i[1]-1]
        if lowY and highX: established = established or keep_mask[source_i[0]-1, source_i[1]+1]
        if highY and lowX: established = established or keep_mask[source_i[0]+1, source_i[1]-1]
        if highY and highX: established = established or keep_mask[source_i[0]+1, source_i[1]+1]
        
        if established:
          if abs(source_i[0]-dest_i[0]) > abs(source_i[1]-dest_i[1]):
            low = int(numpy.ceil(min(source_i[0], dest_i[0])))
            high = int(numpy.ceil(max(source_i[0], dest_i[0])))
            for y in xrange(low, high):
              t = (y - source_i[0]) / (dest_i[0] - source_i[0])
              x = (1.0-t) * source_i[1] + t * dest_i[1]
              keep_mask[y, x] = True
          else:
            low = int(numpy.ceil(min(source_i[1], dest_i[1])))
            high = int(numpy.ceil(max(source_i[1], dest_i[1])))
            for x in xrange(low, high):
              t = (x - source_i[1]) / (dest_i[1] - source_i[1])
              y = (1.0-t) * source_i[0] + t * dest_i[0]
              keep_mask[y, x] = True
    
  # Return...
  return (keep_mask, subpixel)



def cull_lonely(mask):
  """Given a mask this removes all pixels that have no neighbours."""
  # Count how many neighbours each pixel has...
  count = numpy.zeros(mask.shape, dtype=numpy.int32)

  count[1:,:] += mask[:-1,:]
  count[:-1,:] += mask[1:,:]
  count[:,1:] += mask[:,:-1]
  count[:,:-1] += mask[:,1:]

  count[1:,1:] += mask[:-1,:-1]
  count[1:,:-1] += mask[:-1,1:]
  count[:-1,1:] += mask[1:,:-1]
  count[:-1,:-1] += mask[1:,1:]

  # All pixels with zero neighbours get nuked...
  ret = mask.copy()
  ret[numpy.where(count==0)] = False

  return ret
