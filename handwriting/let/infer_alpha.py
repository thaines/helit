# Copyright 2016 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import numpy
import scipy.weave as weave

from utils.start_cpp import start_cpp



def inpaint_stupid(image, mask):
  """Inpaints the masked area using simple linear interpolation from its neighbours."""
  out = image.copy()
  done = numpy.logical_not(mask)
  
  code = start_cpp() + """
  // Keep looping and updating until no change occurs...
   int changes = 1;
   int dir = -1;
   
   while (changes!=0)
   {
    changes = 0;
    
    dir = (dir+1) % 2;
    int startY = (dir==0) ? 0 : Nout[0] - 1;
    int deltaY = (dir==0) ? 1 : -1;
    int startX = (dir==0) ? 0 : Nout[1] - 1;
    int deltaX = (dir==0) ? 1 : -1;
    
    
    for (int y=startY; y>=0 && y<Nout[0]; y += deltaY)
    {
     for (int x=startX; x>=0 && x<Nout[1]; x += deltaX)
     {
      if (MASK2(y, x)!=0)
      {
       bool safeNegY = (y>0) && DONE2(y-1, x)!=0;
       bool safePosY = (y+1<Nout[0]) && DONE2(y+1, x)!=0;
       bool safeNegX = (x>0) && DONE2(y, x-1)!=0;
       bool safePosX = (x+1<Nout[1]) && DONE2(y, x+1)!=0;
      
       int weight = 0;
       float average[3] = {0.0, 0.0, 0.0};
      
       if (safeNegY)
       {
        weight += 1;
        for (int c=0; c<3; c++)
        {
         average[c] += (OUT3(y-1, x, c) - average[c]) / weight;
        }
       }
      
       if (safePosY)
       {
        weight += 1;
        for (int c=0; c<3; c++)
        {
         average[c] += (OUT3(y+1, x, c) - average[c]) / weight;
        }
       }
      
       if (safeNegX)
       {
        weight += 1;
        for (int c=0; c<3; c++)
        {
         average[c] += (OUT3(y, x-1, c) - average[c]) / weight;
        }
       }
      
       if (safePosX)
       {
        weight += 1;
        for (int c=0; c<3; c++)
        {
         average[c] += (OUT3(y, x+1, c) - average[c]) / weight;
        }
       }
      
       if (weight!=0)
       {
        unsigned char avg[3];
        for (int c=0; c<3; c++) avg[c] = (unsigned char)(average[c] + 0.5);
        
        if (DONE2(y, x)!=0)
        {
         float diff = 0.0;
         for (int c=0; c<3; c++)
         {
          diff += fabs(OUT3(y, x, c) - avg[c]);
          OUT3(y, x, c) = avg[c];
         }
        
         if (diff>1e-3) changes += 1;
        }
        else
        {
         changes += 1;
         DONE2(y, x) = 1;
        
         for (int c=0; c<3; c++) OUT3(y, x, c) = avg[c];
        }
       }
      }
     }
    }
   }
  """
  
  weave.inline(code, ['mask', 'out', 'done'])
  
  return out
  
  
  
def infer_alpha_cc(image, mask):
  """Given an image without alpha, as a numpy array [y,x,0 for red, 1 for green, 2 for blue] this returns a new image, with an alpha chanel (Put in after the colours, which are premultiplied.). Makes use of a mask - everything outside is assumed to have an alpha of 0. Uses simple linear interpolation to inpaint the masked areas to get background colour - it is then relativly simple to set every pixel to the lowest alpha that allows its colour to remain identical on the inpainted background plate."""
  
  # First inpaint the area given by the mask via simple interpolation, so we have a background plate...
  bg = inpaint_stupid(image, mask)
  
  # Go through and set alpha so that when compositing over the bg_plate the colour is exactly the same, with it at its minimum value...
  out = numpy.zeros((image.shape[0], image.shape[1], 4), dtype=numpy.uint8)
  
  code = start_cpp() + """
  for (int y=0; y<Nimage[0]; y++)
  {
   for (int x=0; x<Nimage[1]; x++)
   {
    if (MASK2(y, x)!=0)
    {
     // Find the minimum acceptable alpha that still allows the colour to be obtained...
      float min_a = 0.0;
      float delta[3];
      
      for (int c=0; c<3; c++)
      {
       delta[c] = (float)IMAGE3(y, x, c) - (float)BG3(y, x, c);
       
       /*if (fabs(255.0 - BG3(y, x, c))>1e-3) // Commented out on assumption of white backgrounds!
       {
        float pa = delta[c] / (255.0 - BG3(y, x, c));
        if (pa>min_a) min_a = pa;
       }*/
       
       if (fabs(0.0 - BG3(y, x, c))>1e-3)
       {
        float pa = delta[c] / (0.0 - BG3(y, x, c));
        if (pa>min_a) min_a = pa;
       }
      }
      
     // For now use the minimum alpha...
      float a = min_a;
    
     // Solve for the foreground colour...
      float fg[3] = {0.0, 0.0, 0.0};
      if (a>1e-3)
      {
       for (int c=0; c<3; c++)
       {
        fg[c] = ((float)IMAGE3(y, x, c) - (1.0 - a) * BG3(y, x, c)) / a;
       }
      }
      
     // Record the output, premultiplied...
      OUT3(y, x, 3) = a*255.0 + 0.5;
      a = OUT3(y, x, 3) / 255.0;
      
      for (int c=0; c<3; c++)
      {
       int val = a*fg[c] + 0.5;
       if (val<0) val = 0;
       if (val>255) val = 255;
       
       OUT3(y, x, c) = val;
      }
      
     // If we are really close to the background colour we probably are background... so fade towards that - avoids fireflies...
      /*float dist = sqrt(delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2]);
      
      const float low = 16.0;
      const float high = 64.0;
      
      if (dist<high)
      {
       if (a>0.9) // **************
       {
        printf("%i %i alpha = %f\\n", y, x, a);
        printf("image = [%hhu %hhu %hhu]\\n", IMAGE3(y, x, 0), IMAGE3(y, x, 1), IMAGE3(y, x, 2));
        printf("bg = [%hhu %hhu %hhu]\\n", BG3(y, x, 0), BG3(y, x, 1), BG3(y, x, 2));
        printf("delta = [%.3f %.3f %.3f]\\n", delta[0], delta[1], delta[2]);
        printf("out = [%hhu %hhu %hhu %hhu]\\n", OUT3(y, x, 0), OUT3(y, x, 1), OUT3(y, x, 2), OUT3(y, x, 3));
        printf("\\n");
       }
       
       float mult;
       if (dist<low) mult = 0.0;
                else mult = (dist - low) / (high - low);
       
       for (int c=0; c<4; c++) OUT3(y, x, c) *= mult;
      }*/
    }
   }
  }
  """
  
  weave.inline(code, ['image', 'mask', 'bg', 'out'])
  
  return out
