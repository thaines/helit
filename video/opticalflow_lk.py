# Copyright 2012 Tom SF Haines

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.



import numpy
import scipy.weave as weave

from utils.start_cpp import start_cpp

from video_node import *



class OpticalFlowLK(VideoNode):
  """Optical flow using Lucas & Kanade - has a pyramid and only does one iteration per pyramid level by default. Uses a median filter for regularisation. Simple, not horrifically slow but obviously nothing amazing - basically the original algorithm for translation only."""
  def __init__(self):
    self.video = None
    self.channel = 0
    self.prev = None
    self.prevChannel = 0
    self.mask = None
    self.maskChannel = 0

    self.pyramid = None

    # All the default parameters...
    self.doPyramid = True # Decides if it makes a pyramid or not.
    self.minDimCap = 32 # Keeps making levels of the pyramid until both dimensions drop below this.
    self.pyramidSD = 0.8 # Strength of the anti-aliasing Gaussian blur applied to each level of the pyramid.

    self.iters = 1 # Number of iterations per pyramid level.
    self.radiusLK = 1 # Radius of the window used for each Lucas-Kanade iteration.
    self.radiusMF = 1 # Radius of the window used for each median filter step.

  def width(self):
    return self.video.width()

  def height(self):
    return self.video.height()

  def fps(self):
    return self.video.fps()

  def frameCount(self):
    return self.video.frameCount()


  def inputCount(self):
    return 3

  def inputMode(self, channel=0):
    if channel!=2: return MODE_RGB
    else: return MODE_MASK

  def inputName(self, channel=0):
    if channel==0: return 'Next frame - optical flow is calculated from this frame to the previous frame and then negated.'
    elif channel==1: return 'Optional replacement for the previous frame, instead of using the actual previous frame. Allows for easy integration with a lighting correction module.'
    else: return "Optional mask - it only computes optical flow where the mask is true, potentially saving a lot of time. (Note that it takes an 'or' approach for the pyramid, so areas outside the mask will still get values.)"

  def source(self, toChannel, video, videoChannel=0):
    if toChannel==0:
      self.video = video
      self.channel = videoChannel
    elif toChannel==1:
      self.prev = video
      self.prevChannel = videoChannel
    else:
      self.mask = video
      self.maskChannel = videoChannel


  def dependencies(self):
    ret = [self.video]
    if self.prev!=None: ret += [self.prev]
    if self.mask!=None: ret += [self.mask]
    return ret

  def nextFrame(self):
    # First time this is called we need to setup the data structures, ready to be filled in...
    if self.pyramid==None:
      self.__setup_ds()

    # Fill in the pyramids - what is involved depends on if we are supplied with a previous or not...
    if self.prev==None:
      # Rolling - means we can swap the pyramids and then rebuild only for the newest image...
      swap = self.current
      self.current = self.previous
      self.previous = swap

      c = self.video.fetch(self.channel)
      self.__buildPyramid(c, self.current)

    else:
      # Previous image is being provided externally - have to rebuild both pyramids...
      c = self.video.fetch(self.channel)
      p = self.prev.fetch(self.prevChannel)

      self.__buildPyramid(c, self.current)
      self.__buildPyramid(p, self.previous)

    # If there is a mask we need to fetch it and build a pyramid, otherwise just set the entire pyramid true and calculate for all pixels...
    if self.mask!=None:
      m = self.mask.fetch(self.maskChannel)
      self.__buildMaskPyramid(m, self.maskPyramid)
    else:
      for l in xrange(len(self.maskPyramid)):
        self.maskPyramid[l][:,:] = 1

    # Loop the pyramid and handle each level in turn...
    self.uv[:self.pyramid[-1][1],:self.pyramid[-1][0],:] = 0.0

    for l in xrange(len(self.pyramid)-1,-1,-1):
      # Iterations at this level...
      for _ in xrange(self.iters):
        self.__do_iter(self.current[l], self.previous[l], self.maskPyramid[l])
        self.__do_median(self.maskPyramid[l])

      # Upscale uv map...
      if l!=0:
        base = self.uv[:self.current[l].shape[0],:self.current[l].shape[1],:]
        base *= 2.0
        base = numpy.repeat(numpy.repeat(base,2,axis=1),2,axis=0)
        self.uv[:base.shape[0],:base.shape[1],:] = base

    # The map just generated is actually going backwards in time - reverse!..
    self.uv *= -1.0
    return True


  def outputCount(self):
    return 1

  def outputMode(self, channel=0):
    return MODE_FLOW

  def outputName(self, channel=0):
    return 'Optical flow - the vectors from the current frame to the previous frame, negated so they appear to go forwards in time.'

  def fetch(self, channel=0):
    return self.uv


  def __setup_ds(self):
    # For the pyramids we need the frame sizes - calculate...
    self.pyramid = []
    self.pyramid.append((self.video.width(), self.video.height()))
    if self.doPyramid:
      while (self.pyramid[-1][0] > self.minDimCap) or (self.pyramid[-1][1] > self.minDimCap):
        half = [self.pyramid[-1][0]//2,self.pyramid[-1][1]//2]
        if (self.pyramid[-1][0]%2)==1: half[0] += 1
        if (self.pyramid[-1][1]%2)==1: half[1] += 1
        self.pyramid.append(tuple(half))

    # We need two pyramids - one for the current frame, one for the previous frame...
    self.current = map(lambda dim: numpy.empty((dim[1], dim[0], 3), dtype=numpy.float32), self.pyramid)
    self.previous = map(lambda dim: numpy.empty((dim[1], dim[0], 3), dtype=numpy.float32), self.pyramid)

    # We also need a mask pyramid, to save on computation if a mask is provided...
    self.maskPyramid = map(lambda dim: numpy.empty((dim[1], dim[0]), dtype=numpy.uint8), self.pyramid)

    # The output - a uv (u = change in x, v = change in y.) map...
    self.uv = numpy.zeros((self.video.height(),self.video.width(),2), dtype=numpy.float32)

    # Temporary storage used at various points...
    self.image = numpy.empty((self.video.height(), self.video.width(), 3), dtype=numpy.float32)
    sizeMF = 1 + 2 * self.radiusMF
    self.window = numpy.empty((sizeMF,sizeMF),dtype=numpy.float32)


  def __buildPyramid(self, base, pyramid):
    """Given an image and a pyramid as a list of images, largest first and of the same size as image, this fills in the images with a Gaussian pyramid."""
    codeBlur = start_cpp() + """
    // Calculate the filter - we just use 3 points as its a very tiny blur...
     float filter[3];
     filter[0] = exp(-0.5/(strength*strength));
     filter[1] = 1.0;
     filter[2] = filter[0];

     float div = filter[0] + filter[1] + filter[2];
     for (int f=0;f<3;f++) filter[f] /= div;


    // Apply in the vertical dimension, going from img to temp...
     for (int x=0;x<Nimg[1];x++)
     {
      for (int c=0;c<3;c++)
      {
       TEMP3(0,x,c) = (filter[0]+filter[1])*IMG3(0,x,c) + filter[2]*IMG3(1,x,c);
      }
     }

     int ym1 = Nimg[0] - 1;
     for (int y=1;y<ym1;y++)
     {
      for (int x=0;x<Nimg[1];x++)
      {
       for (int c=0;c<3;c++)
       {
        TEMP3(y,x,c) = filter[0]*IMG3(y-1,x,c) + filter[1]*IMG3(y,x,c) + filter[2]*IMG3(y+1,x,c);
       }
      }
     }

     for (int x=0;x<Nimg[1];x++)
     {
      for (int c=0;c<3;c++)
      {
       TEMP3(ym1,x,c) = filter[0]*IMG3(ym1-1,x,c) + (filter[1]+filter[2])*IMG3(ym1,x,c);
      }
     }


    // Apply in the horizontal dimension, going from temp to img...
     for (int y=0;y<Nimg[0];y++)
     {
      for (int c=0;c<3;c++)
      {
       IMG3(y,0,c) = (filter[0]+filter[1])*TEMP3(y,0,c) + filter[2]*TEMP3(y,1,c);
      }
     }

     int xm1 = Nimg[1] - 1;
     for (int y=0;y<Nimg[0];y++)
     {
      for (int x=1;x<xm1;x++)
      {
       for (int c=0;c<3;c++)
       {
        IMG3(y,x,c) = filter[0]*TEMP3(y,x-1,c) + filter[1]*TEMP3(y,x,c) + filter[2]*TEMP3(y,x+1,c);
       }
      }
     }

     for (int y=0;y<Nimg[0];y++)
     {
      for (int c=0;c<3;c++)
      {
       IMG3(y,xm1,c) = filter[0]*TEMP3(y,xm1-1,c) + (filter[1]+filter[2])*TEMP3(y,xm1,c);
      }
     }
    """

    codeHalf = start_cpp() + """
    // Iterate the output image and calculate each pixel in turn...
     for (int y=0;y<NbOut[0];y++)
     {
      int sy = y*2;
      bool safeY = sy+1<NbOut[0];

      for (int x=0;x<NbOut[1];x++)
      {
       int sx = x*2;
       bool safeX = sx+1<NbOut[1];

       float div = 1.0;
       for (int c=0;c<3;c++)
       {
        BOUT3(y,x,c) = BIN3(sy,sx,c);
       }

       if (safeX)
       {
        div += 1.0;
        for (int c=0;c<3;c++) BOUT3(y,x,c) += BIN3(sy,sx+1,c);
       }

       if (safeY)
       {
        div += 1.0;
        for (int c=0;c<3;c++) BOUT3(y,x,c) += BIN3(sy+1,sx,c);
       }

       if (safeX&&safeY)
       {
        div += 1.0;
        for (int c=0;c<3;c++) BOUT3(y,x,c) += BIN3(sy+1,sx+1,c);
       }

       for (int c=0;c<3;c++) BOUT3(y,x,c) /= div;
      }
     }
    """

    temp = self.image
    strength = self.pyramidSD

    pyramid[0][:,:,:] = base
    img = pyramid[0][:,:,:]
    weave.inline(codeBlur, ['img','temp','strength'])

    for l in xrange(1,len(pyramid)):
      bIn = pyramid[l-1]
      bOut = pyramid[l]
      img = bOut

      weave.inline(codeHalf, ['bIn','bOut'])
      weave.inline(codeBlur, ['img','temp','strength'])

  def __buildMaskPyramid(self, mask, pyramid):
    """Given a mask and a pyramid of masks makes a pyramid, where it uses the or operation for combining flags."""

    code = start_cpp() + """
    // Make curr all false...
     for (int y=0;y<Ncurr[0];y++)
     {
      for (int x=0;x<Ncurr[1];x++) CURR2(y,x) = 0;
     }

    // Iterate prev, and update curr...
    for (int y=0;y<Nprev[0];y++)
    {
     for (int x=0;x<Nprev[1];x++)
     {
      if (PREV2(y,x)!=0)
      {
       CURR2(y/2,x/2) = 1;
      }
     }
    }
    """

    pyramid[0][:,:] = mask
    for l in xrange(1,len(pyramid)):
      prev = pyramid[l-1]
      curr = pyramid[l]

      weave.inline(code, ['prev','curr'])


  def __do_iter(self, iFrom, iTo, mask):
    """Given a from image and a to image this does a single LK iteration, starting from and updating the values in self.uv. If images are smaller than self.uv then it just uses the corner - means same uv object can be used throughout pyramid construction."""
    support_code = start_cpp() + """
    // Given a t value in [0,1] calculates the weights of the 4 pixels for a bicubic spline and writes them into out, it also writes into dOut the weights to get the splines differential with respect to t.
    void BicubicMult(float t, float out[4], float dOut[4])
    {
     float t2 = t*t;
     float t3 = t2*t;

     out[0] =    -0.5*t +     t2 - 0.5*t3;
     out[1] = 1.0       - 2.5*t2 + 1.5*t3;
     out[2] =     0.5*t + 2.0*t2 - 1.5*t3;
     out[3] =            -0.5*t2 + 0.5*t3;

     dOut[0] = -0.5 + 2.0*t - 1.5*t2;
     dOut[1] =       -5.0*t + 4.5*t2;
     dOut[2] =  0.5 + 4.0*t - 4.5*t2;
     dOut[3] =           -t + 1.5*t2;
    }

    // This does bicubic interpolation of an image, getting both values and differentials, and handling boundary conditions. Input must include the output of calls to BicubicMult for both directions - this encodes the fractional part of the coordinate. The user provides the integer part.
     void Bicubic(PyArrayObject * image, int y, float multY[4], float dMultY[4], int x, float multX[4], float dMultX[4], float rgb[3], float rgbDy[3], float rgbDx[3])
     {
      // Handle coordinates, doing boundary checking - we use repetition at the borders, which makes it a simple matter of coordinate clamping at the boundaries...Y
       int coordY[4];
       coordY[0] = y-1;
       for (int i=1;i<4;i++) coordY[i] = coordY[i-1] + 1;

       for (int i=0;i<4;i++)
       {
        if (coordY[i]>=0) break;
        coordY[i] = 0;
       }

       for (int i=3;i>=0;i--)
       {
        if (coordY[i]<image->dimensions[0]) break;
        coordY[i] = image->dimensions[0]-1;
       }

       int coordX[4];
       coordX[0] = x-1;
       for (int i=1;i<4;i++) coordX[i] = coordX[i-1] + 1;

       for (int i=0;i<4;i++)
       {
        if (coordX[i]>=0) break;
        coordX[i] = 0;
       }

       for (int i=3;i>=0;i--)
       {
        if (coordX[i]<image->dimensions[1]) break;
        coordX[i] = image->dimensions[1]-1;
       }

      // Apply in both dimensions to get value sequences interpolated in both, from which you would typically inteprolate the value in a second step - needed due to calculation of differentials...
       float iy[4][3]; // Position, rgb. y=dimension you index with.
       float ix[4][3]; // ", but with x.
       bzero(iy,sizeof(float)*4*3);
       bzero(ix,sizeof(float)*4*3);

       for (int v=0;v<4;v++)
       {
        char * baseV = image->data + coordY[v]*image->strides[0];
        for (int u=0;u<4;u++)
        {
         float * val = (float*)(baseV + coordX[u]*image->strides[1]);

         for (int c=0;c<3;c++)
         {
          iy[v][c] += multX[u] * val[c];
          ix[u][c] += multY[v] * val[c];
         }
        }
       }

      // Use one dimension and a further step to get the value...
       bzero(rgb,sizeof(float)*3);
       for (int u=0;u<4;u++)
       {
        for (int c=0;c<3;c++) rgb[c] += ix[u][c] * multX[u];
       }

      // Use both dimensions followed by a differential step to get the differentials for both dx and dy...
       bzero(rgbDy,sizeof(float)*3);
       for (int v=0;v<4;v++)
       {
        for (int c=0;c<3;c++) rgbDy[c] += iy[v][c] * dMultY[v];
       }

       bzero(rgbDx,sizeof(float)*3);
       for (int u=0;u<4;u++)
       {
        for (int c=0;c<3;c++) rgbDx[c] += ix[u][c] * dMultX[u];
       }
     }
    """

    code = start_cpp(support_code) + """
    // Iterate over the pixels and calculate as estimate for each...
     for (int y=0;y<NiFrom[0];y++)
     {
      for (int x=0;x<NiFrom[1];x++)
      {
       if (MASK2(y,x)!=0)
       {
        // Get the range to search - to avoid sampling values outside the image (For the from image - to image is allowed to go outside the range, as handled by the interpolation functions.)...
         int yStart = y - radius;
         int yEnd   = y + radius;
         int xStart = x - radius;
         int xEnd   = x + radius;

         if (yStart<0) yStart = 0;
         if (yEnd>=NiFrom[0]) yEnd = NiFrom[0] - 1;
         if (xStart<0) xStart = 0;
         if (xEnd>=NiFrom[1]) xEnd = NiFrom[1] - 1;

        // Get the offset from uv, split into integer and fractional parts and calculate the weights for the bicubic interpolation...
         int oy = int(UV3(y,x,0));
         float ty = UV3(y,x,0) - oy;
         float multY[4];
         float dMultY[4];
         BicubicMult(ty, multY, dMultY);

         int ox = int(UV3(y,x,1));
         float tx = UV3(y,x,1) - ox;
         float multX[4];
         float dMultX[4];
         BicubicMult(tx, multX, dMultX);

        // Calculate the b value and structural tensor, simultaneously, to avoid computing derivatives repeatedly...
         float st[3] = {0.0,0.0,0.0}; // Linearised symmetric matrix - [0][0], [0][1]/[1][0], [1][1].
         float b[2] = {0.0,0.0};

         for (int v=yStart;v<=yEnd;v++)
         {
          for (int u=xStart;u<=xEnd;u++)
          {
           if (MASK2(v,u)!=0)
           {
            // Get the value in the from image...
             float * from = (float*)(iFrom_array->data + v*iFrom_array->strides[0] + u*iFrom_array->strides[1]);

            // Get the value and differential in the to image...
             float rgb[3];
             float rgbDy[3];
             float rgbDx[3];
             Bicubic(iTo_array, v+oy, multY, dMultY, u+ox, multX, dMultX, rgb, rgbDy, rgbDx);

            // Loop the colour channels - same calculations for each...
             for (int c=0;c<3;c++)
             {
              // Update the structural tensor...
               st[0] += rgbDy[c] * rgbDy[c];
               st[1] += rgbDx[c] * rgbDy[c];
               st[2] += rgbDx[c] * rgbDx[c];

              // Update b...
               float diff = from[c] - rgb[c];
               b[0] += rgbDy[c] * diff;
               b[1] += rgbDx[c] * diff;
             }
           }
          }
         }

        // Invert the structural tensor, solve the equation, update the uv entry...
         double det = double(st[0])*double(st[2]) - double(st[1])*double(st[1]);
         if (fabs(det)>1e-9)
         {
          float temp = st[0];
          st[0] = st[2];
          st[2] = temp;
          st[1] *= -1.0;

          st[0] /= det;
          st[1] /= det;
          st[2] /= det;

          // st is now inverted - easy matter to calculate the change...
           float dv = st[0]*b[0] + st[1]*b[1];
           float du = st[1]*b[0] + st[2]*b[1];

          // Only apply the change if it is sensible - approximation is only good for a pixel or so, so ignore if greater than 2 as it being crazy...
          float changeSqr = dv*dv + du*du;
           if (changeSqr<(2*2))
           {
            UV3(y,x,0) += dv;
            UV3(y,x,1) += du;
           }
         }
       }
      }
     }
    """

    uv = self.uv
    radius = self.radiusLK

    weave.inline(code, ['iFrom', 'iTo', 'mask', 'uv', 'radius'], support_code=support_code)


  def __do_median(self, mask):
    """Applys a median filter to self.uv - pretty simple really. Areas outside the mask are ignored."""

    code = start_cpp() + """
    int size = radius*2 + 1;

    // Iterate and calculate the median for each pixel, writting the output into temp...
     for (int y=0;y<Nmask[0];y++)
     {
      for (int x=0;x<Nmask[1];x++)
      {
       if (MASK2(y,x)!=0)
       {
        // Get ranges, bound checked...
         int startV = y - radius;
         int endV = y + radius;
         int startU = x - radius;
         int endU = x + radius;

         if (startV<0) startV = 0;
         if (endV>=Nmask[0]) endV = Nmask[0]-1;
         if (startU<0) startU = 0;
         if (endU>=Nmask[1]) endU = Nmask[1]-1;

        // Zero out the window, so the distances may be summed in...
         for (int v=startV;v<=endV;v++)
         {
          int wv = v - startV;
          for (int u=startU;u<=endU;u++)
          {
           int wu = u - startU;
           WIN2(wv,wu) = 0.0;
          }
         }

        // Calculate the distances for each entry - take care to avoid duplicate calculation, even though it makes for some messy code...
         for (int v=startV;v<=endV;v++)
         {
          int wv = v - startV;
          for (int u=startU;u<=endU;u++)
          {
           int wu = u - startU;

           if (MASK2(v,u)!=0)
           {
            int ov = v;
            int ou = u;
            while (true)
            {
             ou += 1;
             if (ou>endU)
             {
              ou = startU;
              ov += 1;
              if (ov>endV) break;
             }
             if (MASK2(ov,ou)==0) continue;

             int wov = ov - startV;
             int wou = ou - startU;

             float deltaV = UV3(ov,ou,0) - UV3(v,u,0);
             float deltaU = UV3(ov,ou,1) - UV3(v,u,1);
             float dist = sqrt(deltaU*deltaU + deltaV*deltaV);

             WIN2(wv,wu)   += dist;
             WIN2(wov,wou) += dist;
            }
           }
          }
         }

        // Find and select the best entry...
         float best = 1e100;
         for (int v=startV;v<=endV;v++)
         {
          int wv = v - startV;
          for (int u=startU;u<=endU;u++)
          {
           int wu = u - startU;

           if (MASK2(v,u)!=0)
           {
            if (WIN2(wv,wu)<best)
            {
             best = WIN2(wv,wu);
             TEMP3(y,x,0) = UV3(v,u,0);
             TEMP3(y,x,1) = UV3(v,u,1);
            }
           }
          }
         }
       }
      }
     }

    // Copy from temp into image...
     for (int y=0;y<Nmask[0];y++)
     {
      for (int x=0;x<Nmask[1];x++)
      {
       if (MASK2(y,x)!=0)
       {
        UV3(y,x,0) = TEMP3(y,x,0);
        UV3(y,x,1) = TEMP3(y,x,1);
       }
      }
     }

    """

    uv = self.uv
    temp = self.image
    win = self.window
    radius = self.radiusMF

    weave.inline(code, ['mask','uv','temp','win','radius'])
