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



class LightCorrectMS(VideoNode):
  """This node estimates the lighting change between the current frame and the previous frame, providing several outputs to communicate this change to future nodes. Estimates the change as a per-channel multiplicative constant, for which it gets an estimate from each pixel. Mean shift is then used to find the mode, as in the paper 'Time_delayed Correlation Analysis for Multi-Camera Activity Understanding' by Loy, Xiang and Gong. There is no guarantee that after this values will remain within [0,1]. Also has a mode of operation where instead of the previous it fetches a frame from elsewhere - it does not indicate a dependency on the other source so anything can happen - its primary aim is to allow a loop with a background subtraction node such that it uses the current, or previous, background estimate. Requires that the input colour model be from the colour_bias node, as it makes assumptions dependent on that model."""
  def __init__(self, scale = 8.0/255.0, limit = 0.2, lowLimit = 8.0/255.0, incColour = False):
    """The scale is the width of the window used for mean shift, limit is a cap on how large a change before a pixel is ignored and lowLimit is a limit on how low a pixel value to consider, to avoid photon noise."""
    self.video = None
    self.channel = 0

    self.other = None
    self.otherChannel = 0

    self.scale = scale
    self.limit = limit
    self.lowLimit = lowLimit

    self.curr = None
    self.prev = None
    self.matrix = numpy.identity(4, dtype=numpy.float32) # Previous to current.
    
    self.channels = 3 if incColour else 1

    self.temp = None

  def width(self):
    return self.video.width()

  def height(self):
    return self.video.height()

  def fps(self):
    return self.video.fps()

  def frameCount(self):
    return self.video.frameCount()


  def inputCount(self):
    return 2

  def inputMode(self, channel=0):
    return MODE_RGB

  def inputName(self, channel=0):
    if self.channel==0: return 'The input image, to be compared with the previous image.'
    else: return "An alternate 'previous' image, to get lighting corrections relative to something else, typically the mode of a background estimation from a background subtraction algorithm. Note that this video is not listed in the dependencies as you typically want a circularity, and the previous estimate should not be an issue."

  def source(self, toChannel, video, videoChannel=0):
    if toChannel==0:
      self.video = video
      self.channel = videoChannel
    else:
      self.other = video
      self.otherChannel = videoChannel


  def dependencies(self):
    return [self.video]

  def nextFrame(self):
    # Reset some buffers....
    self.invMatrix = None
    self.prevCorrect = None
    self.currCorrect = None

    # Update the previous and current...
    if self.other==None:
      self.prev = self.curr
      self.curr = self.video.fetch(self.channel)

      if self.curr==None: return False
      if self.prev==None: return True

      self.curr = self.curr.copy()
    else:
      self.prev = self.other.fetch(self.otherChannel)
      self.curr = self.video.fetch(self.channel)

      if self.curr==None: return False
      if self.prev==None: return True


    # Make sure the temporary is the right size...
    pixelCount = self.width() * self.height()
    if self.temp==None or self.temp.shape[0]!=pixelCount:
      self.temp = numpy.empty((pixelCount,3), dtype=numpy.float32)


    # Collect for each channel the ratios that are essentially estimates of the lighting change...
    codeE = start_cpp() + """
    int count = 0;

    for (int y=0;y<Ncurr[0];y++)
    {
     for (int x=0;x<Ncurr[1];x++)
     {
      if ((CURR3(y,x,0)>lowLimit)&&(PREV3(y,x,0)>lowLimit))
      {
       float diff = 0.0;
       for (int c=0;c<3;c++)
       {
        float cVal = CURR3(y,x,c);
        float pVal = PREV3(y,x,c);
        diff += fabs(cVal - pVal);
        TEMP2(count,c) = cVal / pVal;
       }

       if (diff<limit) count += 1;
      }
     }
    }

    return_val = count;
    """

    curr = self.curr
    prev = self.prev
    temp = self.temp
    limit = self.limit
    lowLimit = self.lowLimit

    count = weave.inline(codeE, ['curr', 'prev', 'temp', 'limit', 'lowLimit'])
    
    if count<8:
      self.matrix[:,:] = numpy.identity(4, dtype=numpy.float32)
      return True


    # Sort each channel, ready for mean shift...
    for channel in xrange(3): self.temp[:count,channel].sort()



    # Mean shift for each channel in turn...
    codeMS = start_cpp() + """
    // Parameters and some initialisation...
     const float epsilon = 1e-3;
     const int maxIter = 32;
     const float hsm = 3.0;
     const float winWidth = hsm * float(scale);
       
    // Iterate the channels...
     for (int channel=0; channel<channels; channel++)
     {
      // Median seems a good initial estimate - as long as at least half the pixels are background at the same time in both images it is guaranteed to be a good estimate...
       float estimate = TEMP2(count/2,channel); 
       float minVal = estimate - winWidth;
       float maxVal = estimate + winWidth;

      // Use a pair of binary searches to initialise the range to sum over...
       int low = 0;
       int other = count-1;
       while (low+1<other)
       {
        int half = (low+other)/2;
        if (TEMP2(half,channel)<minVal) low   = half;
                                   else other = half;
       }

       other = low+1;
       int high = count-1;
       while (other+1<high)
       {
        int half = (other+high)/2;
        if (TEMP2(half,channel)>maxVal) high  = half;
                                   else other = half;
       }

      // Iterate and do the mean shift steps...
       for (int iter=0;iter<maxIter;iter++)
       {
        minVal = estimate - winWidth;
        maxVal = estimate + winWidth;

        // Update the low and high values by simply offsetting them until correct...
         for (;low>0;--low)
         {
          if (TEMP2(low,channel)<minVal) break;
         }

         for (;low<count-1;++low)
         {
          if (TEMP2(low,channel)>minVal) break;
         }

         for (;high<count-1;++high)
         {
          if (TEMP2(high,channel)>maxVal) break;
         }
         for (;high>0;--high)
         {
          if (TEMP2(high,channel)<maxVal) break;
         }

         if (low>=high) break; // No data - give up.

        // Iterate the relevant values and add them to the kernel...
         float newEst = 0.0;
         float weightEst = 0.0;

         float weight;
         float prevVal = 1e100;
         float scale2 = float(scale)*float(scale);
         for (int i=low;i<=high;i++)
         {
          if (fabs(prevVal-TEMP2(i,channel))>epsilon)
          {
           prevVal = TEMP2(i,channel);
           float delta = prevVal - estimate;
           // weight = exp(-0.5*delta*delta/scale2); // Gaussian option
           weight = float(scale) / (M_PI * (delta*delta + scale2)); // Cauchy option
          }

          weightEst += weight;
          newEst += (TEMP2(i,channel)-newEst) * weight / weightEst;
         }

        // Update the estimate, exit if change is minor...
         bool done = fabs(newEst-estimate) < epsilon;
         estimate = newEst;
         if (done) break;
       }
        
      // Store it...
       MATRIX2(channel, channel) = estimate;
     }
     
     for (int channel=channels; channel<3; channel++)
     {
      MATRIX2(channel, channel) = 1,0;
     }
    """

    scale = self.scale
    matrix = self.matrix
    channels = self.channels

    weave.inline(codeMS, ['temp', 'count', 'scale', 'matrix', 'channels'])

    return True


  def outputCount(self):
    return 4

  def outputMode(self, channel=0):
    if channel<2: return MODE_MATRIX
    else: return MODE_RGB

  def outputName(self, channel=0):
    if channel==0: return 'A homogenous matrix to manipulate the rgb value of each pixel - converts the previous frame to the current frames lighting levels.'
    elif channel==1: return 'Inverse of channel 0: 4x4 homogeonous matrix on colour to convert from the current frame to the previous.'
    elif channel==2: return 'Previous image converted to lighting of current image'
    elif channel==3: return 'Current image converted to lighting of previous image'

  def fetch(self, channel=0):
    if channel==0:
      return self.matrix

    elif channel==1:
      if self.invMatrix==None:
        self.invMatrix = self.matrix.copy()
        for c in xrange(3): self.invMatrix[c,c] = 1.0/self.matrix[c,c]
      return self.invMatrix

    elif channel==2:
      if self.prevCorrect==None:
        if self.prev==None: # First frame - fallback.
          self.prevCorrect = self.curr.copy()
        else:
          self.prevCorrect = self.prev.copy()
          for c in xrange(3):
            self.prevCorrect[:,:,c] *= self.matrix[c,c]
      return self.prevCorrect

    elif channel==3:
      if self.currCorrect==None:
        self.currCorrect = self.curr.copy()
        for c in xrange(3):
          self.currCorrect[:,:,c] *= 1.0/self.matrix[c,c]
      return self.currCorrect
