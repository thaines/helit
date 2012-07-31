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



five_word_colours = [(0.5,0.5,0.5), (1.0,0.0,0.0), (0.5,1.0,0.0), (0.0,1.0,1.0), (0.5,0.0,1.0)]



class FiveWord(VideoNode):
  """Quantises a video stream into five words per location, specifically 4 directions and no motion. Divides the image into a grid of locations and has a simple vote in each grid cell, with a threshold to decide the difference between moving and not. Has 2 inputs - a flow field from the optical flow and a mask from the background subtraction. Assumes that grid size is a multiple of the dimensions """
  def __init__(self, threshold=0.25, gridSize=8):
    self.threshold = threshold
    self.gridSize = gridSize

    self.flow = None
    self.flowChannel = 0

    self.mask = None
    self.maskChannel = 0

    self.output = None

  def width(self):
    return self.flow.width() / self.gridSize

  def height(self):
    return self.flow.height() / self.gridSize

  def fps(self):
    return self.flow.fps()

  def frameCount(self):
    return self.flow.frameCount()


  def inputCount(self):
    return 2

  def inputMode(self, channel=0):
    if channel==0: return MODE_FLOW
    else: return MODE_MASK

  def inputName(self, channel=0):
    if channel==0: return 'The flow field providing motion.'
    else: return 'The mask indicating foregound areas.'

  def source(self, toChannel, video, videoChannel=0):
    if toChannel==0:
      self.flow = video
      self.flowChannel = videoChannel
    else:
      self.mask = video
      self.maskChannel = videoChannel


  def wordCount(self):
    return 5

  def suggestedColours(self):
    return five_word_colours


  def dependencies(self):
    return [self.flow, self.mask]

  def nextFrame(self):
    # If first call initialise data structures...
    if self.output==None:
      self.output = numpy.empty((self.height(), self.width()), dtype=numpy.int32)
      self.vote = numpy.empty((self.height(), self.width(), 6), dtype=numpy.int32)

    # Get the inputs...
    flow = self.flow.fetch(self.flowChannel)
    mask = self.mask.fetch(self.maskChannel)

    # Do the work...
    code = start_cpp() + """
    // Zero the vote array - first 5 entrys map to the words, 6th entry is the vote for 'no-word'...
     for (int y=0;y<Nvote[0];y++)
     {
      for (int x=0;x<Nvote[1];x++)
      {
       for (int c=0;c<Nvote[2];c++) VOTE3(y,x,c) = 0;
      }
     }

    // Iterate the pixels, make them each cast a vote...
     for (int y=0;y<Nflow[0];y++)
     {
      for (int x=0;x<Nflow[1];x++)
      {
       int vy = y / gridSize;
       int vx = x / gridSize;

       if (MASK2(y,x)!=0)
       {
        float speed = sqrt(FLOW3(y,x,0)*FLOW3(y,x,0) + FLOW3(y,x,1)*FLOW3(y,x,1));
        if (speed<threshold) VOTE3(vy,vx,0) += 1;
        else
        {
         if (fabs(FLOW3(y,x,0))>fabs(FLOW3(y,x,1)))
         {
          // dy is greater than dx...
           if (FLOW3(y,x,0)>0.0) VOTE3(vy,vx,2) += 1;
           else VOTE3(vy,vx,4) += 1;
         }
         else
         {
          // dx is greater than dy...
           if (FLOW3(y,x,1)>0.0) VOTE3(vy,vx,1) += 1;
           else VOTE3(vy,vx,3) += 1;
         }
        }
       }
       else VOTE3(vy,vx,5) += 1;
      }
     }

    // Count the votes, declare the winners...
     for (int y=0;y<Nvote[0];y++)
     {
      for (int x=0;x<Nvote[1];x++)
      {
       int maxIndex = 0;
       for (int c=1;c<Nvote[2];c++)
       {
        if (VOTE3(y,x,c)>VOTE3(y,x,maxIndex)) maxIndex = c;
       }

       if (maxIndex==5) maxIndex = -1;
       OUTPUT2(y,x) = maxIndex;
      }
     }
    """

    output = self.output
    vote = self.vote

    threshold = self.threshold
    gridSize = self.gridSize

    weave.inline(code, ['flow', 'mask', 'output', 'vote', 'threshold', 'gridSize'])

    return True


  def outputCount(self):
    return 1

  def outputMode(self, channel=0):
    return MODE_WORD

  def outputName(self, channel=0):
    return 'Grid of words for each grid location. -1 for no word, 0 for stationary object, then 1 for +ve x, 2 for +ve y, 3 for -ve x and 4 for -ve y.'

  def fetch(self, channel=0):
    return self.output
