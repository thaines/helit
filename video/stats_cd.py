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



import os.path

import numpy
import scipy.weave as weave
import cv

from utils.cvarray import cv2array
from utils.start_cpp import start_cpp
from video_node import *



class StatsCD(VideoNode):
  """Calculates the stats required by the changedetection.net website for analysing a background subtraction algorithm, given the data in the format they provide."""
  def __init__(self, path):
    """You provide the path of the dataset and then the input to channel 0 is the background subtraction mask."""
    self.path = path
    
    # Open the region of interest and convert it into a mask we can use...
    roi = cv.LoadImage(os.path.join(path, 'ROI.bmp'))
    self.roi = (cv2array(roi)!=0).astype(numpy.uint8)
    if len(self.roi.shape)==3: self.roi = self.roi[:,:,0]

    # Get the temporal range of the frames we are to analyse...
    data = open(os.path.join(path, 'temporalROI.txt'), 'r').read()
    tokens = data.split()
    assert(len(tokens)==2)
    self.start = int(tokens[0])
    self.end = int(tokens[1]) # Note that these are inclusive.
    
    # Confusion matrix - first index is truth, second guess, bg=0, fg=1...
    self.con = numpy.zeros((2,2), dtype=numpy.int32)
    
    # Shadow matrix - for all pixels that are in shadow records how many are background (0) and how many are foreground (1)...
    self.shadow = numpy.zeros(2, dtype=numpy.int32)
    
    # Basic stuff...
    self.frame = 0

    self.video = None
    self.channel = 0


  def width(self):
    return self.video.width()

  def height(self):
    return self.video.height()

  def fps(self):
    return self.video.fps()

  def frameCount(self):
    return self.video.frameCount()
  

  def inputCount(self):
    return 1

  def inputMode(self, channel=0):
    return MODE_MASK

  def inputName(self, channel=0):
    return 'Estimated mask.'

  def source(self, toChannel, video, videoChannel=0):
    self.video = video
    self.channel = videoChannel


  def dependencies(self):
    return [self.video]
  
  
  def nextFrame(self):
    # Increase frame number - need to be 1 based for this...
    self.frame += 1
    
    if self.frame>=self.start and self.frame<=self.end:
      # Fetch the provided mask...
      mask = self.video.fetch(self.channel)
      
      # Load the ground truth file...
      fn = os.path.join(self.path, 'groundtruth/gt%06i.png'%self.frame)
      gt = cv2array(cv.LoadImage(fn))
      gt = gt.astype(numpy.uint8)
      if len(gt.shape)==3: gt = gt[:,:,0]
    
      # Loop the pixels and analyse each one, summing into the confusion matrix...
      code = start_cpp() + """
      for (int y=0; y<Ngt[0]; y++)
      {
       for (int x=0; x<Ngt[1]; x++)
       {
        if ((ROI2(y,x)!=0)&&(GT2(y,x)!=170))
        {
         int t = (GT2(y,x)==255)?1:0;
         int g = (MASK2(y,x)!=0)?1:0;
        
         CON2(t,g) += 1;
         
         if (GT2(y,x)==50)
         {
          SHADOW1(g) += 1;
         }
        }
       }
      }
      """
    
      roi = self.roi
      con = self.con
      shadow = self.shadow
      
      weave.inline(code, ['mask', 'gt', 'roi', 'con', 'shadow'])
      
    return self.frame<=self.end

  
  def outputCount(self):
    return 0
  
  
  def getCon(self):
    """Returns the confusion matrix - [truth, guess], 0=background, 1=foreground."""
    return self.con
  
  def getRecall(self):
    div = self.con[1,:].sum()
    if div==0: return 0.0
    else: return self.con[1,1] / float(div)
  
  def getSpecficity(self):
    return self.con[0,0] / float(self.con[0,:].sum())
  
  def getFalsePosRate(self):
    return self.con[0,1] / float(self.con[0,:].sum())

  def getFalseNegRate(self):
    return self.con[1,0] / float(self.con[1,:].sum())
  
  def getPrecision(self):
    div = self.con[:,1].sum()
    if div==0: return 0.0
    else: return self.con[1,1] / float(div)
  
  def getFMeasure(self):
    recall = self.getRecall()
    precision = self.getPrecision()
    div = precision + recall
    if div<1e-12: return 0.0
    else: return (2.0 * precision * recall) / div
  
  def getPercentWrong(self):
    return 100.0 * (self.con[0,1] + self.con[1,0]) / float(self.con.sum())
  
  def getFalsePosRateShadow(self):
    div = self.shadow.sum()
    if div!=0: return self.shadow[1] / float(div)
    else: return 0.0
