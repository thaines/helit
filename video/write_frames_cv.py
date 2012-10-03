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



import os
import os.path

import numpy
import cv
from utils.cvarray import *

from video_node import *



class WriteFramesCV(VideoNode):
  """Saves a video file to disk as a sequence of image files."""
  def __init__(self, out, start_frame = 0, digits = 4):
    """out is the output filename, which must contain a single '#' to indicate where the number should go. start_frame is the number assigned to the first frame - it defaults to 0, but you might want to consider 1. digits is how many digits to use as a minimum in the filename - it defaults to 4, so the first frame will be '0000', the next '0001', and so on."""
    self.out = out
    self.frame = start_frame
    self.digits = digits

    self.video = None
    self.channel = 0
    
    # Make sure the directory exists...
    try:
      os.makedirs(os.path.split(self.out)[0])
    except: pass

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
    return MODE_OTHER

  def inputName(self, channel=0):
    return 'Video stream to be saved to disk - supports MODE_RGB and MODE_FLOAT'

  def source(self, toChannel, video, videoChannel=0):
    self.video = video
    self.channel = videoChannel


  def dependencies(self):
    return [self.video]

  def nextFrame(self):
    # Get the frame...
    frame = self.video.fetch(self.channel)
    if frame==None: return False
    mode = self.video.outputMode(self.channel)

    # Convert to something opencv can use...
    frame = (frame*255.0).astype(numpy.uint8)
    if mode==MODE_RGB:
      out = array2cv(frame[:,:,::-1])
    elif mode==MODE_FLOAT:
      out = array2cv(frame)
    else:
      raise Exception('Unsuported mode for WriteFramesCV')
    
    # Calculate the file name...
    fn = self.out.replace('#', '%0*i'%(self.digits,self.frame), 1)
    
    # Save...
    cv.SaveImage(fn, out)

    # Incriment frame and return...
    self.frame += 1
    return True


  def outputCount(self):
    return 0
