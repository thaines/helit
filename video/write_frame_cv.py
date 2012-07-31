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
import cv
from utils.cvarray import *

from video_node import *



class WriteFrameCV(VideoNode):
  """You provide this node with a list of pairs of (zero indexed) frame numbers and file names - it then saves those particular frames to disk using the open cv image writting functions."""
  def __init__(self, frameList):
    """frameList is a list of 2-tuples, where the first entry is a zero based frame number and the second entry a filename that open cv can use to save that particular frame to disk."""
    self.frameList = frameList
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
    return MODE_OTHER

  def inputName(self, channel=0):
    return 'Video stream to be saved to disk - supports MODE_RGB and MODE_FLOAT'

  def source(self, toChannel, video, videoChannel=0):
    self.video = video
    self.channel = videoChannel


  def dependencies(self):
    return [self.video]

  def nextFrame(self):
    # Check if we care about this frame...
    if self.frame in map(lambda p: p[0], self.frameList):
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
        raise Exception('Unsuported mode for WriteFrameCV')

      # Save...
      for f,fn in self.frameList:
        if f==self.frame:
          cv.SaveImage(fn,out)

    # Incriment frame and return...
    self.frame += 1
    return True


  def outputCount(self):
    return 0
