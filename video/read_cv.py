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
import cv
from utils.cvarray import cv2array

from video_node import *



class ReadCV(VideoNode):
  """Simple wrapper around open cv's video reading interface - limited compatability but easy for reading lots of video files."""
  def __init__(self, fn):
    """Given a filename opens it and streams video from it."""
    if not os.path.exists(fn):
      raise Exception('Filename does not exist on filesystem (filename=%s).'%fn)
    self.vid = cv.CaptureFromFile(fn)
    self.frame = None

  def width(self):
    return int(cv.GetCaptureProperty(self.vid,cv.CV_CAP_PROP_FRAME_WIDTH))

  def height(self):
    return int(cv.GetCaptureProperty(self.vid,cv.CV_CAP_PROP_FRAME_HEIGHT))

  def fps(self):
    return cv.GetCaptureProperty(self.vid,cv.CV_CAP_PROP_FPS)

  def frameCount(self):
    return int(cv.GetCaptureProperty(self.vid,cv.CV_CAP_PROP_FRAME_COUNT))


  def inputCount(self):
    return 0


  def dependencies(self):
    return []

  def nextFrame(self):
    self.frame = cv.QueryFrame(self.vid)

    if self.frame==None: return False
    self.frameNP = cv2array(self.frame)[:,:,::-1].astype(numpy.float32)/255.0
    return True


  def outputCount(self):
    return 1

  def outputMode(self, channel=0):
    return MODE_RGB

  def outputName(self, channel=0):
    return 'image'

  def fetch(self, channel=0):
    return self.frameNP
