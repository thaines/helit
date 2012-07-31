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



class WriteCV(VideoNode):
  """Simple video writting class - turns a video stream into a file on the hardrive. codec defaults to moving jpeg (MJPG), but another good choice is XVID, especially for larger files. Other options exist depending on system."""
  def __init__(self, fn, codec='MJPG'):
    self.fn = fn
    self.codec = codec
    self.writer = None

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
    if self.writer==None:
      four_cv = cv.CV_FOURCC(self.codec[0], self.codec[1], self.codec[2], self.codec[3])
      self.writer = cv.CreateVideoWriter(self.fn, four_cv, self.fps(), (self.width(),self.height()), 1)

    frame = self.video.fetch(self.channel)
    mode = self.video.outputMode(self.channel)

    frame = (frame*255.0).astype(numpy.uint8)
    if mode==MODE_RGB:
      out = array2cv(frame[:,:,::-1])
    elif mode==MODE_FLOAT:
      out = array2cv(frame)
    else:
      raise Exception('Unsuported mode for WriteCV')

    cv.WriteFrame(self.writer, out)
    return True


  def outputCount(self):
    return 0
