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



import math
import numpy

from video_node import *



class Half(VideoNode):
  """Given a colour video stream halfs its resolution in each dimension - as simple a node as you can get really. Requires the input have even dimensions!"""
  def __init__(self):
    self.video = None
    self.channel = 0

    self.output = None

  def width(self):
    return self.video.width()//2

  def height(self):
    return self.video.height()//2

  def fps(self):
    return self.video.fps()

  def frameCount(self):
    return self.video.frameCount()


  def inputCount(self):
    return 1

  def inputMode(self, channel=0):
    return MODE_RGB

  def inputName(self, channel=0):
    return 'Video stream to be downsized'

  def source(self, toChannel, video, videoChannel=0):
    self.video = video
    self.channel = videoChannel


  def dependencies(self):
    return [self.video]

  def nextFrame(self):
    # Fetch the frame...
    img = self.video.fetch(self.channel)
    if img==None:
      self.output = None
      return False

    if len(img.shape)==2:
      img = img.reshape((img.shape[0],img.shape[1],1))

    # If needed create an output
    if self.output==None:
      self.output = numpy.empty((img.shape[0]//2,img.shape[1]//2,3), dtype=numpy.float32)

    self.output[:,:,:] = 0.0
    self.output[:,:,:] += img[0::2,0::2,:]
    self.output[:,:,:] += img[0::2,1::2,:]
    self.output[:,:,:] += img[1::2,0::2,:]
    self.output[:,:,:] += img[1::2,1::2,:]
    self.output[:,:,:] *= 0.25

    return True


  def outputCount(self):
    return 1

  def outputMode(self, channel=0):
    return MODE_RGB

  def outputName(self, channel=0):
    return 'Smaller video stream'

  def fetch(self, channel=0):
    return self.output
