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

from video_node import *



class RenderDiff(VideoNode):
  """Renders the absolute difference between two images, with a multiplicative constant to make small differences visible."""
  def __init__(self, mult=100.0):
    """mult is the multiplicative factor for the difference rendering, to make small differences visible."""
    self.video = [None,None]
    self.channel = [0,0]

    self.mult = mult

    self.result = None

  def width(self):
    return self.video[0].width()

  def height(self):
    return self.video[0].height()

  def fps(self):
    return self.video[0].fps()

  def frameCount(self):
    return self.video[0].frameCount()


  def inputCount(self):
    return 2

  def inputMode(self, channel=0):
    return MODE_RGB

  def inputName(self, channel=0):
    return 'One of the two inputs'

  def source(self, toChannel, video, videoChannel=0):
    assert(video.outputMode(videoChannel)==MODE_RGB)
    self.video[toChannel] = video
    self.channel[toChannel] = videoChannel


  def dependencies(self):
    return self.video

  def nextFrame(self):
    a = self.video[0].fetch(self.channel[0])
    b = self.video[1].fetch(self.channel[1])
    if a==None or b==None:
      self.result = None
      return False

    if self.result==None or self.result.shape!=a.shape:
      self.result = a.copy()

    for c in xrange(3):
      self.result[:,:,c] = self.mult * numpy.fabs(a[:,:,c] - b[:,:,c])

    return True


  def outputCount(self):
    return 1

  def outputMode(self, channel=0):
    return MODE_RGB

  def outputName(self, channel=0):
    return 'Absolute difference between the two inputs'

  def fetch(self, channel=0):
    return self.result
