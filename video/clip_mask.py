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



class ClipMask(VideoNode):
  """Simple class that zeros out all areas of a mask outside a given box, in terms of displacements from the edges. Good for removing an area from a video stream that we do not want analysed, such as the sky, or a body of water."""
  def __init__(self, top = 0, bottom = 0, left = 0, right = 0):
    """The coordinates are distances from the various edges, with 0 meaning to include all pixels."""
    self.top = top
    self.bottom = bottom
    self.left = left
    self.right = right

    self.video = None
    self.channel = 0

    self.output = None

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
    return 'Input mask, to be spatially clipped'

  def source(self, toChannel, video, videoChannel=0):
    self.video = video
    self.channel = videoChannel


  def dependencies(self):
    return [self.video]

  def nextFrame(self):
    # Get the frame...
    self.output = self.video.fetch(self.channel)
    if self.output==None: return False
    self.output = self.output.copy()

    # Clip it...
    if self.left!=0: self.output[:,:self.left] = 0
    if self.right!=0: self.output[:,-1:-self.right:-1] = 0
    if self.top!=0: self.output[:self.top,:] = 0
    if self.bottom!=0: self.output[-1:-self.bottom:-1,:] = 0

    return True


  def outputCount(self):
    return 1

  def outputMode(self, channel=0):
    return MODE_MASK

  def outputName(self, channel=0):
    return 'Clipped mask'

  def fetch(self, channel=0):
    return self.output
