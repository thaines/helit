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



class MaskFlow(VideoNode):
  """Takes as input an optical flow field and a mask - zeros out all optical flow vectors that are outside the mask. Primarilly to allow an optical flow algorithm and a background subtractions algorithms results to be combined to get a 'better', or at least cleaner, result."""
  def __init__(self):
    self.flow = None
    self.flowChannel = 0

    self.mask = None
    self.maskChannel = 0

    self.output = None

  def width(self):
    return self.flow.width()

  def height(self):
    return self.flow.height()

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
    if channel==0: return 'Optical flow field'
    else: return 'Mask indicating where objects we care about are'

  def source(self, toChannel, video, videoChannel=0):
    if toChannel==0:
      self.flow = video
      self.flowChannel = videoChannel
    else:
      self.mask = video
      self.maskChannel = videoChannel


  def dependencies(self):
    return [self.flow, self.mask]

  def nextFrame(self):
    # Fetch the data...
    flow = self.flow.fetch(self.flowChannel)
    mask = self.mask.fetch(self.maskChannel)
    if flow==None or mask==None:
      self.output = None
      return False

    # If needed create the output...
    if self.output==None: self.output = flow.copy()

    # Calculate the output...
    self.output[:,:,:] = flow[:,:,:]
    self.output[:,:,0] *= mask
    self.output[:,:,1] *= mask

    return True


  def outputCount(self):
    return 1

  def outputMode(self, channel=0):
    return MODE_FLOW

  def outputName(self, channel=0):
    return 'Masked-out optical flow field'

  def fetch(self, channel=0):
    return self.output
