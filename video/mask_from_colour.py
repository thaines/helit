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



class MaskFromColour(VideoNode):
  """This converts a colour video stream into a pair of masks - basically you provide a list of exact colours, indicating which are background, which are foreground. This provides the main mask, but then all areas where there is colour that is not one of the known colours are assumed to be for ignoring, so a second mask is created that is only True where an exact match has been acheived. By 'exact' it operates under the assumption of 255 levels per channel."""
  def __init__(self, fgColours, bgColours):
    """You provide two lists of 3-tuples, each being (r,g,b) as floating point colours, [0.0,1.0], that define where the background and foreground are."""
    self.fg = tuple(map(lambda c: numpy.asarray(c, dtype=numpy.float32), fgColours))
    self.bg = tuple(map(lambda c: numpy.asarray(c, dtype=numpy.float32), bgColours))

    self.vid = None
    self.vidChannel = 0

    self.mask = None
    self.maskValid = None

    self.epsilon = 0.5/255.0


  def width(self):
    return self.vid.width()

  def height(self):
    return self.vid.height()

  def fps(self):
    return self.vid.fps()

  def frameCount(self):
    return self.vid.frameCount()


  def inputCount(self):
    return 1

  def inputMode(self, channel=0):
    return MODE_RGB

  def inputName(self, channel=0):
    return 'Image to be turned into a mask.'

  def source(self, toChannel, video, videoChannel=0):
    self.vid = video
    self.vidChannel = videoChannel


  def dependencies(self):
    return [self.vid]

  def nextFrame(self):
    # Fetch the data...
    img = self.vid.fetch(self.vidChannel)
    if img==None:
      self.mask = None
      self.maskValid = None
      return False

    # If not already created generate the stores for the mask and mask validity mask...
    if self.mask==None: self.mask = numpy.empty((img.shape[0], img.shape[1]), dtype=numpy.uint8)
    if self.maskValid==None: self.maskValid = numpy.empty((img.shape[0], img.shape[1]), dtype=numpy.uint8)

    # Reset the masks...
    self.mask[:,:] = 0
    self.maskValid[:,:] = 0

    # Go through and handle the foreground colours...
    for col in self.fg:
      self.mask[(numpy.abs(img-col.reshape((1,1,-1)))<self.epsilon).all(axis=2)] = 1
    self.maskValid[self.mask==1] = 1

    # Go through and handle the background colours...
    for col in self.bg:
      self.maskValid[(numpy.abs(img-col.reshape((1,1,-1)))<self.epsilon).all(axis=2)] = 1

    return True


  def outputCount(self):
    return 2

  def outputMode(self, channel=0):
    return MODE_MASK

  def outputName(self, channel=0):
    if channel==0: return 'The mask generated from the colour scheme'
    else: return 'Indicates where the mask is valid, i.e. where a known colour is.'

  def fetch(self, channel=0):
    if channel==0: return self.mask
    else: return self.maskValid
