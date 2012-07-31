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



class Mask_SABS(VideoNode):
  """Designed to generate the correct masking and validity information given the ground truth data of the 'Stuttgart Artificial Background Subtraction' dataset. Basically binarises the input, with black being background and every other colour being foreground, before using an erode on both channels, such that the pixels that change are marked as invalid for scoring. Outputs two masks - one indicating foreground/background, another indicating where it should be scored."""
  def __init__(self):
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

    # Generate the mask - basically black is False, everything else True...
    self.mask[:,:] = 1
    self.mask[(img<self.epsilon).all(axis=2)] = 0

    # Apply the erosion filter, to generate the validity mask (Done in a non-standard way, as the method is such that this can be done in a simpler way.)...
    count = self.mask.copy() # Will put 0 where False, 1 where True, a fact also used below.
    ind0dec = numpy.append([0],numpy.arange(self.mask.shape[0]-1))
    ind1dec = numpy.append([0],numpy.arange(self.mask.shape[1]-1))
    ind0inc = numpy.append(numpy.arange(1,self.mask.shape[0]),[-1])
    ind1inc = numpy.append(numpy.arange(1,self.mask.shape[1]),[-1])

    count += self.mask[ind0dec,:]
    count += self.mask[ind0inc,:]
    count += self.mask[:,ind1dec]
    count += self.mask[:,ind1inc]

    count += self.mask[ind0dec,:][:,ind1dec]
    count += self.mask[ind0dec,:][:,ind1inc]
    count += self.mask[ind0inc,:][:,ind1dec]
    count += self.mask[ind0inc,:][:,ind1inc]

    self.maskValid[:,:] = 0
    self.maskValid[count==0] = 1
    self.maskValid[count==9] = 1

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
