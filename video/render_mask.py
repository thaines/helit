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



class RenderMask(VideoNode):
  """This class converts a MODE_MASK into a MODE_RGB, with various effects. This includes combining an image and setting a background colour."""
  def __init__(self, fgColour = (1.0,1.0,1.0), bgColour = (0.0,0.0,0.0)):
    self.fgColour = fgColour
    self.bgColour = bgColour

    self.mask = None
    self.maskChannel = 0

    self.fg = None
    self.fgChannel = 0

    self.bg = None
    self.bgChannel = 0

    self.output = None

  def width(self):
    return self.mask.width()

  def height(self):
    return self.mask.height()

  def fps(self):
    return self.mask.fps()

  def frameCount(self):
    return self.mask.frameCount()


  def inputCount(self):
    return 3

  def inputMode(self, channel=0):
    if channel==0: return MODE_MASK
    else: return MODE_RGB

  def inputName(self, channel=0):
    if channel==0: return 'Input mask'
    elif channel==1: return 'Optional video stream to use for the foreground, instead of the colour.'
    elif channel==2: return 'Optional video stream to use for the background, instead of the colour.'

  def source(self, toChannel, video, videoChannel=0):
    if toChannel==0:
      self.mask = video
      self.maskChannel = videoChannel

    elif toChannel==1:
      self.fg = video
      self.fgChannel = videoChannel

    elif toChannel==2:
      self.bg = video
      self.bgChannel = videoChannel


  def dependencies(self):
    ret = [self.mask]
    if self.fg!=None: ret.append(self.fg)
    if self.bg!=None: ret.append(self.bg)
    return ret

  def nextFrame(self):
    mask = self.mask.fetch(self.maskChannel)
    if mask==None:
      self.output = None
      return False

    if self.output==None:
      self.output = numpy.empty((mask.shape[0], mask.shape[1], 3), dtype=numpy.float32)

    mask = mask.astype(numpy.bool)

    if self.bg==None:
      for c in xrange(3):
        self.output[:,:,c] = self.bgColour[c]
    else:
      bg = self.bg.fetch(self.bgChannel)
      if bg==None: return False
      self.output[:,:,:] = bg

    if self.fg==None:
      for c in xrange(3):
        self.output[:,:,c][mask] = self.fgColour[c]
    else:
      fg = self.fg.fetch(self.fgChannel)
      if fg==None: return False
      for c in xrange(3):
        self.output[:,:,c][mask] = fg[:,:,c][mask]

    return True


  def outputCount(self):
    return 1

  def outputMode(self, channel=0):
    return MODE_RGB

  def outputName(self, channel=0):
    return 'Mask rendered as specified'

  def fetch(self, channel=0):
    return self.output