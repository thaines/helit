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



class RenderWord(VideoNode):
  """Renders a grid of words, using a provided colour scheme. No word is automatically set to black."""
  def __init__(self, colours = None, gridSize = 8):
    """Colours is a list of colours as 3-tuples, matching up with the words. Can be left as None to have it call the suggestedColours() method of the provided word video node. gridSize is simply the multiplicative factor to get back to a full sized image."""
    self.colours = colours
    self.gridSize = gridSize

    self.word = None
    self.wordChannel = 0

    self.output = None

  def width(self):
    return self.word.width() * self.gridSize

  def height(self):
    return self.word.height() * self.gridSize

  def fps(self):
    return self.word.fps()

  def frameCount(self):
    return self.word.frameCount()


  def inputCount(self):
    return 1

  def inputMode(self, channel=0):
    return MODE_WORD

  def inputName(self, channel=0):
    return 'Word stream video.'

  def source(self, toChannel, video, videoChannel=0):
    self.word = video
    self.wordChannel = videoChannel


  def dependencies(self):
    return [self.word]

  def nextFrame(self):
    # Fetch the word data...
    word = self.word.fetch(self.wordChannel)
    if word==None:
      self.output = None
      return False

    # Create the output buffer if needed...
    if self.output==None:
      self.output = numpy.empty((self.height(), self.width(), 3), dtype=numpy.float32)
      self.inter = numpy.empty((self.word.height(), self.word.width(), 3), dtype=numpy.float32)

    # If we have no colours fetch them...
    if self.colours==None:
      self.colours = self.word.suggestedColours()
      if self.colours==None: return False

    # Render the words to a small image...
    self.inter[:,:,:] = 0.0

    for w in xrange(len(self.colours)):
      mask = word==w
      for c in xrange(3):
        self.inter[:,:,c][mask] = self.colours[w][c]

    # Scale up...
    self.output[:,:,:] = numpy.repeat(numpy.repeat(self.inter, self.gridSize, axis=0), self.gridSize, axis=1)

    return True


  def outputCount(self):
    return 1

  def outputMode(self, channel=0):
    return MODE_RGB

  def outputName(self, channel=0):
    return 'Visualisation of words'

  def fetch(self, channel=0):
    return self.output
