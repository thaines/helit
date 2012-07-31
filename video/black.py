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



class Black(VideoNode):
  """Dummy node that generates a black video feed of a given size, length and framerate."""
  def __init__(self, width, height, frame_count, fps = 30.0):
    """Constructs the object with the provided properties."""
    self.__frameCount = frame_count
    self.__fps = fps

    self.frame = -1
    self.blackness = numpy.zeros((height,width,3), dtype=numpy.float32)


  def width(self):
    return self.blackness.shape[1]

  def height(self):
    return self.blackness.shape[0]

  def fps(self):
    return self.__fps

  def frameCount(self):
    return self.__frameCount


  def inputCount(self):
    return 0


  def dependencies(self):
    return []

  def nextFrame(self):
    self.frame += 1
    return self.frame < self.__frameCount


  def outputCount(self):
    return 1

  def outputMode(self, channel=0):
    return MODE_RGB

  def outputName(self, channel=0):
    return 'image'

  def fetch(self, channel=0):
    return self.blackness
