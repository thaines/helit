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



class RenderFlow(VideoNode):
  """Renders a MODE_FLOW into a MODE_RGB, using the standard conversion with HSV space, where H becomes direction and S speed, such that it is white when there is no motion and gets more colour as speed increases, whilst the actual colour represents the direction of motion. There are two speed representations - linear and asymptotic."""
  def __init__(self, scale, asymptotic = False):
    """Scale is, if not asymptotic, the maximum speed represented. If it is asymptotic it is the speed that obtains 50% staturation."""
    self.video = None
    self.channel = 0

    self.output = None

    self.scale = scale
    self.asymptotic = asymptotic

    # Constants for hsv to rgb conversion...
    self.pos   = numpy.array([-180.0,-120.0, -60.0,   0.0,  60.0, 120.0, 180.0])
    self.red   = numpy.array([   0.0,   0.0,   1.0,   1.0,   1.0,   0.0,   0.0])
    self.green = numpy.array([   1.0,   0.0,   0.0,   0.0,   1.0,   1.0,   1.0])
    self.blue  = numpy.array([   1.0,   1.0,   1.0,   0.0,   0.0,   0.0,   1.0])
    self.pos *= math.pi/180.0

  def width(self):
    return self.video.width()

  def height(self):
    return self.video.height()

  def fps(self):
    return self.video.fps()

  def frameCount(self):
    return self.video.frameCount()


  def inputCount(self):
    """Returns the number of inputs."""
    raise Exception('inputCount not implimented')

  def inputMode(self, channel=0):
    return MODE_FLOW

  def inputName(self, channel=0):
    return 'flow field to be rendered'

  def source(self, toChannel, video, videoChannel=0):
    self.video = video
    self.channel = videoChannel


  def dependencies(self):
    return [self.video]

  def nextFrame(self):
    # Fetch the frame...
    uv = self.video.fetch(self.channel)

    # If needed create an output
    if self.output==None:
      self.output = numpy.empty((uv.shape[0],uv.shape[1],3), dtype=numpy.float32)
      self.inter = numpy.zeros((uv.shape[0],uv.shape[1],2), dtype=numpy.float32)

    # Fill in the intermediate array - first entry the angle, second entry the speed...
    numpy.arctan2(uv[:,:,0],uv[:,:,1],self.inter[:,:,0])
    self.inter[:,:,1] = numpy.sqrt(numpy.square(uv[:,:,0]) + numpy.square(uv[:,:,1]))

    # Depending on the mode convert the speed into a saturation value...
    if self.asymptotic:
      self.inter[:,:,1] += self.scale
      self.inter[:,:,1] = numpy.divide(1.0, self.inter[:,:,1])
      self.inter[:,:,1] *= -1.0
      self.inter[:,:,1] += 1.0
    else:
      self.inter[:,:,1] /= self.scale
      numpy.clip(self.inter[:,:,1],0.0,1.0,self.inter[:,:,1])

    # Convert our h and s values to rgb...
    self.output[:,:,0] = numpy.interp(self.inter[:,:,0], self.pos, self.red)
    self.output[:,:,1] = numpy.interp(self.inter[:,:,0], self.pos, self.green)
    self.output[:,:,2] = numpy.interp(self.inter[:,:,0], self.pos, self.blue)

    self.output[:,:,0] *= self.inter[:,:,1]
    self.output[:,:,1] *= self.inter[:,:,1]
    self.output[:,:,2] *= self.inter[:,:,1]

    self.output[:,:,:] += 1.0
    self.output[:,:,0] -= self.inter[:,:,1]
    self.output[:,:,1] -= self.inter[:,:,1]
    self.output[:,:,2] -= self.inter[:,:,1]

    return True


  def outputCount(self):
    return 1

  def outputMode(self, channel=0):
    return MODE_RGB

  def outputName(self, channel=0):
    return 'Visualisation of optical flow field'

  def fetch(self, channel=0):
    return self.output
