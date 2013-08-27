# Copyright 2013 Tom SF Haines

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



class RecordAverage(VideoNode):
  """Outputs the average colour to a file, for each frame - has a high degree of configurability."""
  def __init__(self, fn, line = '%(frame)i, %(r)f, %(g)f, %(b)f\n', head = 'frame, r, g, b\n', mult = 1.0, start_frame=0):
    self.f = open(fn, 'w')
    self.f.write(head)
    
    self.line = line
    self.mult = mult
    
    self.video = None
    self.channel = 0
    
    self.frame = start_frame
  
  def __del__(self):
    self.f.close()
  

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
    return MODE_RGB

  def inputName(self, channel=0):
    return 'Video stream for the average to be recorded for'

  def source(self, toChannel, video, videoChannel=0):
    self.video = video
    self.channel = videoChannel
    
    
  def dependencies(self):
    return [self.video]

  def nextFrame(self):
    # Get the frame...
    frame = self.video.fetch(self.channel)
    if frame==None: return False
    
    # Calculate the average...
    data = dict()
    data['frame'] = self.frame
    data['r'] = self.mult * numpy.mean(frame[:,:,0])
    data['g'] = self.mult * numpy.mean(frame[:,:,1])
    data['b'] = self.mult * numpy.mean(frame[:,:,2])
    
    # Write out...
    self.f.write(self.line % data)

    # Incriment frame and return...
    self.frame += 1
    return True


  def outputCount(self):
    return 0
