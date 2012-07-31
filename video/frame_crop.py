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



from video_node import *



class FrameCrop(VideoNode):
  """Provides a simple wrapper to shorten a video - on the first call it skips frames till it gets to the indicated starting frame, then it stops after the given number of frames has been reached. The video being shortened must not be part of the manager."""
  def __init__(self, video, start, length = None):
    """You provide the video object to consume and the start frame, plus an optional length - if the length is omitted it automatically goes to the end of the input video."""
    if length==None: length = video.frameCount() - start

    self.video = video
    self.start = start
    self.length = length
    self.remain = length

  def width(self):
    return self.video.width()

  def height(self):
    return self.video.height()

  def fps(self):
    return self.video.fps()

  def frameCount(self):
    return self.length


  def inputCount(self):
    return 0


  def dependencies(self):
    return self.video.dependencies()

  def nextFrame(self):
    if self.remain<0: return False

    while self.start!=0:
      self.video.nextFrame()
      self.start -= 1

    ret = self.video.nextFrame()
    self.remain -= 1
    if self.remain<0: return False
    return ret


  def outputCount(self):
    return self.video.outputCount()

  def outputMode(self, channel=0):
    return self.video.outputMode(channel)

  def outputName(self, channel=0):
    return self.video.outputName(channel)

  def fetch(self, channel=0):
    return self.video.fetch(channel)
