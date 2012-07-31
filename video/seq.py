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



class Seq(VideoNode):
  """Defines a video created by appending several videos - effectivly pretends they are one big video. This can theoretically result in details such as frame rate and size changing as the video procedes, though that would typically be avoided as most other nodes do not handle this scenario. Breaks some of the rules as the input videos can not be part of the manager due to the unusual calling strategy."""
  def __init__(self, seq):
    """For initialisation it is given a list of objects that impliment the VideoNode interface - it then pretends to be that list of videos in the order given. All videos should be fresh, i.e. nextFrame should never have been called on them, and must not be members of the manager."""
    self.seq = seq
    self.index = 0

  def width(self):
    return self.seq[self.index].width()

  def height(self):
    return self.seq[self.index].height()

  def fps(self):
    return self.seq[self.index].fps()

  def frameCount(self):
    return sum(map(lambda rv: rv.frameCount(),self.seq))


  def inputCount(self):
    return 0


  def dependencies(self):
    ret = []
    for vid in self.seq:
      ret += vid.dependencies()
    return ret

  def nextFrame(self):
    while True:
      ret = self.seq[self.index].nextFrame()
      if (ret==False) and (self.index+1<len(self.seq)):
        self.index += 1
      else:
        break
    return ret


  def outputCount(self):
    return self.seq[self.index].outputCount()

  def outputMode(self, channel=0):
    return self.seq[self.index].outputMode(channel)

  def outputName(self, channel=0):
    return self.seq[self.index].outputName(channel)

  def fetch(self, channel=0):
    return self.seq[self.index].fetch(channel)
