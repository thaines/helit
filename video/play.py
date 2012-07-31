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



import cPickle as pickle
import bz2

from video_node import *



class Play(VideoNode):
  """Plays back a file that has been saved by the Record object. Has an identical output interface to the node that was fed into Record, meaning it can be used identically."""
  def __init__(self, fn):
    # Open the file, verify its correct...
    self.f = bz2.BZ2File(fn, 'r')
    head = self.f.read(8)
    assert(head=='rvnf-001')

    # Read in the header information...
    self.header = pickle.load(self.f)

  def width(self):
    return self.header[0]

  def height(self):
    return self.header[1]

  def fps(self):
    return self.header[2]

  def frameCount(self):
    return self.header[3]


  def inputCount(self):
    return 0


  def dependencies(self):
    return []

  def nextFrame(self):
    try:
      self.frame = pickle.load(self.f)
    except:
      self.frame = None
      return False
    return True


  def outputCount(self):
    return len(self.header[4])

  def outputMode(self, channel=0):
    return self.header[4][channel][0]

  def outputName(self, channel=0):
    return self.header[4][channel][1]

  def fetch(self, channel=0):
    if self.frame==None: return None
    return self.frame[channel]
