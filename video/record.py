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



class Record(VideoNode):
  """Records the state of a single node, so it can be replayed at a later date. This consists of saving the data to a file, without loss. Records all channels of a node - you can use Remap to get something strange. Uses bzip compresion and python serialisation - hardly sophisticated. Partners with Play, which reads the file back in and spits it out, such that it is as though it is identical to the node given to the constructor."""
  def __init__(self, node, fn):
    self.node = node

    # Open the file, save the header, marry the princess...
    self.f = bz2.BZ2File(fn, 'w')

    head = 'rvnf-001' # Recorded video node file, for want of a better term - for safety.
    self.f.write(head)

    oData = []
    for c in xrange(self.node.outputCount()):
      oData.append((self.node.outputMode(c), self.node.outputName(c)))

    header = (self.node.width(), self.node.height(), self.node.fps(), self.node.frameCount(), oData)
    pickle.dump(header, self.f, -1)

  def __del__(self):
    self.f.close() # Unnecesary, but means there is a clear way for a user to force the issue.

  def width(self):
    return self.node.width()

  def height(self):
    return self.node.height()

  def fps(self):
    return self.node.fps()

  def frameCount(self):
    return self.node.frameCount()


  def inputCount(self):
    return 0


  def dependencies(self):
    return [self.node]

  def nextFrame(self):
    frame = []
    for c in xrange(self.node.outputCount()):
      frame.append(self.node.fetch(c))
    pickle.dump(frame,self.f,-1)

    return True


  def outputCount(self):
    return 0
