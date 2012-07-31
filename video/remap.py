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



class Remap(VideoNode):
  """This remaps channels, potentially combining multiple video sources and consequentially generating an arbitrary VideoNode object that can involve any data the user decides. All the standard rules of VideoNode are sustained, as long as all the input videos share resolution and fps. Primarily used when saving out node results to disk, to save the precise set of feeds you want."""
  def __init__(self, channels = []):
    """This initialises the object - channels is a list of pairs, containing first the video object and then the video channel to use, to indicate what the channels of this object are. Channels can have duplicates. The channels can be editted by the input interface after initialisation, and if you give a toChannel number outside the current range it is appended, so this data can be built/editted latter if needed. (There is no way to delete a channel however.)"""
    self.sources = []
    self.channels = []

    for pair in channels:
      if pair[0] not in self.sources: self.sources.append(pair[0])
      index = self.source.index(pair[0])

      self.channels.append((index,pair[1]))

  def width(self):
    return self.sources[0].width()

  def height(self):
    return self.sources[0].height()

  def fps(self):
    return self.sources[0].fps()

  def frameCount(self):
    return self.sources[0].frameCount()


  def inputCount(self):
    return len(self.channels)

  def inputMode(self, channel=0):
    return MODE_OTHER

  def inputName(self, channel=0):
    return 'Anything you want'

  def source(self, toChannel, video, videoChannel=0):
    if video not in self.sources:
      self.sources.append(video)
    index = self.source.index(video)

    if toChannel>=len(self.channels):
      self.channels.append((index,videoChannel))
    else:
      self.channels[toChannel] = (index,videoChannel)


  def dependencies(self):
    return self.sources

  def nextFrame(self):
    return True # No-op as it only remaps method calls.


  def outputCount(self):
    return len(self.channels)

  def outputMode(self, channel=0):
    pair = self.channels[channel]
    return self.sources[pair[0]].mode(pair[1])

  def outputName(self, channel=0):
    pair = self.channels[channel]
    return self.sources[pair[0]].name(pair[1])

  def fetch(self, channel=0):
    pair = self.channels[channel]
    return self.sources[pair[0]].nextFrame(pair[1])
