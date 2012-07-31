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



from collections import defaultdict
import numpy

from video_node import *



class CombineGrid(VideoNode):
  """Given multiple MODE_RGB streams as input this combines them into a single output, arranged as a grid. Resizes appropriatly and handles gaps."""
  def __init__(self, horizontal, vertical = 1):
    self.video = map(lambda _: map(lambda _: None, xrange(horizontal)), xrange(vertical))
    self.channel = map(lambda _: map(lambda _: 0, xrange(horizontal)), xrange(vertical))

    self.output = None

  def width(self):
    widths = defaultdict(int)
    for py in xrange(len(self.video)):
      for px in xrange(len(self.video[py])):
        if self.video[py][px]!=None:
          widths[px] = max((widths[px],self.video[py][px].width()))
    return sum(widths.values())


  def height(self):
    heights = defaultdict(int)
    for py in xrange(len(self.video)):
      for px in xrange(len(self.video[py])):
        if self.video[py][px]!=None:
          heights[py] = max((heights[py],self.video[py][px].height()))
    return sum(heights.values())

  def fps(self):
    for py in xrange(len(self.video)):
      for px in xrange(len(self.video[py])):
        if self.video[py][px]!=None:
          return self.video[py][px].fps()

  def frameCount(self):
    ret = None
    for py in xrange(len(self.video)):
      for px in xrange(len(self.video[py])):
        if self.video[py][px]!=None:
          val = self.video[py][px].frameCount()
          if ret==None or val<ret:
            ret = val
    return ret


  def inputCount(self):
    return len(self.video) * len(self.video[0])

  def inputMode(self, channel=0):
    return MODE_RGB

  def inputName(self, channel=0):
    return 'A video stream for a particular grid cell'

  def source(self, toChannel, video, videoChannel=0):
    py = toChannel // len(self.video[0])
    px = toChannel - py*len(self.video[0])

    self.video[py][px] = video
    self.channel[py][px] = videoChannel


  def dependencies(self):
    ret = []
    for row in self.video: ret += row
    return filter(lambda a: a!=None,ret)

  def nextFrame(self):
    # Work out size info...
    widths = defaultdict(int)
    heights = defaultdict(int)
    for py in xrange(len(self.video)):
      for px in xrange(len(self.video[py])):
        if self.video[py][px]!=None:
          widths[px] = max((widths[px],self.video[py][px].width()))
          heights[py] = max((heights[py],self.video[py][px].height()))

    # Create the output image if needed...
    totalHeight = sum(heights.values())
    totalWidth = sum(widths.values())

    if self.output==None or self.output.shape[0]!=totalHeight or self.output.shape[1]!=totalWidth:
      self.output = numpy.empty((totalHeight,totalWidth,3),dtype=numpy.float32)

    # Zero buffer out...
    self.output[:,:,:] = 0.0

    # Make size dictionaries culumative...
    for py in xrange(len(self.video)): heights[py] += heights[py-1]
    for px in xrange(len(self.video[0])): widths[px] += widths[px-1]

    # Copy each video in turn...
    ret = True
    for py in xrange(len(self.video)):
      for px in xrange(len(self.video[py])):
        if self.video[py][px]!=None:
          # Fetch the frame, handle non-existance...
          frame = self.video[py][px].fetch(self.channel[py][px])
          if frame==None:
            ret = False
            break

          # Copy it to the right place...
          baseY = heights[py-1]
          baseX = widths[px-1]
          if len(frame.shape)==2: # Support for greyscale.
            frame = numpy.dstack((frame,frame,frame))
          self.output[baseY:baseY+frame.shape[0],baseX:baseX+frame.shape[1],:] = frame

    return ret


  def outputCount(self):
    return 1

  def outputMode(self, channel=0):
    return MODE_RGB

  def outputName(self, channel=0):
    return 'All input streams combined'

  def fetch(self, channel=0):
    return self.output
