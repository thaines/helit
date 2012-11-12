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



import os.path

import numpy
import cv
from utils.cvarray import cv2array

from video_node import *



class ReadCV_IS(VideoNode):
  """Presents an image sequence as a video, using open cv's image loading routines to load in the image files as needed."""
  def __init__(self, fn, fps = 30.0):
    """You initialise with a filename and the frames per second, as that can obviously not be obtained from an image sequence. The filename should have a '#' in it indicating where the number should go - it will then work out the rest. Note that it will order the frames numerically, and then use them - if there are gaps it won't care. Will work regardless of if the numbers are padded with zeros or not. The # must appear in the filename part, not the directory part - all files are expected to be in a single directory."""

    if '#' in fn:
      # Split into parts...
      path, fn = os.path.split(fn)
      start, end = map(lambda s: s.replace('#',''),fn.split('#',1))

      # Get all files from the directory, filter it down to only those that match the form...
      files = os.listdir(path)
      def valid(fn):
        if fn[:len(start)]!=start: return False
        if fn[-len(end):]!=end: return False
        if not fn[len(start):-len(end)].isdigit(): return False
        return True
      files = filter(valid,files)

      # Get the relevant numbers, sort the files by them...
      files.sort(key=lambda fn: int(fn[len(start):-len(end)]))

      # Put the paths back...
      self.files = map(lambda f: os.path.join(path, f), files)
    else:
      self.files = [fn]

    self.index = 0

    # We have the file list - now determine the various properties...
    test = cv.LoadImage(self.files[0])

    self.__width = test.width
    self.__height = test.height
    self.__fps = fps
    self.__frameCount = len(self.files)


  def width(self):
    return self.__width

  def height(self):
    return self.__height

  def fps(self):
    return self.__fps

  def frameCount(self):
    return self.__frameCount


  def inputCount(self):
    return 0


  def dependencies(self):
    return []

  def nextFrame(self):
    if self.index<len(self.files):
      try:
        img = cv.LoadImage(self.files[self.index])
        #print img.nChannels, img.width, img.height, img.depth, img.origin
        self.frame = cv2array(img)[:,:,::-1].astype(numpy.float32)/255.0
      except:
        print 'Frame #%i with filename %s failed to load.'%(self.index,self.files[self.index])
      self.index += 1
    else:
      self.frame = None

    return self.frame!=None


  def outputCount(self):
    return 1

  def outputMode(self, channel=0):
    return MODE_RGB

  def outputName(self, channel=0):
    return 'image'

  def fetch(self, channel=0):
    return self.frame
