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



MODE_RGB = 0 # Each frame is indexed as [y,x,channel], where x is [0,width), y [0,height) and channel one of 0=red, 1=green, 2=blue. The colour channels are represented by float32s, in the range [0,1], unless there is a good reason to leave that range.
MODE_MASK = 1 # Each frame is indexed as [y,x], and goes to an 8 bit unsigned int, which contains 0 for not included and 1 for included.
MODE_FLOW = 2 # Optical flow, where each frame is indexed as [y,x,dir], where dir=0 indexes a y-axis offset and 1 indexes a x-axis offset. Uses float32's.
MODE_WORD = 3 # Indexes words, as in the video has been quantised somehow. Indexing is [y,x], and leads to int32's. What the words mean is defined by the situation, but an interface returning this mode is expected to provide the method wordCount() to indicate how many exist, and -1 is used to indicate 'no word'.
MODE_FLOAT = 4 # Each frame is indexed as [y,x], and goes to a float32, which can mean anything.
MODE_MATRIX = 5 # Returns a matrix that represents something - used for colour transformations for instance with regard to lighting changes.
MODE_OTHER = -1 # For anything not covered by another mode, or when multiple modes are supported.



mode_to_string = {MODE_RGB:'rgb', MODE_MASK:'mask', MODE_FLOW:'flow', MODE_WORD:'word',MODE_FLOAT:'float',MODE_MATRIX:'matrix',MODE_OTHER:'other'}



class VideoNode:
  """Interface for a video procesing object that provides the next frame on demand, as a numpy array that is always indexed from the top right of the frame."""
  def width(self):
    """Returns the width of the video."""
    raise Exception('width not implimented')

  def height(self):
    """Returns the height of the video."""
    raise Exception('height not implimented')

  def fps(self):
    """Returns the frames per second of the video, as a floating point value."""
    raise Exception('fps not implimented')

  def frameCount(self):
    """Returns the number of times you can call nextFrame before it starts returning None."""
    raise Exception('frameCount not implimented')


  def inputCount(self):
    """Returns the number of inputs."""
    raise Exception('inputCount not implimented')

  def inputMode(self, channel=0):
    """Returns the required mode for the given input."""
    raise Exception('inputMode not implimented')

  def inputName(self, channel=0):
    """Returns a human readable description of the given input."""
    raise Exception('inputName not implimented.')

  def source(self, toChannel, video, videoChannel=0):
    """Sets a video as input to the video object, in channel to Channel, optionally including which channel to extract from video as videoChannel."""
    raise Exception(' not implimented')


  def dependencies(self):
    """Returns a list of video objects that this video object is dependent on - the nextframe method must be called on all of these prior to it being called on this, otherwise strange stuff will happen. The list is allowed to include duplicates."""
    raise Exception('dependencies not implimented')

  def nextFrame(self):
    """Moves to the next frame, returning True if there is now a set of next frames that can be extracted using the fetch command, and False if not. typically False means we are out of data, as an error would lead to an exception being thrown. Must not be called until the object is setup - i.e. all inputs have been set, and any other object-specific actions."""
    raise Exception('nextFrame not implimented')


  def outputCount(self):
    """Returns the number of outputs - a video object is allowed to have multiple outputs. The output in position 0 is the default and often the only one."""
    raise Exception('outputCount not implimented')

  def outputMode(self, channel=0):
    """Returns one of the modes, which indicates the format of the entity returned by nextFrame. The optional out parameter indicates which output to indicate for. Most of these do not both with outputs."""
    raise Exception('mode not implimented')

  def outputName(self, channel=0):
    """Returns a string indicating what the output in question is - arbitrary and for human consumption only."""
    raise Exception('name not implimented')

  def fetch(self, channel=0):
    """Returns the requested channel, as a numpy array. You can not assume that the object is persistant, i.e. it might be the same object returned each time, but with a different contents. The optional channel parameter indicates which output to get. fetch can be called multiple times for a channel between each call to nextFrame."""
    raise Exception('fetch not implimented')
