# Copyright (c) 2014, Tom SF Haines
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#  * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



class Viewport:
  """Defines a view onto a renderable region. You have the width and height of the window into which to render - these are always integers; and the start and end coordinates of a viewport for you to render into that window. This exists to be provided to a draw method with a cairo context. Tradition dictates that the aspect ratio of the window and viewport be identical - a renderer is allowed to operate under that assumption."""
  def __init__(self, width, height, start_x, start_y, end_x, end_y):
    self.width = width
    self.height = height
    
    self.start_x = start_x
    self.start_y = start_y
    self.end_x = end_x
    self.end_y = end_y
  
  def __str__(self):
    return 'Viewport x(%.3f-%.3f) y(%.3f-%.3f)' % (self.start_x,self.end_x,self.start_y,self.end_y)
  
  
  def view_to_original(self, x, y):
    x = (x / float(self.width))  * (self.end_x - self.start_x) + self.start_x
    y = (y / float(self.height)) * (self.end_y - self.start_y) + self.start_y
    return (x, y)
  
  def original_to_view(self, x, y):
    x = self.width  * (x - self.start_x) / (self.end_x - self.start_x)
    y = self.height * (y - self.start_y) / (self.end_y - self.start_y)
    return (x, y)
  
  def original_to_view_scale(self, scale):
    return scale * float(self.width) / (self.end_x - self.start_x)



class Layer:
  """Defines an interface that layers have to impliment - basically this allows them to render to the drawing area, and influence things such as its default size."""
  def get_size(self):
    """Returns the tuple (height, width), the default size of the layer (Origin is assumed to be (0,0)). This basically defines the area in which a Viewport can be expected to show something interesting."""
    raise NotImplementedError
  
  def draw(self, ctx, vp):
    """Draws to the provided cairo context (ctx) the contents defined by the provided viewport (vp). Note that it could be in a stack with other layers - it probably shouldn't be clearing the background."""
    raise NotImplementedError
