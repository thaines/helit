# Copyright (c) 2014, Tom SF Haines
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#  * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



import math

from viewport_layer import *



class ReticleOverlay(Layer):
  """Draws a reticle on the drawing area - good for zooming."""
  def __init__(self):
    self.render = True
    self.size = 16
  
  
  def set_render(self, render):
    """True to draw the reticle, False to not draw it."""
    self.render = render

  
  def get_size(self):
    """To satisfy the interface for a Layer object"""
    return (self.size*2, self.size*2)


  def draw(self, ctx, vp):
    """Draws a simple reticle."""
    
    if self.render:
      cx = vp.width * 0.5
      cy = vp.height * 0.5
      
      ctx.set_line_width(1.0)
      ctx.set_source_rgba(1.0, 0.0, 0.0, 0.2)
      
      ctx.move_to(cx-self.size, cy-self.size)
      ctx.line_to(cx+self.size, cy-self.size)
      ctx.line_to(cx+self.size, cy+self.size)
      ctx.line_to(cx-self.size, cy+self.size)
      ctx.line_to(cx-self.size, cy-self.size)
      ctx.stroke()

