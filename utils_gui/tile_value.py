# Copyright (c) 2014, Tom SF Haines
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#  * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



import time
import math
import numpy
from collections import OrderedDict

import cairo
from gi.repository import Gdk, GdkPixbuf

from viewport_layer import *



class TileValue(Layer):
  """A wrapper for rendering an image of uint8, where you can choose a colour for each, or show through the background if you want. Has a tile cache, to make it suitably fast to render."""
  def __init__(self, values, tile_size=256, cache_size = 1024):
    """Initalises the value viewer; you can use None if you would rather initialise it later."""
    # Store various bits of info...
    self.tile_size = tile_size
    self.cache_size = cache_size
    self.visible = True

    # Setup the cache and stroe the values...
    self.cache = OrderedDict()
    self.cache_key = 0 # Incrimented each time the contents is replaced, and included in the tile key - avoids memory churn.
    self.set_values(values)
    
    # Indicates if we are using interpolation or not...
    self.interpolate = False
    
    # The colours to use for the mask channels, with None to let the background show through...
    # (If we index outside the array then that is None.)
    self.colour = []
    
    # A temporary, used when creating tiles...
    self.temp = cairo.ImageSurface(cairo.FORMAT_ARGB32, tile_size, tile_size)
    self.temp_ctx = cairo.Context(self.temp)
  

  def set_values(self, values):
    """Replaces the current values array with a new one."""
    # Default values for if None is provided...
    if values==None: values = numpy.zeros((480, 640), dtype=numpy.uint8)
    
    # Some shared setup stuff...
    stride = cairo.ImageSurface.format_stride_for_width(cairo.FORMAT_A8, values.shape[1])
    mask = numpy.empty((values.shape[0], stride), dtype=numpy.uint8)
    temp = cairo.ImageSurface.create_for_data(mask, cairo.FORMAT_A8, mask.shape[1], mask.shape[0], stride)
    
    # For each value create an alpha mask...
    self.values = []
    for i in xrange(values.max()+1):
      # Create a mask for this layer...
      mask[:,:] = 0
      mask[values==i] = 255
    
      # Copy the mask into perminant storage...
      ms = cairo.ImageSurface(cairo.FORMAT_A8, mask.shape[1], mask.shape[0])
      
      ctx = cairo.Context(ms)
      ctx.set_source_surface(temp, 0, 0)
      ctx.paint()
      
      self.values.append(ms)
    
    # Clean up temp - its dependent on the memory in mask...
    del temp
    
    # Change the cache key, so we recycle old tiles rather than use them...
    self.cache_key += 1

  
  def get_size(self):
    """Returns the tuple (height, width) for the original size."""
    return (self.values[0].get_height(), self.values[0].get_width())
  
  
  def get_interpolate(self):
    """Returns True if the mask is interpolated, False if it is not."""
    return self.interpolate
    
  def set_interpolate(self, interpolate):
    """True to interpolate, False to not. Defaults to True"""
    self.interpolate = interpolate
  
  
  def get_colour(self, value):
    """Returns the colour used for the given part of the mask, as 4 tuple with the last element alpha, or None if it is transparent."""
    if value < len(self.colour): return self.colour[value]
    else: return None
  
  def set_colour(self, value, col = None):
    """Sets the colour for a value, as a 4-tuple (r,g,b,a), or None for totally transparent."""
    if len(self.colour)<=value: self.colour += [None] * (value + 1 - len(self.colour))
    self.colour[value] = col
    self.cache_key += 1
  
  
  def fetch_tile(self, scale, bx, by, extra = 0):
    """Fetches a tile, given a scale (Multiplication of original dimensions.) and base coordinates (bx, by) - the tile will be the size provided on class initialisation. This request is going through a caching layer, which will hopefuly have the needed tile. If not it will be generated and the tile that has been used the least recently thrown away."""
    
    # Instance the key for this tile and check if it is in the cache, if so grab it, rearrange it to the top of the queue and return it...
    key = (scale, bx, by, self.interpolate, self.cache_key)
    if key in self.cache:
      ret = self.cache[key]
      del self.cache[key]
      self.cache[key] = ret
      return ret
    
    # Select a new tile to render into - either create one or recycle whatever is going to die at the end of the cache...
    if len(self.cache)>=(self.cache_size-extra):
      ret = self.cache.popitem()[1]
    else:
      ret = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.tile_size, self.tile_size)
    
    # Render the tile, with the appropriate scaling...
    ctx = cairo.Context(ret)
    ctx.save()
    ctx.set_operator(cairo.OPERATOR_SOURCE)
    ctx.set_source_rgba(0.0, 0.0, 0.0, 0.0)
    ctx.paint()
    ctx.restore()
    
    if scale<8.0:
      self.temp_ctx.save()
      self.temp_ctx.translate(-bx, -by)
      self.temp_ctx.scale(scale, scale)
      
      for i, mask in enumerate(self.values[:len(self.colour)]):
        col = self.colour[i]
        if col!=None:
          self.temp_ctx.set_source_rgba(col[0], col[1], col[2], col[3])
          self.temp_ctx.set_operator(cairo.OPERATOR_SOURCE)
          self.temp_ctx.paint()
          
          oasp = cairo.SurfacePattern(mask)
          oasp.set_filter(cairo.FILTER_BILINEAR if self.interpolate else cairo.FILTER_NEAREST)
          
          self.temp_ctx.set_source(oasp)
          self.temp_ctx.set_operator(cairo.OPERATOR_DEST_IN)
          self.temp_ctx.paint()

          ctx.set_source_surface(self.temp)
          ctx.paint()

      self.temp_ctx.restore()
    else:
      # Cairo has an annoying habit of failing if you push the scale too far - this makes it recursive for such situations...
      power = math.log(scale) / math.log(2.0)
      child_scale = 2**(int(math.floor(power))-2)
      remain_scale = scale / child_scale
      
      child_bx = int(math.floor((bx-1) / (remain_scale*0.5*self.tile_size))) * self.tile_size//2
      child_by = int(math.floor((by-1) / (remain_scale*0.5*self.tile_size))) * self.tile_size//2
        
      child = self.fetch_tile(child_scale, child_bx, child_by, extra+1)
      
      oasp = cairo.SurfacePattern(child)
      oasp.set_filter(cairo.FILTER_BILINEAR if self.interpolate else cairo.FILTER_NEAREST)
      
      ctx.translate(-(bx - child_bx*remain_scale), -(by - child_by*remain_scale))
      ctx.scale(remain_scale, remain_scale)
      ctx.set_source(oasp)
      ctx.paint()
    
    # Store the tile in the cache and return...
    self.cache[key] = ret
    return ret
    
    
  def draw(self, ctx, vp):
    """Draws the image into the provided context, matching the provided viewport."""
    if not self.visible:
      return
    
    # Calculate the scale, base coordinates and number of tiles in each dimension...
    scale = vp.width / float(vp.end_x - vp.start_x)
    
    base_x = int(math.floor(vp.start_x * scale))
    base_y = int(math.floor(vp.start_y * scale))
    
    tile_bx = int(math.floor(base_x / float(self.tile_size))) * self.tile_size
    tile_by = int(math.floor(base_y / float(self.tile_size))) * self.tile_size
    
    # Loop and draw the required tiles, with the correct offset...
    y = tile_by
    while y < int(base_y+vp.height+1):
      x = tile_bx
      while x < int(base_x+vp.width+1):
        tile = self.fetch_tile(scale, x, y)
        
        ctx.set_source_surface(tile, x - base_x, y - base_y)
        ctx.rectangle(x - base_x, y - base_y, self.tile_size, self.tile_size)
        ctx.fill()
        
        x += self.tile_size
      y += self.tile_size
