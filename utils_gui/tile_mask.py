# Copyright (c) 2014, Tom SF Haines
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#  * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



import math
import numpy
from collections import OrderedDict

import cairo
from gi.repository import Gdk, GdkPixbuf

from viewport_layer import *



class TileMask(Layer):
  """A wrapper for rendering a mask, represented as a boolean numpy array. Allows you to select a colour for True and a colour for False, or allow the background to show through for either. Has a tile cache, to make it suitably fast to render."""
  def __init__(self, mask, tile_size=256, cache_size = 1024):
    """Initalises the mask viewer; you can use None for the mask if you would rather initialise it later."""
    # Set the mask and setup the cache...
    self.set_mask(mask)
    
    # Store various bits of info...
    self.tile_size = tile_size
    self.cache_size = cache_size
    
    # Indicates if we are using interpolation or not...
    self.interpolate = False
    
    # The colours to use for the mask channels, with None to let the background show through...
    self.colTrue = None
    self.colFalse = (0.0, 0.0, 1.0)
  
  def set_mask(self, mask):
    """Replaces the current mask with a new one."""
    # Default mask for if None is provided...
    if mask==None: mask = numpy.ones((480, 640), dtype=numpy.bool)
    
    # Munge the data into a surface...
    mask = mask.astype(numpy.uint8)
    mask[numpy.nonzero(mask)] = 255
    
    stride = cairo.ImageSurface.format_stride_for_width(cairo.FORMAT_A8, mask.shape[1])
    if stride!=mask.shape[1]:
      mask = numpy.concatenate((mask, numpy.zeros((mask.shape[0], stride-mask.shape[1]), dtype=numpy.uint8)), axis=1).copy('C')
    
    temp = cairo.ImageSurface.create_for_data(mask, cairo.FORMAT_A8, mask.shape[1], mask.shape[0], stride)      
    
    # Copy the mask into perminant storage...
    self.mask = cairo.ImageSurface(cairo.FORMAT_A8, mask.shape[1], mask.shape[0])
    
    ctx = cairo.Context(self.mask)
    ctx.set_source_surface(temp, 0, 0)
    ctx.paint()
    
    # Clean up temp - its dependent on the memory in mask...
    del temp
    
    # Reset the cache...
    self.cache = OrderedDict()
    
    
  def get_mask(self):
    """Returns the mask, as an ImageSurface with format A8."""
    return self.mask
  
  def get_size(self):
    """Returns the tuple (height, width) for the original size."""
    return (self.mask.get_height(), self.mask.get_width())
  
  
  def get_interpolate(self):
    """Returns True if the mask is interpolated, False if it is not."""
    return self.interpolate
    
  def set_interpolate(self, interpolate):
    """True to interpolate, False to not. Defaults to True"""
    self.interpolate = interpolate
  
  
  def get_false(self):
    """Returns the colour used for the False region of the mask, or None if it is transparent."""
    return self.colFalse

  def set_false(self, col = None):
    """Sets the colour associated with the False regions of the mask. A 3-tuple of rgb, in 0-1, or None if you want to show through the layer below."""
    self.colFalse = col

  def get_true(self):
    """Returns the colour used for the True region of the mask, or None if it is transparent."""
    return self.colTrue
  
  def set_true(self, col = None):
    """Sets the colour associated with the True regions of the mask. A 3-tuple of rgb, in 0-1, or None if you want to show through the layer below."""
    self.colTrue = col
  
  
  def fetch_tile(self, scale, bx, by, extra = 0):
    """Fetches a tile, given a scale (Multiplication of original dimensions.) and base coordinates (bx, by) - the tile will be the size provided on class initialisation. This request is going through a caching layer, which will hopefuly have the needed tile. If not it will be generated and the tile that has been used the least recently thrown away."""
    
    # Instance the key for this tile and check if it is in the cache, if so grab it, rearrange it to the top of the queue and return it...
    key = (scale, bx, by, self.interpolate)
    if key in self.cache:
      ret = self.cache[key]
      del self.cache[key]
      self.cache[key] = ret
      return ret
    
    # Select a new tile to render into - either create one or recycle whatever is going to die at the end of the cache...
    if len(self.cache)>=(self.cache_size-extra):
      ret = self.cache.popitem()[1]
    else:
      ret = cairo.ImageSurface(cairo.FORMAT_A8, self.tile_size, self.tile_size)
    
    # Render the tile, with the appropriate scaling...
    ctx = cairo.Context(ret)
    ctx.set_operator(cairo.OPERATOR_SOURCE)
    ctx.set_source_rgba(0.0, 0.0, 0.0, 0.0)
    ctx.paint()
    
    if scale<8.0:
      oasp = cairo.SurfacePattern(self.mask)
      oasp.set_filter(cairo.FILTER_BILINEAR if self.interpolate else cairo.FILTER_NEAREST)
    
      ctx.translate(-bx, -by)
      ctx.scale(scale, scale)
      ctx.set_source(oasp)
      ctx.paint()
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
    """Draws the mask into the provided context, matching the provided viewport."""
    
    # Calculate the scale, base coordinates and number of tiles in each dimension...
    scale = vp.width / float(vp.end_x - vp.start_x)
    
    base_x = int(math.floor(vp.start_x * scale))
    base_y = int(math.floor(vp.start_y * scale))
    
    tile_bx = int(math.floor(base_x / float(self.tile_size))) * self.tile_size
    tile_by = int(math.floor(base_y / float(self.tile_size))) * self.tile_size
    
    # Clip to only include the required region...
    ctx.rectangle(0.0 - base_x, 0.0 - base_y, scale*self.mask.get_width(), scale*self.mask.get_height())
    ctx.clip()
    
    # Loop and draw the required tiles, with the correct offset...
    y = tile_by
    while y < int(base_y+vp.height+1):
      x = tile_bx
      while x < int(base_x+vp.width+1):
        tile = self.fetch_tile(scale, x, y)
        
        if self.colTrue!=None:
          ctx.set_source_rgb(self.colTrue[0], self.colTrue[1], self.colTrue[2])
          ctx.mask_surface(tile, x - base_x, y - base_y)
        
        if self.colFalse!=None:
          d = numpy.frombuffer(tile.get_data(), dtype=numpy.uint8)
          d[:] = 255 - d[:]
          
          ctx.set_source_rgb(self.colFalse[0], self.colFalse[1], self.colFalse[2])        
          ctx.mask_surface(tile, x - base_x, y - base_y)
          
          d[:] = 255 - d[:]
        
        x += self.tile_size
      y += self.tile_size
