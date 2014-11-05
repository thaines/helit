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



class TileImage(Layer):
  """A wrapper around an image, that breaks it into tiles for accelerated rendering. Has full support for scaling, so you can get tiles at any scale you desire."""
  def __init__(self, fn, tile_size=256, cache_size = 1024):
    """Loads the given image (None for a default one) and sets the various parameters that control the tiling behaviour. Note that the tile size must be a multiple of 4, otherwise there will be nasty artifacts."""
    # The replace call does the file loading and cache setup...
    self.load(fn)
    
    # Store various bits of info...
    self.tile_size = tile_size
    self.cache_size = cache_size
    
    # Indicates if we are using interpolation or not...
    self.interpolate = True
    
    # A tint, so we can fade the image out to put content over the top of it...
    self.tint = 0.0
    
    # So we can hide it...
    self.visible = True


  def load(self, fn):
    """Replaces the current image with a new one. You can provide none to get a default image, as standin until you load a real image."""
    # Load the image and convert it into a cairo surface...
    if fn==None:
      self.original = cairo.ImageSurface(cairo.FORMAT_ARGB32, 640, 480)
      
      ctx = cairo.Context(self.original)
      ctx.set_source_rgba(1.0, 1.0, 1.0, 0.0)
      ctx.paint()
    else:
      pixbuf = GdkPixbuf.Pixbuf.new_from_file(fn)
      self.original = cairo.ImageSurface(cairo.FORMAT_ARGB32, pixbuf.get_width(), pixbuf.get_height())
    
      ctx = cairo.Context(self.original)
      Gdk.cairo_set_source_pixbuf(ctx, pixbuf, 0, 0)
      ctx.paint()
    
      del pixbuf
    
    # Reset the cache...
    self.cache = OrderedDict()
  
  
  def set_blank(self, width = 640, height = 480, r = 1.0, g = 1.0, b = 1.0, a = 1.0):
    """Allows you to set it up to contain a blank image, with the given colours and alpha."""
    self.original = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
      
    ctx = cairo.Context(self.original)
    ctx.set_source_rgba(r, g, b, a)
    ctx.paint()
    
    self.cache = OrderedDict()
  
  
  def from_array(self, image):
    """Replaces the current image, after being given an array of uint8, [y, x, c], where alpha is c=3, blue is c=0, green is c=1 and red is c=2."""
    
    stride = cairo.ImageSurface.format_stride_for_width(cairo.FORMAT_ARGB32, image.shape[1])
    if stride!=image.strides[0]:
      image = numpy.concatenate((image, numpy.zeros((image.shape[0], (stride-image.strides[0])/4, 4), dtype=numpy.uint8)), axis=1).copy('C')
    temp = cairo.ImageSurface.create_for_data(image, cairo.FORMAT_ARGB32, image.shape[1], image.shape[0], stride)
    
    self.original = cairo.ImageSurface(cairo.FORMAT_ARGB32, image.shape[1], image.shape[0])
    
    ctx = cairo.Context(self.original)
    ctx.set_operator(cairo.OPERATOR_SOURCE)
    ctx.set_source_surface(temp, 0, 0)
    ctx.paint()
    
    # Reset the cache...
    self.cache = OrderedDict()
  
  
  def get_original(self):
    """Returns the original image, as an ImageSurface."""
    return self.original
  
  def get_size(self):
    """Returns the tuple (height, width) for the original size."""
    return (self.original.get_height(), self.original.get_width())
  
  def get_interpolate(self):
    """Returns True if the image is interpolated, False if it is not."""
    return self.interpolate
    
  def set_interpolate(self, interpolate):
    """True to interpolate, False to not. Defaults to True"""
    self.interpolate = interpolate
    
  def get_tint(self):
    """Returns the tint strength used when drawing the image."""
    return self.tint
    
  def set_tint(self, t):
    """Sets a tint, that is only used during drawing - essentially fades out the image by interpolating it towards white, with the original image at t=0.0, and just white at t=1.0"""
    self.tint = t
  
  
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
      ret = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.tile_size, self.tile_size)
    
    # Render the tile, with the appropriate scaling...
    ctx = cairo.Context(ret)
    ctx.save()
    ctx.set_operator(cairo.OPERATOR_SOURCE)
    ctx.set_source_rgba(1.0, 1.0, 1.0, 0.0)
    ctx.paint()
    ctx.restore()
    
    if scale<8.0:
      oasp = cairo.SurfacePattern(self.original)
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
    
    # Do some tinting...
    if self.tint>1e-4:
      ctx.set_source_rgba(1.0, 1.0, 1.0, self.tint)
      ctx.paint()


if __name__ == '__main__':
  ti = TileImage('test.png')
  
  t1 = ti.fetch_tile(1.0, 0, 0)
  t1.write_to_png('t1.png')
  
  t2 = ti.fetch_tile(1.0, 256, 0)
  t2.write_to_png('t2.png')
  
  t3 = ti.fetch_tile(0.5, 0, 0)
  t3.write_to_png('t3.png')
  
  t4 = ti.fetch_tile(2.0, 0, 0)
  t4.write_to_png('t4.png')
  
  t5 = ti.fetch_tile(2.0, 256, 0)
  t5.write_to_png('t5.png')
  
  t6 = ti.fetch_tile(2.0, 256, 256)
  t6.write_to_png('t6.png')
  
  t7 = ti.fetch_tile(0.1, 0, 0)
  t7.write_to_png('t7.png')
  
  t8 = ti.fetch_tile(16.0, 256, 256)
  t8.write_to_png('t8.png')
