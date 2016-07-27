# Copyright 2016 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import numpy
from collections import OrderedDict

import cairo
from gi.repository import Gdk, GdkPixbuf

from utils_gui.viewport_layer import *



class LineLayer(Layer):
  """A wrapper for rendering a line, as represented by a LineGraph. Has three rendering modes, depending on if you want to match the scale of the line to the viewport or not."""

  RENDER_CORRECT = 0 # Does everything correctly.
  RENDER_THIN = 1 # Draws the line to be a single pixel wide, regardless of its original scale.
  RENDER_ONE = 2 # Draws the line to be one pixel wide, taking into account scale.
  RENDER_WEIRD = 3 # Draws the line correctly without taking into account the scale, so its only actually correct for a 1:1 zoom.

  def __init__(self, tile_size=256, cache_size = 1024):
    """Initalises the line viewer."""
    # Set the line object and setup the cache...
    self.set_line(None)

    # Store various bits of info...
    self.tile_size = tile_size
    self.cache_size = cache_size

    # Which mode to render with...
    self.mode = LineLayer.RENDER_CORRECT

    # The colours to use for the line...
    self.colour_zero = (0.0, 0.0, 1.0)
    self.colour_one  = (0.0, 0.0, 0.0)
    self.colour_two  = (1.0, 0.0, 0.0)


  def set_line(self, line):
    """Replaces the current line with a new one; you can send in None to disable this layer."""
    # Default mask for if None is provided...
    self.line = line

    # Reset the cache...
    self.cache = OrderedDict()


  def get_line(self):
    """Returns the LineGraph object being rendered, or None if there is none."""
    return self.line

  def get_size(self):
    """Returns the tuple (height, width) for the original size - basically the dimensions required to fully include the line, assuming it never goes negative."""
    if self.line==None: return (240, 320)
    min_x, max_x, min_y, max_y = self.line.get_bounds()
    return (int(numpy.ceil(max_y)), int(numpy.ceil(max_x)))


  def get_mode(self):
    """Returns the rendering mode - one of the class constants."""
    return self.mode

  def set_mode(self, mode):
    """Set the rendering mode to one of the options provided as class constants."""
    self.mode = mode


  def get_colour(self):
    """Returns the colour used for the line - a 3-tuple of tuples, where each is 3 floats, (r,g,b), in the range 0.0 to 1.0. The first is for density 0, the next for density 1, the final for density 2"""
    return self.colour

  def set_colour(self, zero = (0.0, 0.0, 1.0), one = (0.0, 0.0, 0.0), two = (1.0, 0.0, 0.0)):
    """Sets the colour the line is rendered with - 3 colours actually, for a density of zero, one and two."""
    self.colour_zero = zero
    self.colour_one  = one
    self.colour_two  = two


  def fetch_tile(self, scale, bx, by):
    """Fetches a tile, given a scale (Multiplication of original dimensions.) and base coordinates (bx, by) - the tile will be the size provided on class initialisation. This request is going through a caching layer, which will hopefuly have the needed tile. If not it will be generated and the tile that has been used the least recently thrown away."""

    # Instance the key for this tile and check if it is in the cache, if so grab it, rearrange it to the top of the queue and return it...
    key = (scale, bx, by, self.mode)
    if key in self.cache:
      ret = self.cache[key]
      del self.cache[key]
      self.cache[key] = ret
      return ret

    # Select a new tile to render into - either create one or recycle whatever is going to die at the end of the cache...
    if len(self.cache)>=self.cache_size:
      ret = self.cache.popitem()[1]
    else:
      ret = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.tile_size, self.tile_size)

    # Render the tile...
    ## Background...
    ctx = cairo.Context(ret)

    ctx.save()
    ctx.set_source_rgba(0.0, 0.0, 0.0, 0.0)
    ctx.set_operator(cairo.OPERATOR_SOURCE)
    ctx.paint()
    ctx.restore()

    ## Lines...
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    
    if self.line!=None:
      rbx = bx / scale
      rby = by / scale
      rsize = self.tile_size / scale
      for es in self.line.within(rbx, rbx + rsize, rby, rby + rsize):
        for ei in xrange(*es.indices(self.line.edge_count)):
          edge = self.line.get_edge(ei)
          vf = self.line.get_vertex(edge[0])
          vt = self.line.get_vertex(edge[1])
          
          density = 0.5 * (vf[6] + vt[6])
          if density<1.0:
            low  = self.colour_zero
            high = self.colour_one
          else:
            density -= 1.0
            if density>1.0: density = 1.0
            
            low  = self.colour_one
            high = self.colour_two
            
          r = (1.0-density) * low[0] + density * high[0]
          g = (1.0-density) * low[1] + density * high[1]
          b = (1.0-density) * low[2] + density * high[2]
          ctx.set_source_rgb(r, g, b)
            
          if self.mode==LineLayer.RENDER_CORRECT: ctx.set_line_width(scale * (vf[4] + vt[4]))
          elif self.mode==LineLayer.RENDER_THIN: ctx.set_line_width(1.0)
          elif self.mode==LineLayer.RENDER_ONE: ctx.set_line_width(scale)
          else: ctx.set_line_width(vf[4] + vt[4])

          ctx.move_to(scale * vf[0] - bx, scale * vf[1] - by)
          ctx.line_to(scale * vt[0] - bx, scale * vt[1] - by)

          ctx.stroke()

    # Store the tile in the cache and return...
    self.cache[key] = ret
    return ret


  def draw(self, ctx, vp):
    """Draws the mask into the provided context, matching the provided viewport."""

    # Calculate the scale, base coordinates and number of tiles in each dimension...
    scale = vp.width / float(vp.end_x - vp.start_x)

    base_x = int(numpy.floor(vp.start_x * scale))
    base_y = int(numpy.floor(vp.start_y * scale))

    tile_bx = int(numpy.floor(base_x / float(self.tile_size))) * self.tile_size
    tile_by = int(numpy.floor(base_y / float(self.tile_size))) * self.tile_size

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
