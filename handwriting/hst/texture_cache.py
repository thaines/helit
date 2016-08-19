# Copyright 2016 Tom SF Haines

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import os.path
from collections import OrderedDict

import numpy

import cairo
from gi.repository import Gdk, GdkPixbuf



class TextureCache:
  """Acts like a dictionary that goes from texture filename -> numpy array, except its actually loading them on the fly, with a clamp on how many it keeps in memory at any given time. Returns None if a file does not exist. Will automatically use alpha files if they exist."""
  def __init__(self, max_tex = 32):
    self.limit = max_tex
    self.cache = OrderedDict()
  
  
  def __getitem__(self, fn):
    """Converts a filename into a numpy array of the texture, or returns None if there is no such file."""
    # Handle it already being in the cache...
    if fn in self.cache:
      ret = self.cache[fn]
      del self.cache[fn]
      self.cache[fn] = ret # Put it to the back of the list.
      return ret
    
    # Load the file and convert it into a numpy array...
    alt_fn =  os.path.splitext(fn)[0] + '_alpha.png'
    if os.path.exists(alt_fn):
      pixbuf = GdkPixbuf.Pixbuf.new_from_file(alt_fn)
    elif os.path.exists(fn):
      pixbuf = GdkPixbuf.Pixbuf.new_from_file(fn)
    else:
      return None
      
    texture = cairo.ImageSurface(cairo.FORMAT_ARGB32, pixbuf.get_width(), pixbuf.get_height())
    
    ctx = cairo.Context(texture)
    ctx.set_operator(cairo.OPERATOR_SOURCE)
    Gdk.cairo_set_source_pixbuf(ctx, pixbuf, 0, 0)
    ctx.paint()
    
    del pixbuf
      
    ret = numpy.fromstring(texture.get_data(), dtype=numpy.uint8)
    ret = ret.reshape((texture.get_height(), texture.get_width(), -1))
    
    # Handle cache expiry and return...
    self.cache[fn] = ret
    if len(self.cache) > self.limit:
      self.cache.popitem(False)
    return ret
