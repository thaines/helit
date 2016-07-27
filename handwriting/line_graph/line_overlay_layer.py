# Copyright 2016 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import numpy

from utils_gui.viewport_layer import *



class LineOverlayLayer(Layer):
  """Augments a line, as drawn by a LineLayer, with all the extra information stored in a LineGraph, specifically the split locations and links, drawn only for the segment closest to the mouse cursor."""
  def __init__(self):
    self.line = None
    
    self.x = 0.0
    self.y = 0.0
    
    self.segment = None
    self.bound = None
    self.edge = None
    self.t = None
  
  
  def set_line(self, line):
    """Replaces the current line with a new one; you can send in None to disable this layer."""
    self.line = line


  def get_line(self):
    """Returns the LineGraph object being rendered, or None if there is none."""
    return self.line
  
  def get_size(self):
    """Returns the tuple (height, width) for the original size - basically the dimensions required to fully include the line, assuming it never goes negative."""
    if self.line==None: return (240, 320)
    min_x, max_x, min_y, max_y = self.line.get_bounds()
    return (int(numpy.ceil(max_y)), int(numpy.ceil(max_x)))
  
  
  def set_segment(self, segment):
    """Call this to set which segment is game."""
    self.segment = segment
    self.bound = None
    
  def get_segment(self):
    """Returns the index of the segment that is nearest the mouse cursor, or None if this is not currently defined."""
    return self.segment
  
  def get_edge(self):
    """Returns the tuple (edge,t) of the one closest to the mouse cursor, or None if that is not defined."""
    if self.edge==None: return None
    return (self.edge, self.t)
  
  
  def set_mouse(self, x, y):
    """This is given coordinates in the screen coordinate system, and updates the view that the system provides. After calling this a redraw should probably occur."""
    self.x = x
    self.y = y


  def draw(self, ctx, vp):
    """Draws the many overlay details onto the viewport."""
    
    if self.line:
      # Get the segment we are closest to, which defines the set of things we are going to visualise...
      ## Convert the mouse coordinates to line graph coordinates...
      x, y = vp.view_to_original(self.x, self.y)
    
      ## Find the closest edge...
      distance, self.edge, self.t = self.line.nearest(x, y)
      
      ## Find which segment the edge location belongs to...
      if self.line.segments<0: self.segment = None
      seg = self.line.get_segment(self.edge, self.t)
      
      
      # Draw all links and splits that involve the current segment...
      ## Fetch them all...
      things = self.line.get_splits(seg)
      
      ## Draw the links...
      splits = []
      ctx.set_line_width(1.0)
      ctx.set_source_rgba(0.8, 0.8, 0.0, 0.8)
      for thing in things:
        if len(thing)==3: splits.append(thing)
        else:
          start = self.line.get_point(thing[0], thing[1])
          end = self.line.get_point(thing[2], thing[3])
          
          sx, sy = vp.original_to_view(start[0], start[1])
          ex, ey = vp.original_to_view(end[0], end[1])
          
          ctx.move_to(sx, sy)
          ctx.line_to(ex, ey)
          ctx.stroke()
      
      ## Draw the splits...
      ctx.set_line_width(2.0)
      ctx.set_source_rgba(1.0, 0.3, 0.0, 0.8)
      for split in splits:
        # Get the location of the split...
        point = self.line.get_point(split[0], split[1])
        lx, ly = vp.original_to_view(point[0], point[1])
        
        # Get a normal vector for it, pointing towards the segment...
        es = self.line.get_point(split[0], 0.0)
        ee = self.line.get_point(split[0], 1.0)
        
        nx = ee[0] - es[0]
        ny = ee[1] - es[1]
        nl = numpy.sqrt(nx*nx + ny*ny)
        
        if nl<1e-6:
          nx = 1.0
          ny = 0.0
        else:
          nx /= nl
          ny /= nl
        
        mult = vp.original_to_view_scale(point[4])
        nx *= mult
        ny *= mult
                
        # Get a tangent space...
        tx = ny
        ty = -nx
        
        nx *= split[2]
        ny *= split[2]
          
        # Draw the end cap...
        ctx.move_to(lx + tx*1.1 + nx*0.8, ly + ty*1.1 + ny*0.8)
        ctx.line_to(lx + tx, ly + ty)
        ctx.line_to(lx - tx, ly - ty)
        ctx.line_to(lx - tx*1.1 + nx*0.8, ly - ty*1.1 + ny*0.8)

        ctx.stroke()
      
      # Draw a bounding box around the segment that we are hovering over/near...
      ## Get the bounds; includes caching...
      if self.segment==seg and self.bound!=None:
        bound = self.bound
      else:
        bound = self.line.get_bounds(seg)
        self.segment = seg
        self.bound = bound
      
      ## Draw the bounds...
      bx, by = vp.original_to_view(bound[0], bound[2])
      bw = vp.original_to_view_scale(bound[1]-bound[0])
      bh = vp.original_to_view_scale(bound[3]-bound[2])
      
      ctx.set_line_width(1.0)
      ctx.set_source_rgba(0.5, 0.0, 1.0, 0.9)
      ctx.rectangle(bx, by, bw, bh)
      ctx.stroke()
      

      # Render the nearest point indicator...
      ## Get the edge details...
      x, y, u, v, w, radius, density, weight = self.line.get_point(self.edge, self.t)
      
      ## Convert to view coordinates and scale...
      vx, vy = vp.original_to_view(x, y)
      vr = vp.original_to_view_scale(radius)
      
      ## Render...
      ctx.set_line_width(1.0)
      ctx.set_source_rgb(0.8, 0.4, 0.0)
      ctx.arc(vx, vy, vr, 0.0, numpy.pi*2.0)
      ctx.stroke()
      
      ctx.set_source_rgb(0.8, 0.0, 0.0)
      ctx.arc(vx, vy, 1.0, 0.0, numpy.pi*2.0)
      ctx.stroke()
      
      
      # Render the density/orientation indicator...      
      edge = self.line.get_edge(self.edge)
      v_from = self.line.get_vertex(edge[0])
      v_to = self.line.get_vertex(edge[1])
      
      dx = v_to[0] - v_from[0]
      dy = v_to[1] - v_from[1]
      dx, dy = -dy, dx
      
      l = numpy.sqrt(dx*dx + dy*dy)
      if l>1e-3:
        dx /= l
        dy /= l
      
        dx *= radius * density
        dy *= radius * density
      
        sx = x - dx
        sy = y - dy
        ex = x + dx
        ey = y + dy
      
        sx, sy = vp.original_to_view(sx, sy)
        ex, ey = vp.original_to_view(ex, ey)
        
        ctx.set_line_width(1.0)
        ctx.set_source_rgb(0.4, 0.8, 0.0)
      
        ctx.move_to(sx, sy)
        ctx.line_to(ex, ey)
        ctx.stroke()
