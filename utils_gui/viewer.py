# Copyright (c) 2014, Tom SF Haines
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#  * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



import math

import cairo
from gi.repository import Gtk, Gdk, GdkPixbuf

from viewport_layer import *



class Viewer(Gtk.DrawingArea):
  """A drawing area with some extra functionality - specifically the ability to pan and zoom. It supports layers, which render the actual drawing area by implimenting a suitable interface, and the ability to add handlers for various input events. Note that mouse wheel and right clicking (dragging) are reserved for viewport control however."""
  def __init__(self):
    Gtk.DrawingArea.__init__(self)

    # Setup event handling...
    self.add_events(Gdk.EventMask.POINTER_MOTION_MASK | Gdk.EventMask.BUTTON_PRESS_MASK | Gdk.EventMask.BUTTON_RELEASE_MASK | Gdk.EventMask.SCROLL_MASK)
    self.connect('realize', self.__realise)
    self.connect('draw', self.__draw)
    self.connect('scroll-event', self.__on_wheel)
    self.connect('motion-notify-event', self.__on_move)
    self.connect('button-press-event', self.__on_press)
    self.connect('button-release-event', self.__on_release)

    # Storage for the various entities attached that handle drawing and events...
    self.layers = []
    self.on_draw = None
    self.on_click = None
    self.on_drag = None
    self.on_move = None
    self.on_paint = None

    # Viewport that is used, or None if it has not been initialised yet...
    self.viewport = None

    # Previous coordinates, for the mouse move event to know deltas...
    self.prev_x = 0.0
    self.prev_y = 0.0

    # Coordinates when the lmb went down, for the lmb drag handler (None when not down)...
    self.down_x = None
    self.down_y = None

    # Colours for some default behaviours, or None if these behaviours are disabled...
    self.bg_col = None
    self.drag_col = None

    # Dimensions of current contents, for limiting how far you can zoom out...
    self.width = 320.0
    self.height = 240.0


  def clear_layers(self):
    """Removes all existing layers, so you can create a new stack."""
    self.layers = []
    self.queue_draw()

  def add_layer(self, layer):
    """Adds a layer to the end of the layer list, returns an id you can use to delete it."""
    assert(isinstance(layer, Layer))
    ret = len(self.layers)
    self.layers.append(layer)
    return ret

  def del_layer(self, ident):
    """Terminates a layer given an ident."""
    self.layers[ident] = None

  def set_bg(self, r = None, g = None, b = None):
    """Sets a background colour to be drawn before all the layers - whilst usually reuired this defaults to off. You provide a colour as the usual three channels in the rnage [0,1], or None to disable."""
    if r==None: self.bg_col = None
    else: self.bg_col = (r, g, b)
  
  def get_bg(self):
    """Returns None if no background colour is selected, or a tuple (r,g,b) if it is."""
    return self.bg_col

  
  def set_on_draw(self, func):
    """Sets a function (Pass None if you want to disable it) that will be called when the display is redrawn - it gets no parameters."""
    self.on_draw = func
    
  def set_on_click(self, func):
    """Sets a function (Pass None if you want to disable it) that will be called when the user clicks in the viewport with the lmb. It will receive two parameters - the x and y (floating point) coordinates of where was clicked. All will be in the unscaled coordinate system."""
    self.on_click = func

  def set_on_drag(self, func):
    """Sets a function (Pass None if you want to disable it) that will be called when the user drags from one point in the viewport to another with the lmb down. It will receive four parameters - the x and y of the start and then the x and y of the end (All floating point). All will be in the unscaled coordinate system."""
    self.on_drag = func
    
  def set_drag_col(self, r = None, g = None, b = None):
    """Sets a colour for a line to be drawn when dragging; if no colour is provided (The default) then no line is drawn. Standard three channels, in range [0,1]. Only works if there is a dragging function, set with set_on_drag."""
    if r==None: self.drag_col = None
    else: self.drag_col = (r, g, b)

  def set_on_move(self, func):
    """Sets a function (Pass None if you want to disable it) that will be called when the user moves their mouse. It will be given (x,y) in the unscaled coordinate system as inputs; it should return True if it wants the screen to redrawn as a result of the movement."""
    self.on_move = func
  
  def set_on_paint(self, func):
    """Sets a function (Pass None to disable) that is called when dragging across the image with the lmb down for each on move sample. Input will be (start x, start y, end x, end y) in the unscaled coordinate system; return True to redraw the screen, False to not."""
    self.on_paint = func
    
    
  def reset_view(self):
    """Resets the zoom and pan, so that it reverts to the default."""
    self.viewport = None
    self.queue_draw()
  
  def get_viewport(self):
    """Returns the current viewport, or None if it is not defined."""
    return self.viewport

  def one_to_one(self):
    """Adjusts the viewport such that it is a one to one between the viewing region and the actual window."""

    if self.viewport!=None:
      cx = 0.5 * (self.viewport.start_x + self.viewport.end_x)
      cy = 0.5 * (self.viewport.start_y + self.viewport.end_y)

      ox = 0.5 * self.viewport.width
      oy = 0.5 * self.viewport.height

      self.viewport.start_x = cx - ox
      self.viewport.end_x = cx + ox
      self.viewport.start_y = cy - oy
      self.viewport.end_y = cy + oy

      self.queue_draw()

  
  def __realise(self, widget):
    self.get_window().set_cursor(Gdk.Cursor(Gdk.CursorType.CROSSHAIR))
  
  
  def __draw(self, widget, ctx):
    """Draws everything - basically an optional background, all layers and an optional drag line."""
    # If there is a callback method call it...
    if self.on_draw!=None:
      self.on_draw()

    # Optionally reset the background colour...
    if self.bg_col!=None:
      ctx.save()
      ctx.set_source_rgb(self.bg_col[0], self.bg_col[1], self.bg_col[2])
      ctx.paint()
      ctx.restore()

    # If not currently set initialise the viewport to the default zoom/view...
    if self.viewport==None and len(self.layers)!=0:
      self.height, self.width = reduce(lambda pa, pb: (max(pa[0],pb[0]), max(pa[1],pb[1])), map(lambda l: l.get_size(), self.layers))

      self.viewport = Viewport(1, 1, 0.0, 0.0, self.width, self.height)
      self.viewport.viewer = self

    # Update the viewport for the widget dimensions...
    if self.viewport!=None:
      size = widget.get_allocation()
      self.viewport.width = size.width
      self.viewport.height = size.height

      # Adjust aspect ratio, with adjustment to avoid showing less content...
      aspX = self.viewport.width / float(self.viewport.end_x - self.viewport.start_x)
      aspY = self.viewport.height / float(self.viewport.end_y - self.viewport.start_y)

      if math.fabs(aspX-aspY)>1e-3:
        if aspX>aspY:
          # Need to scale the width...
          scale = aspX / aspY
          offset = 0.5*(scale * (self.viewport.end_x - self.viewport.start_x) - (self.viewport.end_x - self.viewport.start_x))
          self.viewport.start_x -= offset
          self.viewport.end_x += offset

        else:
          # Need to scale the height...
          scale = aspY / aspX
          offset = 0.5*(scale * (self.viewport.end_y - self.viewport.start_y) - (self.viewport.end_y - self.viewport.start_y))
          self.viewport.start_y -= offset
          self.viewport.end_y += offset

    # Iterate and draw each layer in turn...
    for layer in self.layers:
      ctx.save()
      layer.draw(ctx, self.viewport)
      ctx.restore()

    # Optionally draw a drag line, if the user is dragging...
    if self.drag_col!=None and self.down_x!=None:
      ctx.save()

      ctx.set_line_width(1.0)
      ctx.set_source_rgb(self.drag_col[0], self.drag_col[1], self.drag_col[2])
      ctx.move_to(self.down_x, self.down_y)
      ctx.line_to(self.prev_x, self.prev_y)
      ctx.stroke()

      ctx.restore()


  def zoom(self, way):
    """True to zoom in, False to zoom out, or a number for fine control - positive to zoom in."""
    if self.viewport!=None:
      cx = 0.5 * (self.viewport.start_x + self.viewport.end_x)
      cy = 0.5 * (self.viewport.start_y + self.viewport.end_y)

      ox = self.viewport.end_x - cx
      oy = self.viewport.end_y - cy

      if isinstance(way, float):
        mult = math.pow(0.992, way)
        ox *= mult
        oy *= mult
      elif way:
        ox *= 0.75
        oy *= 0.75
      else:
        ox /= 0.75
        oy /= 0.75

      if ox<1.0 or oy<1.0:
        if ox<oy: mult = 1.0/ox
        else: mult = 1.0/oy
        ox *= mult
        oy *= mult
      elif ox>(self.width*2.0) and oy>(self.height*2.0):
        m1 = (self.width*2.0) / ox
        m2 = (self.height*2.0) / oy
        mult = max(m1, m2)
        ox *= mult
        oy *= mult

      self.viewport.start_x = cx - ox
      self.viewport.end_x = cx + ox
      self.viewport.start_y = cy - oy
      self.viewport.end_y = cy + oy

      self.queue_draw()

  def __on_wheel(self, widget, event):
    """The mouse wheel causes the view to zoom."""
    self.zoom((event.direction &Gdk.ScrollDirection.DOWN)==0)


  def move(self, off_x, off_y):
    """Moves the view, with the offset in view pixels."""
    if self.viewport!=None:
      # Convert to viewport coordinates...
      off_x /= float(self.viewport.width)
      off_y /= float(self.viewport.height)

      off_x *= self.viewport.end_x - self.viewport.start_x
      off_y *= self.viewport.end_y - self.viewport.start_y

      # Apply...
      self.viewport.start_x += off_x
      self.viewport.end_x += off_x
      self.viewport.start_y += off_y
      self.viewport.end_y += off_y

      # Redraw...
      self.queue_draw()

  def __on_move(self, widget, event):
    """Handles the user scrolling by holding down the right mouse button or middle mouse button. Also works with button on pen of Wacom tablets."""

    # If any of the relevent scrolling devics are pressed calculate and apply the relevent motion...
    render = False
    
    b1 = (event.state & Gdk.ModifierType.BUTTON1_MASK)!=0
    b2 = (event.state & Gdk.ModifierType.BUTTON2_MASK)!=0
    b3 = (event.state & Gdk.ModifierType.BUTTON3_MASK)!=0
    pressure = event.get_axis(Gdk.AxisUse.PRESSURE)
    
    if b1 and self.on_paint!=None:
      render = render or self.on_paint(self.prev_x, self.prev_y, event.x, event.y)
    elif b2 or (b3 and pressure==None):
      # Get offset in screen coordinates...
      off_x = self.prev_x - event.x
      off_y = self.prev_y - event.y

      self.move(off_x, off_y)

    elif b3 and pressure!=None:
      # Get vertical offset, for zoom purposes...
      off_y = self.prev_y - event.y

      self.zoom(off_y)
    
    # Handle the user supplied on move handler if it has been provided...
    if self.on_move!=None:
      render = render or self.on_move(event.x, event.y)

    # Redraw if needed, noting that move does this anyway...
    if render==True or (event.state & Gdk.ModifierType.BUTTON1_MASK)!=0:
      self.queue_draw()

    # Record the previous values and return...
    self.prev_x = event.x
    self.prev_y = event.y
    return True


  def __on_press(self, widget, event):
    """Records the start of a button press."""
    if event.button==1 and (self.on_click!=None or self.on_drag!=None):
      self.down_x = event.x
      self.down_y = event.y
      return True
    return False


  def __on_release(self, widget, event):
    """Handle user input, as either a click or drag with the left mouse button."""
    if event.button==1 and (self.on_click!=None or self.on_drag!=None) and self.down_x!=None:
      if self.on_click==None:
        self.on_drag(self.down_x, self.down_y, self.prev_x, self.prev_y)
      elif self.on_drag==None:
        self.on_click(self.prev_x, self.prev_y)
      else:
        dx = self.down_x - self.prev_x
        dy = self.down_y - self.prev_y
        dist = math.sqrt(dx*dx + dy*dy)
        if dist<4.0: self.on_click(self.prev_x, self.prev_y)
        else: self.on_drag(self.down_x, self.down_y, self.prev_x, self.prev_y)

    # Clean up...
    self.down_x = None
    self.down_y = None
