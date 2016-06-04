#! /usr/bin/env python

# Copyright 2016 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import os.path
import shutil
from collections import OrderedDict

import numpy
from scipy.misc import imread, toimage

from ply2 import ply2

from gi.repository import Gtk

from utils_gui.viewport_layer import *
from utils_gui.viewer import Viewer
from utils_gui.tile_image import TileImage

from hg.homography import match
from hg.transform import sample



class GridViewer(Layer):
  """Displays the grid of the homography, including storage of the corner locatins of the grid in normalised coordinates."""
  def __init__(self):
    self.size = (240, 320) # Height, width.
    self.corners = numpy.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=numpy.float32) # x then y, clockwise from origin.
    self.visible = True
    
    self.vp = None
   
   
  def get_size(self):
    return self.size
   
   
  def draw(self, ctx, vp):
    self.vp = vp
     
    if not self.visible:
      return
    
    ctx.set_source_rgb(0.9, 0.8, 0.7)
    
    # Horizontal lines...
    for t in map(lambda i: i / 26.0, xrange(27)):
      tm1 = 1.0 - t
      
      start_x = self.corners[0,0] * tm1 + self.corners[3,0] * t
      start_y = self.corners[0,1] * tm1 + self.corners[3,1] * t
      end_x = self.corners[1,0] * tm1 + self.corners[2,0] * t
      end_y = self.corners[1,1] * tm1 + self.corners[2,1] * t
       
      start_x *= self.size[1]
      start_y *= self.size[0]
      end_x *= self.size[1]
      end_y *= self.size[0]
      
      start_x, start_y = vp.original_to_view(start_x, start_y)
      end_x, end_y = vp.original_to_view(end_x, end_y)
      
      ctx.move_to(start_x, start_y)
      ctx.line_to(end_x, end_y)
    ctx.stroke()
    
    # Vertical lines...
    for t in map(lambda i: i / 26.0, xrange(27)):
      tm1 = 1.0 - t
      
      start_x = self.corners[0,0] * tm1 + self.corners[1,0] * t
      start_y = self.corners[0,1] * tm1 + self.corners[1,1] * t
      end_x = self.corners[3,0] * tm1 + self.corners[2,0] * t
      end_y = self.corners[3,1] * tm1 + self.corners[2,1] * t
       
      start_x *= self.size[1]
      start_y *= self.size[0]
      end_x *= self.size[1]
      end_y *= self.size[0]
       
      start_x, start_y = vp.original_to_view(start_x, start_y)
      end_x, end_y = vp.original_to_view(end_x, end_y)
      
      ctx.move_to(start_x, start_y)
      ctx.line_to(end_x, end_y)
    ctx.stroke()
    
    # Corners points...
    for i, col in [(0,(0.0, 0.0, 0.0)), (1,(0.8, 0.0, 0.6)), (2,(0.0, 0.8, 0.8)), (3,(0.0, 0.8, 0.2))]:
      ctx.set_source_rgb(*col)
       
      x = self.corners[i,0]
      y = self.corners[i,1]
       
      x *= self.size[1]
      y *= self.size[0]
      
      x, y = vp.original_to_view(x, y)
       
      ctx.arc(x, y, 6.0, 0.0, numpy.pi*2.0)
      ctx.stroke()
  
  
  def update(self, x, y):
    """To be called with the x and y coordinate when the user indicates an update - will snap the closest corner."""
    
    # Check we can do required conversions, as otherwise we havea  problem...
    if self.vp==None:
      return
    
    # Convert to internal coordinate system...
    x, y = self.vp.view_to_original(x, y)
    
    x = x / float(self.size[1])
    y = y / float(self.size[0])
    
    # Find nearest corner...
    best = None
    best_dist = numpy.inf # Actually distance squared
    
    for i in xrange(4):
      dist = (self.corners[i,0] - x)**2 + (self.corners[i,1] - y)**2
      
      if dist < best_dist:
        best = i
        best_dist = dist
    
    # Snap...
    self.corners[best,0] = x
    self.corners[best,1] = y



class Calibrate(Gtk.Window):
  def __init__(self):
    # Do the basic window setup...
    Gtk.Window.__init__(self, title='Colour Response Calibration')
    self.connect('delete-event', Gtk.main_quit)
    self.set_default_size(1024, 576)
    
    
    # Setup the viewing area...
    self.viewer = Viewer()
    self.viewer.set_bg(1.0, 1.0, 1.0)
    
    self.viewer.set_on_click(self.__on_click)
    
    
    # Add two images - one for the calibration target, one for the current image, and then just flip visibility depending on viewing mode...
    self.scan = TileImage(None)
    self.scan.visible = True
    self.viewer.add_layer(self.scan)
    
    self.calibration = TileImage('calibration_target.png')
    self.calibration.visible = False
    self.viewer.add_layer(self.calibration)
    
    
    # Setup two homographies, again one visible one invisible depending on mode...
    self.scan_grid = GridViewer()
    self.scan_grid.visible = False # So it doesn't show until the image is loaded.
    self.scan_grid.size = self.scan.get_size()
    self.viewer.add_layer(self.scan_grid)
    
    self.calibration_grid = GridViewer()
    self.calibration_grid.visible = False
    self.calibration_grid.size = self.calibration.get_size()
    self.viewer.add_layer(self.calibration_grid)
    
    
    # Prep the menus...
    uimanager = self.__create_uimanager()
    
    accelerator_group = uimanager.get_accel_group()
    self.add_accel_group(accelerator_group)
    
    menu_bar = uimanager.get_widget('/menu_bar')
    icon_bar = uimanager.get_widget('/icon_bar')
    
    # Layout the window...
    vertical = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
    vertical.pack_start(menu_bar, False, False, 0)
    vertical.pack_start(self.viewer, True, True, 0)
    vertical.pack_start(icon_bar, False, False, 0)
    self.add(vertical)
  
  
  def __create_uimanager(self):
    # The file menu...
    action_file = Gtk.Action("file", "File", None, None)
    action_open_scan = Gtk.Action('open_scan','Open Scan...','Opens Calibration Scan...', Gtk.STOCK_OPEN)
    action_open_scan.connect('activate', self.__open_scan)
    action_open_calibration = Gtk.Action('open_calibration','Open Calibration Target...','Opens a calibration target', Gtk.STOCK_OPEN)
    action_open_calibration.connect('activate', self.__open_calibration)
    action_save = Gtk.Action('save','Save Calibration...','Infers and saves the colour transform', Gtk.STOCK_SAVE)
    action_save.connect('activate', self.__save_func)
    action_quit = Gtk.Action('quit','Quit','Terminates this application', Gtk.STOCK_QUIT)
    action_quit.connect('activate', Gtk.main_quit)

    group_file = Gtk.ActionGroup('File')
    group_file.add_action_with_accel(action_file, '')
    group_file.add_action_with_accel(action_open_scan, '<ctrl>o')
    group_file.add_action_with_accel(action_open_calibration, '<ctrl>p')
    group_file.add_action_with_accel(action_save, '<ctrl>s')
    group_file.add_action_with_accel(action_quit, '<ctrl>q')
    
    # The view menu...
    action_view = Gtk.Action("view", "View", None, None)
    action_switch = Gtk.Action('switch','Switch Image Display','Swaps if we are looking at the scan or calibration target', Gtk.STOCK_CONVERT)
    action_switch.connect('activate', self.__switch)
    
    action_rotate = Gtk.Action('rotate','Rotate','Rotates current calibration grid by 90 degrees. Mostly for if you scanned the calibration target upsidedown.', Gtk.STOCK_ORIENTATION_LANDSCAPE)
    action_rotate.connect('activate', self.__rotate)
    
    group_view = Gtk.ActionGroup('View')
    group_view.add_action_with_accel(action_view, '')
    group_view.add_action_with_accel(action_switch, '<ctrl>v')
    group_view.add_action_with_accel(action_rotate, '<ctrl>r')
    
    # String specifiying things...
    UI = """
    <ui>
     <menubar name='menu_bar'>
      <menu action='file'>
       <menuitem action='open_scan'/>
       <menuitem action='open_calibration'/>
       <menuitem action='save'/>
       <separator/>
       <menuitem action='quit'/>
      </menu>
      <menu action='view'>
       <menuitem action="switch"/>
       <menuitem action="rotate"/>
      </menu>
     </menubar>
     <toolbar name='icon_bar'>
      <toolitem action='open_scan'/>
      <toolitem action='save'/>
      <separator/>
      <toolitem action='switch'/>
     </toolbar>
    </ui>
    """
    
    # Finally, create the UI manager...
    uimanager = Gtk.UIManager()
    uimanager.add_ui_from_string(UI)
    uimanager.insert_action_group(group_file)
    uimanager.insert_action_group(group_view)
    
    return uimanager
  
  
  def __on_click(self, x, y):
    if self.scan.visible:
      self.scan_grid.update(x, y)
      
    else:
      self.calibration_grid.update(x, y)
    
    self.viewer.queue_draw()


  def __open_scan(self, widget):
    dialog = Gtk.FileChooserDialog('Select a calibration scan...', self, Gtk.FileChooserAction.OPEN, (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_OPEN, Gtk.ResponseType.OK))
  
    filter_image = Gtk.FileFilter()
    filter_image.set_name('Image files')
    filter_image.add_pattern('*.jpg')
    filter_image.add_pattern('*.png')
    filter_image.add_pattern('*.tiff')
    dialog.add_filter(filter_image)
  
    filter_all = Gtk.FileFilter()
    filter_all.set_name('All files')
    filter_all.add_pattern('*')
    dialog.add_filter(filter_all)
  
    dialog.set_filename(os.path.join(os.path.abspath('.'), '.*'))
  
    response = dialog.run()
    if response==Gtk.ResponseType.OK:
      fn = dialog.get_filename()
      print 'Openning %s...' % fn
      
      self.scan.load(fn)
      self.scan_grid.size = self.scan.get_size()
      
      self.scan.visible = True
      self.scan_grid.visible = True
      self.calibration.visible = False
      self.calibration_grid.visible = False
      
      self.viewer.queue_draw()
  
    dialog.destroy()
  
  
  def __open_calibration(self, widget):
    dialog = Gtk.FileChooserDialog('Select a calibration target...', self, Gtk.FileChooserAction.OPEN, (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_OPEN, Gtk.ResponseType.OK))
  
    filter_image = Gtk.FileFilter()
    filter_image.set_name('Image files')
    filter_image.add_pattern('*.jpg')
    filter_image.add_pattern('*.png')
    filter_image.add_pattern('*.tiff')
    dialog.add_filter(filter_image)
  
    filter_all = Gtk.FileFilter()
    filter_all.set_name('All files')
    filter_all.add_pattern('*')
    dialog.add_filter(filter_all)
  
    dialog.set_filename(os.path.join(os.path.abspath('.'), '.*'))
  
    response = dialog.run()
    if response==Gtk.ResponseType.OK:
      fn = dialog.get_filename()
      print 'Openning %s...' % fn
      
      self.calibration.load(fn)
      self.calibration_grid.size = self.calibration.get_size()
      
      self.scan.visible = False
      self.scan_grid.visible = False
      self.calibration.visible = True
      self.calibration_grid.visible = True
      
      self.viewer.queue_draw()
  
    dialog.destroy()


  def __save_func(self, widget):
    dialog = Gtk.FileChooserDialog('Select a .colour_map file...', self, Gtk.FileChooserAction.SAVE, (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_SAVE, Gtk.ResponseType.OK))
  
    filter_stroke = Gtk.FileFilter()
    filter_stroke.set_name('Colour map files')
    filter_stroke.add_pattern('*.colour_map')
    dialog.add_filter(filter_stroke)
  
    filter_all = Gtk.FileFilter()
    filter_all.set_name('All files')
    filter_all.add_pattern('*')
    dialog.add_filter(filter_all)
  
    response = dialog.run()
    if response==Gtk.ResponseType.OK:
      fn = dialog.get_filename()
      
      if not fn.endswith('.colour_map'):
        fn += '.colour_map'
        
      print 'Saving to %s'%fn
      
      # Calculate the homographies...
      shared = numpy.array([[0,0], [1,0], [1,1], [0,1]], dtype=numpy.float32)
      
      scan_hg = match(shared, self.scan_grid.corners * [self.scan.get_size()[::-1]])
      calibration_hg = match(shared, self.calibration_grid.corners * [self.calibration.get_size()[::-1]])
            
      # Sample the images to get the colour maps...
      tweak = 0.2 # Spacing to leave around edge of grid cell to avoid hitting lines.
      samples = 8 # How many samples to take in each dimension from a grid cell.
      sample_count = 9 * 8 * 8 # Number of grid locations.
      
      col_in = numpy.zeros((sample_count, 3), dtype=numpy.float32)
      col_out = numpy.zeros((sample_count, 3), dtype=numpy.float32)
      
      numpy.seterr(all='raise')
      
      for name, surface, hg, dest in [('scan', self.scan.get_original(), scan_hg, col_in), ('calibration', self.calibration.get_original(), calibration_hg, col_out)]:
        
        # Convert cairo image surface to numpy array, then to image format the sample method takes...
        shape = (surface.get_height(), surface.get_width())
        image = numpy.frombuffer(surface.get_data(), dtype=numpy.uint8).reshape((shape[0], shape[1], 4)).astype(numpy.float32)
        
        image = {'r' : image[:,:,0], 'g' : image[:,:,1], 'b' : image[:,:,2]}
        
        # Loop grid locations, recording mean colour for each in turn...
        for j in xrange(26):
          if j==8 or j==17: continue
          jc = j
          if j>8: jc -= 1
          if j>17: jc -= 1
        
          for i in xrange(26):
            if i==8 or i==17: continue
            ic = i
            if i>8: ic -= 1
            if i>17: ic -= 1
            
            # Calculate the area to sample...
            lowX = (i+tweak) / 26.0
            highX = (i+1.0-tweak) / 26.0
            lowY = (j+tweak) / 26.0
            highY = (j+1.0-tweak) / 26.0
            
            # Convert to a coordinate list...
            xaxis = numpy.linspace(lowX, highX, samples)
            yaxis = numpy.linspace(lowY, highY, samples)
            
            coords = numpy.transpose(numpy.meshgrid(yaxis, xaxis, indexing='ij')).reshape((-1,2))[:,::-1].astype(numpy.float32)
            
            # Apply homography to get coordinates in the image...
            coords = numpy.concatenate((coords, numpy.ones((coords.shape[0], 1), dtype=numpy.float32)), axis=1)
            
            coords = hg.dot(coords.T).T
            
            coords /= coords[:,2,None]
            coords = coords[:,:2]
            
            if name!='calibration':
              print name, jc, ic, '|', coords[coords.shape[0]//2,:]
            
            # Sample...
            patch = sample(image, coords, 1)
            
            # Take the mean and record...
            dest[jc * 24 + ic, 0] = patch['r'].mean()
            dest[jc * 24 + ic, 1] = patch['g'].mean()
            dest[jc * 24 + ic, 2] = patch['b'].mean()
            

      # Save the file...        
      if os.path.exists(fn):
        shutil.copy2(fn, fn+'~')
      
      data = dict()
      data['type'] = ['colour_map.rgb']
      
      data['element'] = dict()
      data['element']['sample'] = OrderedDict()
      
      data['element']['sample']['in.r'] = col_in[:,0]
      data['element']['sample']['in.g'] = col_in[:,1]
      data['element']['sample']['in.b'] = col_in[:,2]
      
      data['element']['sample']['out.r'] = col_out[:,0]
      data['element']['sample']['out.g'] = col_out[:,1]
      data['element']['sample']['out.b'] = col_out[:,2]
      
      ply2.write(fn, data)
  
    dialog.destroy()


  def __switch(self, widget):
    self.scan.visible = not self.scan.visible
    self.scan_grid.visible = self.scan.visible
    
    self.calibration.visible = not self.scan.visible
    self.calibration_grid.visible = self.calibration.visible
    
    self.viewer.queue_draw()
  
  
  def __rotate(self, widget):
    if self.scan.visible:
      self.scan_grid.corners = numpy.roll(self.scan_grid.corners, 1, axis=0)
    
    else:
      self.calibration_grid.corners = numpy.roll(self.calibration_grid.corners, 1, axis=0)
    
    self.viewer.queue_draw()



if __name__=='__main__':
  calibrate = Calibrate()
  calibrate.show_all()

  Gtk.main()
