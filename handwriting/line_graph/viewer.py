#! /usr/bin/env python

# Copyright 2016 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import sys
import os.path

from line_graph import LineGraph

from ply2 import ply2

from utils_gui.viewer import *
from line_layer import LineLayer
from line_overlay_layer import LineOverlayLayer



class LineGraphViewer(Gtk.Window):
  def __init__(self):
    # Do the basic window setup...
    Gtk.Window.__init__(self, title='Image Viewer')
    self.connect('delete-event', Gtk.main_quit)
    self.set_default_size(1024, 576)
    
    # Setup the viewing area...
    self.viewer = Viewer()
    self.viewer.set_bg(1.0, 1.0, 1.0)
    
    self.viewer.set_on_move(self.__on_move)
    
    # The line graph layers...
    self.line = LineLayer()
    self.viewer.add_layer(self.line)
    
    self.overlay = LineOverlayLayer()
    self.viewer.add_layer(self.overlay)
    
    # If a file name is provied load the line graph...
    if len(sys.argv)>1:
      lg = LineGraph()
      lg.from_dict(ply2.read(sys.argv[1]))
      
      self.line.set_line(lg)
      self.overlay.set_line(lg)
      
      self.viewer.reset_view()
      self.viewer.queue_draw()
    
    # Fetch the menu and set it up - this is mostly exported to another method because it gets messy...
    self.uimanager = self.__create_uimanager()
    self.add_accel_group(self.uimanager.get_accel_group())

    menu_bar = self.uimanager.get_widget('/menu_bar')
    
    # Layout the parts...
    vertical = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
    vertical.pack_start(menu_bar, False, False, 0)
    vertical.pack_start(self.viewer, True, True, 0)
    self.add(vertical)
  
  
  def __create_uimanager(self):
    # Create the actions for the file menu action group...
    action_file = Gtk.Action('file', 'File', None, None)

    action_open_image = Gtk.Action('open_lg', 'Open Line Graph', 'Opens a line graph...', Gtk.STOCK_OPEN)
    action_open_image.connect('activate', self.__open_lg)

    action_quit = Gtk.Action('quit', 'Quit', 'Terminates this application', Gtk.STOCK_QUIT)
    action_quit.connect('activate', Gtk.main_quit)

    # Create the file menu action group...
    group_file = Gtk.ActionGroup('File')
    group_file.add_action_with_accel(action_file, '')
    group_file.add_action_with_accel(action_open_image, '<ctrl>o')
    group_file.add_action_with_accel(action_quit, '<ctrl>q')
    
    # Create the UI description string thingy...
    ui = """
    <ui>
     <menubar name='menu_bar'>
      <menu action='file'>
       <menuitem action='open_lg'/>
       <separator/>
       <menuitem action='quit'/>
      </menu>
     </menubar>
    </ui>
    """

    # Use the various assets we have created to make and return the manager...
    ret = Gtk.UIManager()
    ret.add_ui_from_string(ui)
    ret.insert_action_group(group_file)

    return ret
  
  
  def __open_lg(self, widget):
    dialog = Gtk.FileChooserDialog('Select a line graph...', self, Gtk.FileChooserAction.OPEN, (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_OPEN, Gtk.ResponseType.OK))

    filter_image = Gtk.FileFilter()
    filter_image.set_name('Line graph')
    filter_image.add_pattern('*.line_graph')
    dialog.add_filter(filter_image)

    filter_all = Gtk.FileFilter()
    filter_all.set_name('All files')
    filter_all.add_pattern('*')
    dialog.add_filter(filter_all)

    dialog.set_filename(os.path.join(os.path.abspath('.'), '.*'))

    response = dialog.run()
    if response==Gtk.ResponseType.OK:
      fn = dialog.get_filename()
      print 'Openning %s...'%fn

      lg = LineGraph()
      lg.from_dict(ply2.read(fn))
      
      self.line.set_line(lg)
      self.overlay.set_line(lg)
      
      self.viewer.reset_view()
      self.viewer.queue_draw()

      # Report back...
      print 'File(s) loaded'

    dialog.destroy()
  
  
  def __on_move(self, x, y):
    # Update the closest point to the mouse cursor on the line...
    self.overlay.set_mouse(x, y)
    self.viewer.queue_draw()



if __name__=='__main__':
  lgv = LineGraphViewer()
  lgv.show_all()

  Gtk.main()
