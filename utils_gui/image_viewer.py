#! /usr/bin/env python

# Copyright (c) 2014, Tom SF Haines
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#  * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



import os.path

from viewer import *
from tile_image import *



class ImageViewer(Gtk.Window):
  def __init__(self):
    # Do the basic window setup...
    Gtk.Window.__init__(self, title='Image Viewer')
    self.connect('delete-event', Gtk.main_quit)
    self.set_default_size(1024, 576)
    
    # Setup the viewing area...
    self.viewer = Viewer()
    self.viewer.set_bg(0.3, 0.3, 0.3)
    
    # The actual image layer, empty...
    self.image = TileImage(None)
    self.viewer.add_layer(self.image)
    
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

    action_open_image = Gtk.Action('open_image', 'Open Image', 'Opens an image...', Gtk.STOCK_OPEN)
    action_open_image.connect('activate', self.__open_image)

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
       <menuitem action='open_image'/>
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


  def __open_image(self, widget):
    dialog = Gtk.FileChooserDialog('Select an image...', self, Gtk.FileChooserAction.OPEN, (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_OPEN, Gtk.ResponseType.OK))

    filter_image = Gtk.FileFilter()
    filter_image.set_name('Image files')
    filter_image.add_pattern('*.jpg')
    filter_image.add_pattern('*.JPG')
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
      self.fn = dialog.get_filename()
      print 'Openning %s...'%self.fn

      self.image.load(self.fn)
      self.viewer.reset_view()
      self.viewer.queue_draw()

      # Report back...
      print 'File(s) loaded'

    dialog.destroy()



if __name__=='__main__':
  iv = ImageViewer()
  iv.show_all()

  Gtk.main()
