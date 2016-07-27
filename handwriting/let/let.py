# Copyright 2016 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import sys
import os.path
import shutil
import numpy
import time
import datetime

from gi.repository import Gtk, Gdk, GdkPixbuf, Pango
import cairo

from line_graph.utils_gui.viewer import Viewer
from line_graph.utils_gui.tile_image import TileImage
from line_graph.utils_gui.tile_mask import TileMask
from line_graph.utils_gui.tile_value import TileValue
from line_graph.utils_gui.reticle_overlay import ReticleOverlay

from line_graph.line_layer import LineLayer
from line_graph.line_overlay_layer import LineOverlayLayer
from rule_layer import RuleLayer


from misc.tps import TPS
import threshold
from threshold_line import ThresholdLine
import skeleton
import line_feat
import infer_alpha
from line_graph import line_graph

from auto_tag import AutoTagDialog

from ply2 import ply2



class LET(Gtk.Window):
  def __init__(self):
    # Do the basic window setup...
    Gtk.Window.__init__(self, title='Line Extraction & Tagging')
    self.connect('delete-event', Gtk.main_quit)
    self.set_default_size(1024, 576)
    
    # Misc variables...
    self.fn = None # Filename of the image that has been loaded.

    # Setup the viewing area...
    self.viewer = Viewer()
    self.viewer.set_bg(1.0, 1.0, 1.0)
    self.viewer.set_drag_col(1.0, 0.0, 0.0)
    
    self.viewer.set_on_draw(self.__on_draw)
    self.viewer.set_on_move(self.__on_move)
    self.viewer.set_on_click(self.__on_click)
    self.viewer.set_on_drag(self.__on_drag)
    self.viewer.set_on_paint(self.__on_paint)

    # Storage for the current image - make it something empty until an actual load...
    self.image = TileImage(None)
    self.viewer.add_layer(self.image)

    # State of the thresholding...
    self.threshold = None
    self.threshold_tiles = TileMask(self.threshold)
    self.viewer.add_layer(self.threshold_tiles)
    self.threshold_tiles.set_false(None)
    
    # State of thresholding overrides...
    self.threshold_lock = None
    self.threshold_lock_tiles = TileValue(self.threshold_lock)
    self.viewer.add_layer(self.threshold_lock_tiles)
    self.threshold_lock_changed = False
    
    # Ruled line renderer, so we can put the letters on lines...
    self.ruled = RuleLayer()
    self.viewer.add_layer(self.ruled)
    
    # State of the line extraction...
    self.line = None # LineGraph object.
    self.line_tiles = LineLayer()
    self.viewer.add_layer(self.line_tiles)

    # Overlay to give extra details on the line, including quite a lot of gui...
    self.line_overlay = LineOverlayLayer()
    self.viewer.add_layer(self.line_overlay)
    
    # Add a targetting reticle - makes accurate zooming easier...
    self.reticle = ReticleOverlay()
    self.viewer.add_layer(self.reticle)

    # Fetch the menu and set it up - this is mostly exported to another method because it gets messy...
    self.uimanager = self.__create_uimanager()

    self.add_accel_group(self.uimanager.get_accel_group())

    menu_bar = self.uimanager.get_widget('/menu_bar')
    icon_bar = self.uimanager.get_widget('/icon_bar')
    
    # The text box for editing tags...
    self.tag = Gtk.Entry()
    if 'bigtags' in sys.argv:
      self.tag.modify_font(Pango.font_description_from_string('monospace 32'))
    self.tag.get_buffer().connect('inserted-text', self.__tag_insert)
    self.tag.get_buffer().connect('deleted-text', self.__tag_delete)
    
    self.segment = None # The segment having its tags edited.

    # Layout the parts...
    self.status = Gtk.Label('-')
    
    vertical = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
    vertical.pack_start(menu_bar, False, False, 0)
    vertical.pack_start(self.viewer, True, True, 0)
    vertical.pack_start(self.tag, False, False, 0)
    vertical.pack_start(self.status, False, False, 0)
    vertical.pack_start(icon_bar, False, False, 0)
    self.add(vertical)
    
    # Variable that contains a thin plate spline that converts colour into density...
    self.density = None
    
    # To avoid a crash if the user tries to edit a blank image...
    self.start_edit = datetime.datetime.utcnow()
    self.alg_time = 0.0
    
    # Parameters for thresholding...
    self.__pen_fineliner()
    
    # Auto model, so its cached between runs...
    self.auto_model = None


  def __create_uimanager(self):
    # Create the actions for the file menu action group...
    action_file = Gtk.Action('file', 'File', None, None)

    action_open_image = Gtk.Action('open_image', 'Open Image', 'Opens an image for processing...', Gtk.STOCK_OPEN)
    action_open_image.connect('activate', self.__open_image)
    
    action_save_lg = Gtk.Action('save_lg', 'Save Line Graph', 'Saves the line-graph, i.e the line and its meta data.', Gtk.STOCK_SAVE)
    action_save_lg.connect('activate', self.__save_lg)
    
    action_save_density = Gtk.Action('save_density', 'Save Directory Density', 'Saves the density model, such that all other files in the same directory will automatically use it.', Gtk.STOCK_EXECUTE)
    action_save_density.connect('activate', self.__save_density)

    action_quit = Gtk.Action('quit', 'Quit', 'Terminates this application', Gtk.STOCK_QUIT)
    action_quit.connect('activate', Gtk.main_quit)

    # Create the file menu action group...
    group_file = Gtk.ActionGroup('File')
    group_file.add_action_with_accel(action_file, '')
    group_file.add_action_with_accel(action_open_image, '<ctrl>o')
    group_file.add_action_with_accel(action_save_lg, '<ctrl>s')
    group_file.add_action_with_accel(action_save_density, '<ctrl>d')
    group_file.add_action_with_accel(action_quit, '<ctrl>q')

    # Create the actions for the view menu action group...
    action_view = Gtk.Action('view', 'View', None, None)

    action_fullscreen = Gtk.ToggleAction('fullscreen', 'Fullscreen', 'Toggles fullscreen mode', Gtk.STOCK_FULLSCREEN)
    action_fullscreen.connect('activate', self.__fullscreen)
    self.action_image_visible = Gtk.ToggleAction('show_image', 'Show Image', 'Toggles if the image is visible or not', Gtk.STOCK_SELECT_COLOR)
    self.action_image_visible.set_active(True)
    self.action_image_visible.connect('activate', self.__image_visible)
    action_interpolate = Gtk.ToggleAction('interpolate', 'Interpolate', 'Toggles if the image is interpolated or not', Gtk.STOCK_REFRESH)
    action_interpolate.set_active(True)
    action_interpolate.connect('activate', self.__interpolate)
    
    action_fit_to_screen = Gtk.Action('fit_to_screen', 'Fit to screen', 'Adjusts the view so that the sctroke fills the entire screen.', Gtk.STOCK_ZOOM_FIT)
    action_fit_to_screen.connect('activate', lambda w: self.viewer.reset_view())
    action_one_to_one = Gtk.Action('one_to_one', 'Reset zoom', 'Resets the zoom to the original recording scale.', Gtk.STOCK_ZOOM_100)
    action_one_to_one.connect('activate', lambda w: self.viewer.one_to_one())
    action_zoom_in = Gtk.Action('zoom_in', 'Zoom In', 'Get closer to the image without moving your head.', Gtk.STOCK_ZOOM_IN)
    action_zoom_in.connect('activate', lambda w: self.viewer.zoom(True))
    action_zoom_out = Gtk.Action('zoom_out', 'Zoom Out', 'Get further away from the image without moving your head.', Gtk.STOCK_ZOOM_OUT)
    action_zoom_out.connect('activate', lambda w: self.viewer.zoom(False))

    action_move_up = Gtk.Action('move_up', 'Move Up', 'Travel north across the stroke', Gtk.STOCK_GO_UP)
    action_move_up.connect('activate', lambda w: self.viewer.move(0, -64))
    action_move_right = Gtk.Action('move_right', 'Move Right', 'Travel east across the stroke', Gtk.STOCK_GO_FORWARD)
    action_move_right.connect('activate', lambda w: self.viewer.move(64, 0))
    action_move_down = Gtk.Action('move_down', 'Move Down', 'Travel south across the stroke', Gtk.STOCK_GO_DOWN)
    action_move_down.connect('activate', lambda w: self.viewer.move(0, 64))
    action_move_left = Gtk.Action('move_left', 'Move Left', 'Travel west across the stroke', Gtk.STOCK_GO_BACK)
    action_move_left.connect('activate', lambda w: self.viewer.move(-64, 0))

    # Create the view menu action group...
    group_view = Gtk.ActionGroup('View')
    group_view.add_action_with_accel(action_view, '')
    group_view.add_action_with_accel(action_fullscreen, 'F11')
    group_view.add_action_with_accel(self.action_image_visible, '<ctrl>v')
    group_view.add_action_with_accel(action_interpolate, '<ctrl>i')
    group_view.add_action_with_accel(action_fit_to_screen, '<ctrl>f')
    group_view.add_action_with_accel(action_one_to_one, '<ctrl>1')
    group_view.add_action_with_accel(action_zoom_in, '<ctrl>Up')
    group_view.add_action_with_accel(action_zoom_out, '<ctrl>Down')
    group_view.add_action_with_accel(action_move_up, '<alt>Up')
    group_view.add_action_with_accel(action_move_right, '<alt>Right')
    group_view.add_action_with_accel(action_move_down, '<alt>Down')
    group_view.add_action_with_accel(action_move_left, '<alt>Left')

    # Create the actions for the line extraction menu...
    action_extraction = Gtk.Action("extraction", "Extraction", None, None)

    self.action_show_threshold = Gtk.ToggleAction('show_threshold', 'Show Threshold','Toggles if the threshold for the image is shown or not', Gtk.STOCK_CUT)
    self.action_show_threshold.connect('activate', self.__threshold_visible)
    
    self.action_show_override = Gtk.ToggleAction('show_override', 'Show Override','Toggles if the override for the image thresholding is shown or not', Gtk.STOCK_DISCONNECT)
    self.action_show_override.connect('activate', self.__override_visible)

    self.action_show_line = Gtk.ToggleAction('show_line', 'Show Line', 'Toggles if the line extracted from the text is shown or not', Gtk.STOCK_SELECT_ALL)
    self.action_show_line.connect('activate', self.__line_visible)
    
    self.action_render_correct = Gtk.RadioAction('render_correct', 'Render Correct Line', 'Makes the line render with the correct width.', Gtk.STOCK_PRINT, LineLayer.RENDER_CORRECT)
    self.action_render_thin = Gtk.RadioAction('render_thin', 'Render Thin Line', 'Makes the line render with a line of width one in viewport space.', Gtk.STOCK_PRINT_REPORT, LineLayer.RENDER_THIN)
    self.action_render_one = Gtk.RadioAction('render_one', 'Render Bare Line', 'Makes the line render with a line of width one in image space.', Gtk.STOCK_PRINT_PREVIEW, LineLayer.RENDER_ONE)
    self.action_render_weird = Gtk.RadioAction('render_weird', 'Render Weird Line', 'Makes the line render with the correct width but ignoring viewport scale.', Gtk.STOCK_PRINT_WARNING, LineLayer.RENDER_WEIRD)
    
    self.action_render_thin.join_group(self.action_render_correct)
    self.action_render_one.join_group(self.action_render_correct)
    self.action_render_weird.join_group(self.action_render_correct)
    self.action_render_thin.set_active(True)
    self.line_tiles.set_mode(LineLayer.RENDER_THIN)
    
    self.action_render_correct.connect('activate', self.__line_render_state)
    self.action_render_thin.connect('activate', self.__line_render_state)
    self.action_render_one.connect('activate', self.__line_render_state)
    self.action_render_weird.connect('activate', self.__line_render_state)
    
    self.action_alpha = Gtk.Action('do_alpha', 'Infer Alpha', 'Infers the alpha of the image using the extracted line, saves it for future use.', Gtk.STOCK_FIND)
    self.action_alpha.connect('activate', self.__alpha)

    # Create the line extraction menu group...
    group_extraction = Gtk.ActionGroup('Extraction')
    group_extraction.add_action_with_accel(action_extraction, '')
    group_extraction.add_action_with_accel(self.action_show_threshold, '<ctrl>t')
    group_extraction.add_action_with_accel(self.action_show_override, '<ctrl><alt>t')
    group_extraction.add_action_with_accel(self.action_show_line, '<ctrl>l')
    
    group_extraction.add_action_with_accel(self.action_render_correct, 'F1')
    group_extraction.add_action_with_accel(self.action_render_thin, 'F2')
    group_extraction.add_action_with_accel(self.action_render_one, 'F3')
    group_extraction.add_action_with_accel(self.action_render_weird, 'F4')
    
    group_extraction.add_action_with_accel(self.action_alpha, '<ctrl>a')
    

    # Create the actions for the algorithms menu...
    action_algorithms = Gtk.Action('algorithms', 'Algorithms', None, None)
    
    action_threshold = Gtk.Action('threshold', 'Threshold', '(Re-)Runs the thresholding algorithm', Gtk.STOCK_APPLY)
    action_threshold.connect('activate', lambda w: self.run_threshold())
    
    action_threshold_config = Gtk.Action('threshold_config', 'Configure Thresholding...', 'Allows you to configure the thresholding system', Gtk.STOCK_EDIT)
    action_threshold_config.connect('activate', self.__configure_thresholding)
    
    action_line = Gtk.Action('line', 'Extract Line', '(Re-)Runs the line extraction algorithm', Gtk.STOCK_APPLY)
    action_line.connect('activate', lambda w: self.make_line())
    
    action_at = Gtk.Action('auto_tag', 'Auto Tag', 'Automatically splits and tags the text', Gtk.STOCK_EXECUTE)
    action_at.connect('activate', self.__auto_tag)
    
    # Create the algorithms menu group...
    group_algorithms = Gtk.ActionGroup('Algorithms')
    group_algorithms.add_action_with_accel(action_algorithms, '')
    group_algorithms.add_action_with_accel(action_threshold, '<alt>t')
    group_algorithms.add_action_with_accel(action_threshold_config, '<ctrl><alt>t')
    group_algorithms.add_action_with_accel(action_line, '<alt>l')
    group_algorithms.add_action_with_accel(action_at, '<alt>a')
    
    
    # Create the actions for the ruled menu...
    action_ruled = Gtk.Action('ruled', 'Ruled', None, None)
    
    self.action_show_ruled = Gtk.ToggleAction('show_ruled', 'Show Rule', 'Toggles if the rule is visible or not', Gtk.STOCK_JUSTIFY_CENTER)
    self.action_show_ruled.set_active(True)
    self.action_show_ruled.connect('activate', self.__ruled_visible)
    
    self.action_reset_ruled = Gtk.Action('reset_ruled', 'Reset Rule', 'Resets the rule, for if it goes pear shaped', Gtk.STOCK_DELETE)
    self.action_reset_ruled.connect('activate', self.__ruled_reset)
    
    # Create the ruled menu group...
    group_ruled = Gtk.ActionGroup('Ruled')
    group_ruled.add_action_with_accel(action_ruled, '')
    group_ruled.add_action_with_accel(self.action_show_ruled, '<ctrl>r')
    group_ruled.add_action_with_accel(self.action_reset_ruled, '<alt>r')
    
    
    # Create the default parameters menu...
    action_defaults = Gtk.Action('defaults', 'Defaults', None, None)
    
    action_pen_biro = Gtk.Action('pen_biro', 'Biro Parameters', 'Sets the parameters of various things to be optimal for a biro', Gtk.STOCK_FLOPPY)
    action_pen_biro.connect('activate', self.__pen_biro)
    
    action_pen_pencil = Gtk.Action('pen_pencil', 'Pencil Parameters', 'Sets the parameters of various things to be optimal for a pencil', Gtk.STOCK_FLOPPY)
    action_pen_pencil.connect('activate', self.__pen_pencil)
    
    action_pen_fountain = Gtk.Action('pen_fountain', 'Fountain Parameters', 'Sets the parameters of various things to be optimal for a fountain pen', Gtk.STOCK_FLOPPY)
    action_pen_fountain.connect('activate', self.__pen_fountain)
    
    action_pen_gel = Gtk.Action('pen_gel', 'Gel Parameters', 'Sets the parameters of various things to be optimal for a gel pen', Gtk.STOCK_FLOPPY)
    action_pen_gel.connect('activate', self.__pen_gel)
    
    action_pen_fineliner = Gtk.Action('pen_fineliner', 'Fineliner Parameters', 'Sets the parameters of various things to be optimal for a fineliner', Gtk.STOCK_FLOPPY)
    action_pen_fineliner.connect('activate', self.__pen_fineliner)
    
    group_defaults = Gtk.ActionGroup('Defaults')
    group_defaults.add_action_with_accel(action_defaults, '')
    group_defaults.add_action_with_accel(action_pen_biro, '')
    group_defaults.add_action_with_accel(action_pen_pencil, '')
    group_defaults.add_action_with_accel(action_pen_fountain, '')
    group_defaults.add_action_with_accel(action_pen_gel, '')
    group_defaults.add_action_with_accel(action_pen_fineliner, '')
    
    
    # Create the actions for the cursor painting mode...
    action_mouse = Gtk.Action('mouse', 'Mouse', None, None)
    
    self.action_mouse_tag = Gtk.RadioAction('mouse_tag', 'Tag mode', 'Cursor allows you to tag the text', Gtk.STOCK_EDIT, -1)
    self.action_mouse_rule = Gtk.RadioAction('mouse_rule', 'Rule mode', 'For updating the rule homography', Gtk.STOCK_PAGE_SETUP, -2)
    self.action_mouse_either = Gtk.RadioAction('mouse_either', 'Set either mode', 'For indicating that the segmentation is unknown; as this is the default its really for removing mistakes.', Gtk.STOCK_CLEAR, 0)
    self.action_mouse_text = Gtk.RadioAction('mouse_text', 'Set text mode', 'For indicating that the area is part of the text', Gtk.STOCK_OK, 2)
    self.action_mouse_bg = Gtk.RadioAction('mouse_bg', 'Set bg mode', 'For indicating that the area is part of the background', Gtk.STOCK_NO, 1)
    
    self.action_mouse_tag.set_active(True)
    self.action_mouse_rule.join_group(self.action_mouse_tag)
    self.action_mouse_either.join_group(self.action_mouse_tag)
    self.action_mouse_text.join_group(self.action_mouse_tag)
    self.action_mouse_bg.join_group(self.action_mouse_tag)
    
    self.action_mouse_tag.connect('activate', self.__mouse_state)
    self.action_mouse_rule.connect('activate', self.__mouse_state)
    self.action_mouse_either.connect('activate', self.__mouse_state)
    self.action_mouse_text.connect('activate', self.__mouse_state)
    self.action_mouse_bg.connect('activate', self.__mouse_state)
    
    # Create the mouse mode menu group...
    group_mouse = Gtk.ActionGroup('Mouse')
    group_mouse.add_action_with_accel(action_mouse, '')
    
    group_mouse.add_action_with_accel(self.action_mouse_tag, 'F5')
    group_mouse.add_action_with_accel(self.action_mouse_rule, 'F6')
    group_mouse.add_action_with_accel(self.action_mouse_either, 'F7')
    group_mouse.add_action_with_accel(self.action_mouse_text, 'F8')
    group_mouse.add_action_with_accel(self.action_mouse_bg, 'F9')


    # Create the UI description string thingy...
    ui = """
    <ui>
     <menubar name='menu_bar'>
      <menu action='file'>
       <menuitem action='open_image'/>
       <menuitem action='save_lg'/>
       <menuitem action='do_alpha'/>
       <separator/>
       <menuitem action='save_density'/>
       <separator/>
       <menuitem action='quit'/>
      </menu>
      <menu action='view'>
       <menuitem action='fullscreen'/>
       <menuitem action='show_image'/>
       <menuitem action='interpolate'/>
       <separator/>
       <menuitem action='fit_to_screen'/>
       <menuitem action='one_to_one'/>
       <menuitem action='zoom_out'/>
       <menuitem action='zoom_in'/>
       <separator/>
       <menuitem action='move_up'/>
       <menuitem action='move_right'/>
       <menuitem action='move_down'/>
       <menuitem action='move_left'/>
      </menu>
      <menu action='mouse'>
       <menuitem action='mouse_tag'/>
       <menuitem action='mouse_rule'/>
       <menuitem action='mouse_either'/>
       <menuitem action='mouse_text'/>
       <menuitem action='mouse_bg'/>
      </menu>
      <menu action='extraction'>
       <menuitem action='show_threshold'/>
       <menuitem action='show_override'/>
       <menuitem action='show_line'/>
       <menuitem action='show_ruled'/>
       <separator/>
       <menuitem action='render_correct'/>
       <menuitem action='render_thin'/>
       <menuitem action='render_one'/>
       <menuitem action='render_weird'/>
      </menu>
      <menu action='algorithms'>
       <menuitem action='threshold'/>
       <menuitem action='threshold_config'/>
       <separator/>
       <menuitem action='line'/>
       <separator/>
       <menuitem action='reset_ruled'/>
       <separator/>
       <menuitem action='auto_tag'/>
      </menu>
      <menu action='defaults'>
       <menuitem action='pen_biro'/>
       <menuitem action='pen_pencil'/>
       <menuitem action='pen_fountain'/>
       <menuitem action='pen_gel'/>
       <menuitem action='pen_fineliner'/>
      </menu>
     </menubar>
     <toolbar name='icon_bar'>
      <toolitem action='open_image'/>
      <toolitem action='save_lg'/>
      <toolitem action='do_alpha'/>
      <separator/>
      <toolitem action='fullscreen'/>
      <toolitem action='interpolate'/>
      <separator/>
      <toolitem action='zoom_out'/>
      <toolitem action='fit_to_screen'/>
      <toolitem action='one_to_one'/>
      <toolitem action='zoom_in'/>
      <separator/>
      <toolitem action='mouse_tag'/>
      <toolitem action='mouse_rule'/>
      <toolitem action='mouse_either'/>
      <toolitem action='mouse_text'/>
      <toolitem action='mouse_bg'/>
      <separator/>
      <toolitem action='show_image'/>
      <toolitem action='show_threshold'/>
      <toolitem action='show_override'/>
      <toolitem action='show_line'/>
      <toolitem action='show_ruled'/>
      <separator/>
      <toolitem action='render_correct'/>
      <toolitem action='render_thin'/>
      <toolitem action='render_one'/>
      <toolitem action='render_weird'/>
      <separator/>
      <toolitem action='auto_tag'/>
     </toolbar>
    </ui>
    """

    # Use the various assets we have created to make and return the manager...
    ret = Gtk.UIManager()
    ret.add_ui_from_string(ui)
    ret.insert_action_group(group_file)
    ret.insert_action_group(group_view)
    ret.insert_action_group(group_extraction)
    ret.insert_action_group(group_algorithms)
    ret.insert_action_group(group_ruled)
    ret.insert_action_group(group_mouse)
    ret.insert_action_group(group_defaults)

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
      if self.fn.endswith('_alpha.png'):
        self.fn = self.fn[:-10] + '.png'
      if self.fn.endswith('_override.png'):
        self.fn = self.fn[:-13] + '.png'
      print 'Openning %s...'%self.fn

      self.image.load(self.fn)
      self.viewer.reset_view()
      self.viewer.queue_draw()

      self.threshold = None
      self.line = None

      # Check if there is an associated line graph, if so load it...
      self.start_edit = datetime.datetime.utcnow()
      self.alg_time = 0.0
      
      lg_fn = os.path.splitext(self.fn)[0] + '.line_graph'
      if os.path.exists(lg_fn):
        print 'Line graph detected - opening %s...'%lg_fn
        f = open(lg_fn, 'r')
        data = ply2.read(f)
        f.close()
        
        if 'meta' not in data:
          data['meta'] = dict()
          
        self.ruled.ply2_load(data)
        
        offset = datetime.timedelta(data['meta']['edit_time.days'] if 'edit_time.days' in data['meta'] else 0, data['meta']['edit_time.seconds'] if 'edit_time.seconds' in data['meta'] else 0)
        self.start_edit -= offset
        
        if 'algorithm_time.seconds' in data['meta']:
          self.alg_time = data['meta']['algorithm_time.seconds']
        
        self.line = line_graph.LineGraph()
        self.line.from_dict(data)
        
        self.action_show_line.set_active(True)
      
      # Setup a default threshold lock...
      img = self.image.get_original()
      self.threshold_lock = numpy.zeros((img.get_height(), img.get_width()), dtype=numpy.uint8)

      # Check if there is a threshold lock to load...
      fn = os.path.splitext(self.fn)[0] + '_override.png'
      if os.path.exists(fn):
        pixbuf = GdkPixbuf.Pixbuf.new_from_file(fn)
        
        image = cairo.ImageSurface(cairo.FORMAT_ARGB32, pixbuf.get_width(), pixbuf.get_height())
    
        ctx = cairo.Context(image)
        Gdk.cairo_set_source_pixbuf(ctx, pixbuf, 0, 0)
        ctx.paint()
        
        override = numpy.fromstring(image.get_data(), dtype=numpy.uint8)
        override = override.reshape((image.get_height(), image.get_width(), -1))
        
        self.threshold_lock[override[:,:,0]>128] = 1
        self.threshold_lock[override[:,:,2]>128] = 2
      
      # Make the the threshold override visible...
      self.threshold_lock_tiles.set_values(self.threshold_lock)
      
      # Synch everything with the states of the toggle switchers...
      self.__threshold_visible(self.action_show_threshold)
      self.__line_visible(self.action_show_line)
      
      # Report back...
      print 'File(s) loaded'

    dialog.destroy()
    
    
  def __save_lg(self, widget):
    if self.fn==None:
      print 'You must first open an image'
      return
    if self.line==None:
      print 'No line graph to save'
      return
      
    fn = os.path.splitext(self.fn)[0] + '.line_graph'
    
    print 'Saving to %s'%fn
    
    if os.path.exists(fn):
      shutil.copy2(fn, fn+'~')
    
    data = self.line.as_dict()
    if 'meta' not in data:
      data['meta'] = dict()

    data['meta']['image'] = os.path.relpath(self.fn, os.path.dirname(fn))
    data['meta']['saved'] = datetime.datetime.utcnow().isoformat(' ')
    
    edit_time = datetime.datetime.utcnow() - self.start_edit
    data['meta']['edit_time.days'] = edit_time.days
    data['meta']['edit_time.seconds'] = edit_time.seconds
    data['meta']['algorithm_time.seconds'] = float(self.alg_time)
    
    self.ruled.ply2_save(data)
    
    ply2.write(fn, data)
    
    fn = os.path.splitext(self.fn)[0] + '_override.png'
    if self.threshold_lock.max()>0 or os.path.exists(fn):
      print 'Saving threshold assist to %s'%fn
      
      if os.path.exists(fn):
        shutil.copy2(fn, fn+'~')
      
      # Convert the threshold lock to colour...
      tl_col = numpy.zeros((self.threshold_lock.shape[0], self.threshold_lock.shape[1], 4), dtype=numpy.uint8)
      
      tl_col[self.threshold_lock==1,0] = 255
      tl_col[self.threshold_lock==2,2] = 255
      tl_col[:,:,3] = 255
      
      # Convert to a cairo surface...
      stride = cairo.ImageSurface.format_stride_for_width(cairo.FORMAT_ARGB32, tl_col.shape[1])
      if stride!=tl_col.strides[0]:
        tl_col = numpy.concatenate((tl_col, numpy.zeros((tl_col.shape[0], (stride-tl_col.strides[0])/4, 4), dtype=numpy.uint8)), axis=1).copy('C')
      temp = cairo.ImageSurface.create_for_data(tl_col, cairo.FORMAT_ARGB32, tl_col.shape[1], tl_col.shape[0], stride)
    
      surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, tl_col.shape[1], tl_col.shape[0])
    
      ctx = cairo.Context(surface)
      ctx.set_operator(cairo.OPERATOR_SOURCE)
      ctx.set_source_surface(temp, 0, 0)
      ctx.paint()
      
      # Save it to the required file...
      surface.write_to_png(fn)
      print 'Saved %s' % fn
    
    print 'Success!'


  def __fullscreen(self, widget):
    """Toggles the window being fullscreen."""
    if widget.get_active():
      self.fullscreen()
    else:
      self.unfullscreen()


  def __interpolate(self, widget):
    """Toggles if the image is interpolated or not."""
    if widget.get_active():
      self.image.set_interpolate(True)
    else:
      self.image.set_interpolate(False)

    self.viewer.queue_draw()


  def __image_visible(self, widget = None):
    iv = self.action_image_visible.get_active()
    lv = self.action_show_line.get_active()
    
    if iv:
      if lv: self.image.set_tint(1.0/3.0)
      else: self.image.set_tint(0.0)
    else: self.image.set_tint(1.0)
    
    self.viewer.queue_draw()
  
  
  def run_threshold(self, dld = False): # dld = don't load density.
    # Get the actual image, so we can work with it...
    start_time = time.clock()
    
    surface = self.image.get_original()
    image = numpy.frombuffer(surface.get_data(),
    dtype=numpy.uint8).reshape((surface.get_height(), surface.get_width(), 4))[:,:,:3]

    # Threshold the image by mean shift on the colours, then classifying modes as bg/fg, followed by graph cuts for regularisation...
    if self.density==None or dld==True:
      fn = os.path.join(os.path.dirname(self.fn), 'density.tps')
      if dld==False and os.path.isfile(fn):
        print 'Loading density model...'
        f = open(fn, 'r')
        data = ply2.read(f)
        f.close()
        
        x = numpy.concatenate((data['element']['sample']['r'].reshape((-1,1)), data['element']['sample']['g'].reshape((-1,1)), data['element']['sample']['b'].reshape((-1,1))), axis=1)
        a = data['element']['sample']['w']
        b = numpy.array([data['element']['plane']['r'][0], data['element']['plane']['g'][0], data['element']['plane']['b'][0], data['element']['plane']['w'][0]], dtype=numpy.float32)
        
        self.density = TPS(3)
        self.density.learn(x, None, a, b)
      else:
        print 'Calculating density model...'
        #model = threshold.cuboid_bg_model(image, 95.0)
        _, self.density = threshold.cluster_colour(image, size=self.threshold_cluster_size, halves=self.threshold_cluster_halves)
      
    print 'Calculating density map...'
    density = line_feat.apply_tps_all(image, self.density)
    
    print 'Regularising density map...'
    density = threshold.density_median(density, 2, 1.0)
    
    if self.alt_threshold:
      print 'Applying alternate threshold model...'
      tl = ThresholdLine()
      self.threshold = tl(1.0-numpy.exp(-32.0*density))
      self.threshold[self.threshold_lock==1] = False
      self.threshold[self.threshold_lock==2] = True
    else:
      print 'Applying threshold model...'
      try:
        #self.threshold = threshold.threshold_reg(image, model)
        self.threshold = threshold.threshold_density(image, density, self.threshold_gc_bg_cost, self.threshold_gc_data_mult, self.threshold_gc_smooth_max, self.threshold_gc_lonely, self.threshold_gc_half_life, self.threshold_lock)
      except MemoryError:
        print 'Out of memory when applying graph cuts - falling back to normal thresholding'
        #self.threshold = threshold.threshold(image, model)
        self.threshold = density > 0.1
      
    # Smooth the threshold, then force the lock...
    if self.threshold_smooth!=0:
      print 'Smoothing...'
      self.threshold = threshold.smooth(self.threshold, self.threshold_smooth)
      self.threshold[self.threshold_lock==1] = False
      self.threshold[self.threshold_lock==2] = True
      
    # As above, but using a more sophisticated approach...
    if self.threshold_smooth_sd!=0 and self.alt_threshold!=True:
      print 'Smoothing signed distance field...'
      self.threshold = threshold.smooth_signed_distance(self.threshold, self.threshold_smooth_sd)
      self.threshold[self.threshold_lock==1] = False
      self.threshold[self.threshold_lock==2] = True
    
    # Terminate islands that are too small - above can leave the occasional small hole...
    if self.threshold_islands!=0:
      print 'Nuking islands...'
      self.threshold = threshold.nuke_islands(self.threshold, self.threshold_islands)
      self.threshold[self.threshold_lock==1] = False
      self.threshold[self.threshold_lock==2] = True
    
    print 'Thresholding complete.'
    
    # Arrange for the threshold to be rendered when needed...
    self.threshold_tiles.set_mask(self.threshold)
    self.viewer.queue_draw()
    
    self.alg_time += time.clock() - start_time


  def __threshold_visible(self, widget):
    """Toggles if the threshold is visible or not - calculates the threshold if needed."""
    if widget.get_active():
      # If needed calculate the threshold...
      if self.threshold==None:
        self.run_threshold()

      # Arrange for its visibility...
      self.threshold_tiles.set_false((0.75,0.75,1.0))
    else:
      # Switch off its visibility...
      self.threshold_tiles.set_false(None)
    
    if self.action_mouse_tag.get_current_value()<0 or widget.get_active():
      self.action_show_override.set_active(widget.get_active())
      
    self.viewer.queue_draw()

  
  def __override_visible(self, widget):
    if widget.get_active():
      self.threshold_lock_tiles.set_colour(1, (0.0, 0.0, 1.0, 0.5)) # Background
      self.threshold_lock_tiles.set_colour(2, (1.0, 0.0, 0.0, 0.5)) # Foreground
    else:
      self.threshold_lock_tiles.set_colour(1)
      self.threshold_lock_tiles.set_colour(2)
      
    self.viewer.queue_draw()


  def make_line(self):
    """Calculate and extract the line for the text that has been loaded in."""
    # We need the threshold...
    if self.threshold==None:
      self.run_threshold()
      
    start_time = time.clock()
      
    # Get the actual image, so we can work with it...
    surface = self.image.get_original()
    image = numpy.frombuffer(surface.get_data(),
    dtype=numpy.uint8).reshape((surface.get_height(), surface.get_width(), 4))[:,:,:3]


    # Use erosion of the threshold to get a line...
    print 'Thinning the threshold to get an initial line...'
    mask = skeleton.zhang_suen(self.threshold)
    mask = skeleton.cull_lonely(mask)
      
    print 'Calculating initial line radius...'
    radius = line_feat.calc_radius(self.threshold, mask)
    print '  |Radius range = [%.1f...%.1f]' % (radius[mask].min(), radius[mask].max())
    
    
    # Refine the line using subspace constrained mean shift...
    if False: #self.density!=None:
      print 'Refining the line...'
      density = line_feat.apply_tps_all(image, self.density)
      rad = numpy.median(radius[mask])
      mask, subspace = skeleton.refine_mask(density, mask, rad)
        
      mask = skeleton.zhang_suen(mask) # Just incase.
      mask = skeleton.cull_lonely(mask)
        
      print 'Recalculating radius...'
      radius = line_feat.calc_radius(self.threshold, mask)
      print '  |Radius range = [%.1f...%.1f]' % (radius[mask].min(), radius[mask].max())
    
    
    if self.density!=None:
      print 'Calculating line density...'
      average = line_feat.calc_average(image, mask, radius)
      density = line_feat.apply_tps(average, mask, self.density)
      density[density<0.0] = 0.0
      density[density>2.0] = 2.0
      print '  |Density range = [%.2f...%.2f]' % (density[mask].min(), density[mask].max())
    else: density = None
    
    
    if self.line!=None:
      print 'Storing current tags...'
      tags = self.line.get_tags()
      splits = self.line.get_splits()
      
      def proc_tag(tag):
        if len(tag)==3:
          return (tag[0], self.line.get_point(tag[1], tag[2])[:2])
        
        else:
          return (tag[0], self.line.get_point(tag[1], tag[2])[:2], self.line.get_point(tag[3], tag[4])[:2])
      tags = [proc_tag(tag) for tag in tags]
      
      def proc_split(split):
        if len(split)==3:
          return (self.line.get_point(split[0], split[1])[:2],)
        
        else:
          return (self.line.get_point(split[0], split[1])[:2], self.line.get_point(split[2], split[3])[:2])
      splits = [proc_split(split) for split in splits]
    
    else:
      tags = []
      splits = []
    
    
    print 'Generating the line graph...'
    self.line = line_graph.LineGraph()
    if density!=None: self.line.from_mask(mask, radius, density)
    else: self.line.from_mask(mask, radius, density)
    self.line.smooth(0.1, 20)
    
    
    if len(tags)!=0 or len(splits)!=0:
      print 'Reapplying tags onto new line...'
      
      for tag in tags:
        if len(tag)==2:
          # Normal tag...
          _, edge, t = self.line.nearest(tag[1][0], tag[1][1])
          self.line.add_tag(edge, t, tag[0])
          
        else:
          # Tagged link - never actually used, but kept just in case...
          _, edge_a, t_a = self.line.nearest(tag[1][0], tag[1][1])
          _, edge_b, t_b = self.line.nearest(tag[2][0], tag[2][1])
          self.line.add_link(edge_a, t_a, edge_b, t_b, tag[0])
      
      for split in splits:
        if len(split)==1:
          _, edge, t = self.line.nearest(split[0][0], split[0][1])
          self.line.add_split(edge, t)
        
        else:
          _, edge_a, t_a = self.line.nearest(split[0][0], split[0][1])
          _, edge_b, t_b = self.line.nearest(split[1][0], split[1][1])
          self.line.add_link(edge_a, t_a, edge_b, t_b)
    
    
    print 'Line extraction complete.'
    self.alg_time += time.clock() - start_time
    
    self.line_tiles.set_line(self.line)
    self.line_overlay.set_line(self.line)


  def __line_visible(self, widget):
    """Toggles if the line is visible or not - calculates it if needed."""
    if widget.get_active():
      # If needed calculate the line...
      if self.line==None:
        self.make_line()

      # Arrange for its visibility...
      self.line_tiles.set_line(self.line)
      self.line_overlay.set_line(self.line)
    else:
      # Switch off its visibility...
      self.line_tiles.set_line(None)
      self.line_overlay.set_line(None)
    
    self.__image_visible()
  
  
  def __line_render_state(self, widget = None):
    """Updates the render state of the line to match the GUI."""
    rs = self.action_render_correct.get_current_value()
    self.line_tiles.set_mode(rs)
    
    self.viewer.queue_draw()
  
  
  def __on_move(self, x, y):
    # Update the closest point to the mouse cursor on the line...
    self.line_overlay.set_mouse(x, y)
    
    try:
      # Update which segment is highlighted, if it has changed...
      segment = self.line_overlay.get_segment()
      if self.segment!=segment:
        # Segment has changed - need to update the tag line...

        if segment==None: self.tag.set_text('')
        else:
          tags = self.line.get_tags(segment)
          self.tag.set_text('|'.join(map(lambda t: t[0], tags)))
      
      self.segment = segment
      
    except RuntimeError:
      self.line.segment()
      self.segment = None
    
    # Update the status bar to show the coordinates in line space...
    if self.viewer.viewport!=None:
      lsx, lsy = self.viewer.viewport.view_to_original(x, y)
      hls = numpy.linalg.inv(self.ruled.homography).dot([lsx, lsy, 1])
      lsx = hls[0] / hls[2]
      lsy = hls[1] / hls[2]
      
      self.status.set_text('(%.2f,%.2f)' % (lsx, lsy))
    
    return True
  
  
  def __on_draw(self):
    if self.threshold_lock_changed:
      self.threshold_lock_tiles.set_values(self.threshold_lock)
      self.threshold_lock_changed = False


  def __on_click(self, x, y):
    """Allows you to click where you want to assign pixels when refining the thresholding."""
    mode = self.action_mouse_tag.get_current_value()
    if mode>=0 and self.action_show_threshold.get_active():
      x, y = self.viewer.get_viewport().view_to_original(x, y)
      x, y = int(x), int(y)
      
      if x>=0 and y>=0 and x<self.threshold_lock.shape[1] and y<self.threshold_lock.shape[0]:
        self.threshold_lock[y,x] = mode
        self.threshold_lock_tiles.set_values(self.threshold_lock)
        self.viewer.queue_draw()


  def __on_drag(self, sx, sy, ex, ey):
    """Either does a split or a merge, depending on the nature of the drag. A merge can either involve killing splits, or it can involve creating a link. A split can involve adding a split or destroying a link."""
    # Convert from screen coordinates to the coordinates of the line graph...
    vp = self.viewer.get_viewport()
    if vp==None: return # Shouldn't happen, but just incase.
    
    sx, sy = vp.view_to_original(sx, sy)
    ex, ey = vp.view_to_original(ex, ey)
    
    mode = self.action_mouse_tag.get_current_value()
    
    if mode==-1: # Tagging mode...
      # No line means do nothing...
      if self.line==None: return
    
      # Get then closest segment at the start and end point...
      _, se_i, se_t = self.line.nearest(sx, sy)
      _, ee_i, ee_t = self.line.nearest(ex, ey)
    
      ss = self.line.get_segment(se_i, se_t)
      es = self.line.get_segment(ee_i, ee_t)
    
      # If its in the same segment then its a split, otherwise a merge...
      if ss==es:
        # Fetch all the intercepts...
        collisions = self.line.intersect(sx, sy, ex, ey)
      
        # Check if we are probably meaning to break a link, and do that instead if need be...
        if len(collisions)==0:
          link_collisions = self.line.intersect_links(sx, sy, ex, ey)
          if len(link_collisions)==1:
            self.line.rem(link_collisions[0][1], link_collisions[0][2])
            print 'split: Break link between edges %i and %i' % (link_collisions[0][1], link_collisions[0][3])
            self.viewer.queue_draw()
            return
        
        # If we have just one intercept then do the split, otherwise its ambiguous so cancel it...
        if len(collisions)==1:  
          # Simple split...
          self.line.add_split(collisions[0][0], collisions[0][1])
          print 'split: edge %i at position %.3f' % (collisions[0][0], collisions[0][1])
        elif len(collisions)==0:
          print 'split: cancelled as no intercept'
        else:
          print 'split: cancelled due to ambiguity'
      else:
        count = self.line.merge(ss, es)
        if count!=0:
          print 'merge: Dissolved %i splits' % count
        else:
          self.line.add_link(se_i, se_t, ee_i, ee_t)
          print 'merge: Created a link from edge %i to edge %i'%(se_i, ee_i)

      self.viewer.queue_draw()
    
    elif mode==-2: # Rule edit mode
      self.ruled.drag(sx, sy, ex, ey)
      self.viewer.queue_draw()


  def __on_paint(self, sx, sy, ex, ey):
    mode = self.action_mouse_tag.get_current_value()
    if mode>=0:
      sx, sy = self.viewer.get_viewport().view_to_original(sx, sy)
      ex, ey = self.viewer.get_viewport().view_to_original(ex, ey)
      
      steps = int(numpy.ceil(max(numpy.fabs(ex-sx), numpy.fabs(ey-sy))))
      if steps<1: steps = 1
      dx = (ex - sx) / steps
      dy = (ey - sy) / steps
      
      px = int(sx)
      py = int(sy)
      preferX = numpy.fabs(ex-sx) > numpy.fabs(ey-sy)
      
      change = False
      for s in xrange(steps+1):
        x = int(sx + dx*s)
        y = int(sy + dy*s)
      
        if x>=0 and y>=0 and x<self.threshold_lock.shape[1] and y<self.threshold_lock.shape[0] and self.threshold_lock[y,x]!=mode:
          self.threshold_lock[y,x] = mode
          change = True
        
        if px!=x and py!=y:
          # Diagonal move - break it with an extra pixel...
          if preferX:
            if x>=0 and py>=0 and x<self.threshold_lock.shape[1] and py<self.threshold_lock.shape[0] and self.threshold_lock[py,x]!=mode:
              self.threshold_lock[py,x] = mode
              change = True
          else:
            if px>=0 and y>=0 and px<self.threshold_lock.shape[1] and y<self.threshold_lock.shape[0] and self.threshold_lock[y,px]!=mode:
              self.threshold_lock[y,px] = mode
              change = True
        
        px = x
        py = y
          
      if change:
        self.threshold_lock_changed = True
        return True
    return False


  def __tag_insert(self, widget, pos, text, count):
    edge, t = self.line_overlay.get_edge()
    segment = self.line_overlay.get_segment()
    
    if edge!=None and self.segment!=None and self.segment==segment:
      new_tags = map(lambda s: s.strip(), widget.get_text().split('|'))
      old_tags = self.line.get_tags(self.segment)
      
      # Remove tags that are no longer there...
      for tag in old_tags:
        if tag[0] not in new_tags:
          self.line.rem(tag[1], tag[2])
      
      # Create a list of tags that are there...
      old_tags = map(lambda t: t[0], old_tags)
      
      # Add tags that are new...
      for tag in filter(lambda t: t not in old_tags, new_tags):
        if tag!='': self.line.add_tag(edge, t, tag)
      
      # Segmentation is now invalid - need to prevent bad stuff occuring (This is horribly inefficient)...
      self.segment = self.line.get_segment(edge, t)
      self.line_overlay.set_segment(self.segment)
  
  
  def __tag_delete(self, widget, pos, count):
    self.__tag_insert(widget, pos, None, count)
  
  
  def __ruled_visible(self, widget):
    self.ruled.set_visible(widget.get_active())
    self.viewer.queue_draw()
  
  def __ruled_reset(self, widget):
    self.ruled.reset()
    self.viewer.queue_draw()
  
  
  def __alpha(self, widget):
    """Infers the alpha of the image and saves it out, if needed, making use of the thresholding (Anything outside has alpha 0!)"""
    # Only bother if the alpha file doesn't already exist...
    fn = os.path.splitext(self.fn)[0] + '_alpha.png'
    if not os.path.exists(fn):
      # We need the threshold...
      if self.threshold==None:
        self.run_threshold()
      
      start_time = time.clock()
      
      # Get the actual image, so we can work with it...
      surface = self.image.get_original()
      image = numpy.frombuffer(surface.get_data(),
      dtype=numpy.uint8).reshape((surface.get_height(), surface.get_width(), 4))[:,:,:3]
      
      # For safeties sake dilate the threshold, so the inpainting doesn't risk being distorted by slight smudges at the edge...
      enlarged = threshold.dilate(self.threshold, 8)
      
      # Calculate the alpha...
      iwa = infer_alpha.infer_alpha_cc(image, enlarged)
      
      # Convert to a cairo surface...
      stride = cairo.ImageSurface.format_stride_for_width(cairo.FORMAT_ARGB32, iwa.shape[1])
      if stride!=image.strides[0]:
        iwa = numpy.concatenate((iwa, numpy.zeros((iwa.shape[0], (stride-iwa.strides[0])/4, 4), dtype=numpy.uint8)), axis=1).copy('C')
      temp = cairo.ImageSurface.create_for_data(iwa, cairo.FORMAT_ARGB32, iwa.shape[1], iwa.shape[0], stride)
    
      surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, iwa.shape[1], iwa.shape[0])
    
      ctx = cairo.Context(surface)
      ctx.set_operator(cairo.OPERATOR_SOURCE)
      ctx.set_source_surface(temp, 0, 0)
      ctx.paint()
      
      # Save it to the required file...
      surface.write_to_png(fn)
      print 'Saved %s' % fn
      
      self.alg_time += time.clock() - start_time
  
  
  def __configure_thresholding(self, widget):
    """Creates and runs a dialog for configuring the parameters of the thresholding"""
    dialog = Gtk.Dialog('Thresholding Parameters')
    
    dialog.add_button('Cancel', Gtk.ResponseType.CANCEL)
    dialog.add_button('Accept', Gtk.ResponseType.OK)

    cluster_size = Gtk.SpinButton()
    cluster_size.set_adjustment(Gtk.Adjustment(self.threshold_cluster_size, 0.1, 129.0, 0.1, 1.0, 1.0))
    cluster_size.set_digits(1)
    
    cluster_halves = Gtk.SpinButton()
    cluster_halves.set_adjustment(Gtk.Adjustment(self.threshold_cluster_halves, 0, 5, 1, 1, 1))
    cluster_halves.set_digits(1)
    
    gc_bg_cost = Gtk.SpinButton()
    gc_bg_cost.set_adjustment(Gtk.Adjustment(self.threshold_gc_bg_cost, 0.0, 129.0, 0.5, 0.5, 1.0))
    gc_bg_cost.set_digits(2)
    
    gc_data_mult = Gtk.SpinButton()
    gc_data_mult.set_adjustment(Gtk.Adjustment(self.threshold_gc_data_mult, 0.0, 129.0, 0.5, 0.5, 1.0))
    gc_data_mult.set_digits(2)
    
    gc_smooth_max = Gtk.SpinButton()
    gc_smooth_max.set_adjustment(Gtk.Adjustment(self.threshold_gc_smooth_max, 0.0, 129.0, 0.5, 0.5, 1.0))
    gc_smooth_max.set_digits(2)
    
    gc_lonely = Gtk.SpinButton()
    gc_lonely.set_adjustment(Gtk.Adjustment(self.threshold_gc_lonely, 0.0, 129.0, 0.5, 0.5, 1.0))
    gc_lonely.set_digits(2)
    
    gc_half_life = Gtk.SpinButton()
    gc_half_life.set_adjustment(Gtk.Adjustment(self.threshold_gc_half_life, 0.0, 129.0, 0.5, 0.5, 1.0))
    gc_half_life.set_digits(2)
    
    smooth = Gtk.SpinButton()
    smooth.set_adjustment(Gtk.Adjustment(self.threshold_smooth, 0, 9, 1, 1, 1))
    smooth.set_digits(1)
    
    smooth_sd = Gtk.SpinButton()
    smooth_sd.set_adjustment(Gtk.Adjustment(self.threshold_smooth_sd, 0, 17, 1, 1, 1))
    smooth_sd.set_digits(1)
    
    inuke = Gtk.SpinButton()
    inuke.set_adjustment(Gtk.Adjustment(self.threshold_islands, 0, 65, 1, 1, 1))
    inuke.set_digits(1)
    
    box = dialog.get_content_area()
    
    l = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
    l.pack_start(Gtk.Label('Cluster Size:'), False, False, 8)
    l.pack_start(cluster_size, True, True, 0)
    box.pack_start(l, False, False, 0)
    
    l = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
    l.pack_start(Gtk.Label('Cluster Halves:'), False, False, 8)
    l.pack_start(cluster_halves, True, True, 0)
    box.pack_start(l, False, False, 0)
    
    l = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
    l.pack_start(Gtk.Label('Background Cost:'), False, False, 8)
    l.pack_start(gc_bg_cost, True, True, 0)
    box.pack_start(l, False, False, 0)
    
    l = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
    l.pack_start(Gtk.Label('Density Multiplier:'), False, False, 8)
    l.pack_start(gc_data_mult, True, True, 0)
    box.pack_start(l, False, False, 0)
    
    l = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
    l.pack_start(Gtk.Label('Smooth Half Life:'), False, False, 8)
    l.pack_start(gc_half_life, True, True, 0)
    box.pack_start(l, False, False, 0)
    
    l = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
    l.pack_start(Gtk.Label('Smoothing Cap:'), False, False, 8)
    l.pack_start(gc_smooth_max, True, True, 0)
    box.pack_start(l, False, False, 0)
    
    l = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
    l.pack_start(Gtk.Label('Noise Cost:'), False, False, 8)
    l.pack_start(gc_lonely, True, True, 0)
    box.pack_start(l, False, False, 0)
    
    l = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
    l.pack_start(Gtk.Label('Post-smooth:'), False, False, 8)
    l.pack_start(smooth, True, True, 0)
    box.pack_start(l, False, False, 0)
    
    l = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
    l.pack_start(Gtk.Label('Post-smooth signed distance:'), False, False, 8)
    l.pack_start(smooth_sd, True, True, 0)
    box.pack_start(l, False, False, 0)
    
    l = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
    l.pack_start(Gtk.Label('Island nuke limit:'), False, False, 8)
    l.pack_start(inuke, True, True, 0)
    box.pack_start(l, False, False, 0)
    
    box.show_all()
    
    response = dialog.run()
    if response==Gtk.ResponseType.OK:
      self.threshold_cluster_size = cluster_size.get_value()
      self.threshold_cluster_halves = cluster_halves.get_value_as_int()
    
      self.threshold_gc_bg_cost = gc_bg_cost.get_value()
      self.threshold_gc_data_mult = gc_data_mult.get_value()
      self.threshold_gc_smooth_max = gc_smooth_max.get_value()
      self.threshold_gc_lonely = gc_lonely.get_value()
      self.threshold_gc_half_life = gc_half_life.get_value()
    
      self.threshold_smooth = smooth.get_value_as_int()
      self.threshold_smooth_sd = smooth_sd.get_value_as_int()
      self.threshold_islands = inuke.get_value_as_int()
    
    dialog.destroy()
  
  
  def __mouse_state(self, widget):
    mode = self.action_mouse_tag.get_current_value()
    
    if mode==-1:
      self.ruled.set_grid(False)
      self.viewer.set_drag_col(1.0, 0.0, 0.0)
    elif mode==-2:
      self.ruled.set_grid(True)
      self.viewer.set_drag_col(0.75, 0.5, 0.0)
    else:
      self.ruled.set_grid(False)
      self.viewer.set_drag_col(None)

    self.viewer.queue_draw()


  def __save_density(self, widget):
    """Saves the density to use for the directory, into a standard file that is automatically loaded when needed."""
    # Density is calculated during the thresholding - run it...
    self.run_threshold(True)
      
    # Extract the info...
    x = self.density.get_x()
    a = self.density.get_a()
    b = self.density.get_b()

    # Work out filename, handle backup...
    fn = os.path.join(os.path.dirname(self.fn), 'density.tps')
    if os.path.exists(fn):
      shutil.copy2(fn, fn+'~')

    # Save the ply file containing a thin plate spline (should have used ply2 library, but can't be arsed to change it now!)...
    print 'Saving density file...'
    f = open(fn, 'w')
    
    f.write('ply\n')
    f.write('format ascii 2.0\n')
    f.write('type rgb_to_density\n')
    
    f.write('element sample %i\n'%x.shape[0])
    f.write('property real32 r\n')
    f.write('property real32 g\n')
    f.write('property real32 b\n')
    f.write('property real32 w\n')
    
    f.write('element plane 1\n')
    f.write('property real32 r\n')
    f.write('property real32 g\n')
    f.write('property real32 b\n')
    f.write('property real32 w\n')
    
    f.write('end_header\n')
    
    for i in xrange(x.shape[0]):
      f.write('%f %f %f %f\n' % (x[i,0], x[i,1], x[i,2], a[i]))
    
    f.write('%f %f %f %f\n' % (b[0], b[1], b[2], b[3]))
    
    f.close()
    print 'Done.'


  def __auto_tag(self, widget):
    at = AutoTagDialog(self)
    at.run()
    at.destroy()
  
  
  def __pen_biro(self, widget = None): # Could be better, but not aweful.
    self.alt_threshold = False # Not working any better than default:-(
    
    self.threshold_cluster_size = 16.0
    self.threshold_cluster_halves = 2
    
    self.threshold_gc_bg_cost = 4.0
    self.threshold_gc_data_mult = 16.0
    self.threshold_gc_smooth_max = 24.0
    self.threshold_gc_lonely = 0.0
    self.threshold_gc_half_life = 32.0
    
    self.threshold_smooth = 0
    self.threshold_smooth_sd = 2
    self.threshold_islands = 6


  def __pen_pencil(self, widget = None): # Not aweful.
    self.alt_threshold = False # Not working any better than default:-(
    
    self.threshold_cluster_size = 16.0
    self.threshold_cluster_halves = 2
    
    self.threshold_gc_bg_cost = 8.0
    self.threshold_gc_data_mult = 12.0
    self.threshold_gc_smooth_max = 64.0
    self.threshold_gc_lonely = 4.0
    self.threshold_gc_half_life = 128.0
    
    self.threshold_smooth = 1
    self.threshold_smooth_sd = 1
    self.threshold_islands = 24


  def __pen_fountain(self, widget = None): # Acceptable.
    self.alt_threshold = False # Not working any better than default:-(
    
    self.threshold_cluster_size = 16.0
    self.threshold_cluster_halves = 2
    
    self.threshold_gc_bg_cost = 4.0
    self.threshold_gc_data_mult = 16.0
    self.threshold_gc_smooth_max = 24.0
    self.threshold_gc_lonely = 0.0
    self.threshold_gc_half_life = 32.0
    
    self.threshold_smooth = 0
    self.threshold_smooth_sd = 2
    self.threshold_islands = 6
    
    
  def __pen_gel(self, widget = None): # Awful.
    self.alt_threshold = False # Not working any better than default:-(
    
    self.threshold_cluster_size = 16.0
    self.threshold_cluster_halves = 2
    
    self.threshold_gc_bg_cost = 4.0
    self.threshold_gc_data_mult = 16.0
    self.threshold_gc_smooth_max = 24.0
    self.threshold_gc_lonely = 0.0
    self.threshold_gc_half_life = 32.0
    
    self.threshold_smooth = 0
    self.threshold_smooth_sd = 2
    self.threshold_islands = 6
    
    
  def __pen_fineliner(self, widget = None): # Good.
    self.alt_threshold = False # Not working any better than default:-(
    
    self.threshold_cluster_size = 16.0
    self.threshold_cluster_halves = 2
    
    self.threshold_gc_bg_cost = 8.0
    self.threshold_gc_data_mult = 12.0
    self.threshold_gc_smooth_max = 64.0
    self.threshold_gc_lonely = 4.0
    self.threshold_gc_half_life = 128.0
    
    self.threshold_smooth = 1
    self.threshold_smooth_sd = 1
    self.threshold_islands = 24
