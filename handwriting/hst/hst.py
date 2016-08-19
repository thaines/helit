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

import os.path
import shutil

from gi.repository import Gtk, Gdk

from line_graph.utils_gui.viewer import Viewer
from line_graph.utils_gui.tile_image import TileImage
from line_graph.line_layer import LineLayer

from glyph_db import GlyphDB, Glyph
from chunk_db import ChunkDB
from generate import *

from texture_cache import TextureCache



class HST(Gtk.Window):
  def __init__(self):
    # Do the basic window setup...
    Gtk.Window.__init__(self, title='Handwriting Synthesis Tool')
    self.connect('delete-event', Gtk.main_quit)
    self.set_default_size(1024, 576)
    
    # Create the menus/actions...
    self.uimanager = self.__create_uimanager()

    self.add_accel_group(self.uimanager.get_accel_group())

    menu_bar = self.uimanager.get_widget('/menu_bar')
    icon_bar = self.uimanager.get_widget('/icon_bar')
    
    # Create the two panes, with the generate button in the middle...
    self.notebook = Gtk.Notebook()
    self.right = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
    generate = Gtk.Button('Generate')
    generate.get_child().set_angle(90)
    generate.connect('clicked', self.__generate)
    
    horizontal = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
    horizontal.pack_start(self.notebook, False, False, 0)
    horizontal.pack_start(generate, False, False, 0)
    horizontal.pack_start(self.right, True, True, 0)
    
    vertical = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
    vertical.pack_start(menu_bar, False, False, 0)
    vertical.pack_start(horizontal, True, True, 0)
    self.add(vertical)
    
    
    # Put lots of tabs on the left...
    self.samples = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
    self.notebook.append_page(self.samples, Gtk.Label('Examples'))
    
    self.style = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
    self.notebook.append_page(self.style, Gtk.Label('Style'))
    
    self.stats = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
    self.notebook.append_page(self.stats, Gtk.Label('Stats'))
    
    self.text_view = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
    self.notebook.append_page(self.text_view, Gtk.Label('Text'))
    
    self.selection = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
    self.notebook.append_page(self.selection, Gtk.Label('Selection'))
    
    self.spacing = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
    self.notebook.append_page(self.spacing, Gtk.Label('Spacing'))
    
    self.chunks = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
    self.notebook.append_page(self.chunks, Gtk.Label('Chunks'))
   
    self.rendering = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
    self.notebook.append_page(self.rendering, Gtk.Label('Rendering'))
    
    self.glyph = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
    self.notebook.append_page(self.glyph, Gtk.Label('Glyph'))
    
    self.notebook.connect('switch-page', self.__tab_change)


    # Put the output viewer on the right...
    self.viewer = Viewer()
    self.viewer.set_bg(1.0, 1.0, 1.0)
    self.viewer.set_on_move(self.__on_move)
    self.right.pack_start(self.viewer, True, True, 0)
    

    self.image = TileImage(None)
    self.viewer.add_layer(self.image)
    
    self.line = None
    self.line_tiles = LineLayer()
    self.viewer.add_layer(self.line_tiles)
    
    # A status bar, so we know what the mouse cursor is near...
    self.status = Gtk.Label('-')
    self.right.pack_start(self.status, False, False, 0)
    self.right.pack_start(icon_bar, False, False, 0)


    # Setup the samples tab...
    self.glyph_db = GlyphDB()
    
    self.add_glyphs = Gtk.Button('Load Line Graph...')
    self.samples.pack_start(self.add_glyphs, False, False, 0)
    self.add_glyphs.connect('clicked', self.__add_glyphs)
    
    self.file_view_scroll = Gtk.ScrolledWindow()
    self.file_view_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
    self.samples.pack_start(self.file_view_scroll, True, True, 0)
    
    self.file_list = Gtk.ListStore(str, int)
    
    self.file_view = Gtk.TreeView(self.file_list)
    self.file_view_scroll.add(self.file_view)
    
    render_fn = Gtk.CellRendererText()
    column_fn = Gtk.TreeViewColumn('Filename', render_fn, text=0)
    self.file_view.append_column(column_fn)
    
    render_count = Gtk.CellRendererText()
    column_count = Gtk.TreeViewColumn('Glyphs', render_count, text=1)
    self.file_view.append_column(column_count)
    
    self.rem_glyphs = Gtk.Button('Remove Selected')
    self.samples.pack_start(self.rem_glyphs, False, False, 0)
    self.rem_glyphs.connect('clicked', self.__rem_glyphs)
    
    
    # Setup the style tab...
    self.chunk_db = ChunkDB()
    
    self.add_chunks = Gtk.Button('Load Line Graph...')
    self.style.pack_start(self.add_chunks, False, False, 0)
    self.add_chunks.connect('clicked', self.__add_chunks)
    
    self.chunk_view_scroll = Gtk.ScrolledWindow()
    self.chunk_view_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
    self.style.pack_start(self.chunk_view_scroll, True, True, 0)
    
    self.chunk_list = Gtk.ListStore(str, int)
    
    self.chunk_view = Gtk.TreeView(self.chunk_list)
    self.chunk_view_scroll.add(self.chunk_view)
    
    render_fn = Gtk.CellRendererText()
    column_fn = Gtk.TreeViewColumn('Filename', render_fn, text=0)
    self.chunk_view.append_column(column_fn)
    
    render_count = Gtk.CellRendererText()
    column_count = Gtk.TreeViewColumn('Chunks', render_count, text=1)
    self.chunk_view.append_column(column_count)
    
    self.rem_chunks = Gtk.Button('Remove Selected')
    self.style.pack_start(self.rem_chunks, False, False, 0)
    self.rem_chunks.connect('clicked', self.__rem_chunks)
    
    
    # Setup the sample statistics tab...
    self.stats_scroll = Gtk.ScrolledWindow()
    self.stats_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
    self.stats.pack_start(self.stats_scroll, True, True, 0)
    self.stats_text = Gtk.TextView()
    self.stats_scroll.add(self.stats_text)
    self.stats_text.set_editable(False)
    self.stats_text.set_cursor_visible(False)
    
    
    # Setup the text entry tab...
    self.text = Gtk.TextView()
    self.text_view.pack_start(self.text, True, True, 0)
    self.text.get_buffer().set_text('Jack quietly moved\nup front and seized\nthe big ball of wax.')


    # Setup the character selection tab...
    sel_vert = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
    self.selection.pack_start(sel_vert, True, True, 0)
    
    self.sel_alg = Gtk.ComboBoxText()
    self.sel_alg.append_text('random')
    self.sel_alg.append_text('smart random')
    self.sel_alg.append_text('dynamic programming')
    self.sel_alg.append_text('perturbed dynamic programming')
    self.sel_alg.set_active(3)
    
    l = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
    l.pack_start(Gtk.Label('Algorithm:'), False, False, 8)
    sel_vert.pack_start(l, False, False, 0)
    sel_vert.pack_start(self.sel_alg, False, False, 0)
    
    sel_vert.pack_start(Gtk.Separator(), False, False, 8)
    
    self.fetch_count = Gtk.SpinButton()
    self.fetch_count.set_adjustment(Gtk.Adjustment(16, 1, 32, 1, 1, 1))
    self.fetch_count.set_digits(1)
    
    l = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
    l.pack_start(Gtk.Label('Glyph selection set size:'), False, False, 8)
    sel_vert.pack_start(l, False, False, 0)
    sel_vert.pack_start(self.fetch_count, False, False, 0)
    
    sel_vert.pack_start(Gtk.Separator(), False, False, 8)
    
    self.match_strength = Gtk.SpinButton()
    self.wrong_place_cost = Gtk.SpinButton()
    self.space_mult = Gtk.SpinButton()
    
    self.match_strength.set_adjustment(Gtk.Adjustment(8.0, 0.1, 128.0, 0.1, 1.0, 4.0))
    self.wrong_place_cost.set_adjustment(Gtk.Adjustment(4.0, 0.0, 128.0, 0.1, 1.0, 4.0))
    self.space_mult.set_adjustment(Gtk.Adjustment(0.5, 0.0, 1.0, 0.1, 0.5, 0.5))
    
    self.match_strength.set_digits(1)
    self.wrong_place_cost.set_digits(1)
    self.space_mult.set_digits(1)
    
    l = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
    l.pack_start(Gtk.Label('Cost function strength:'), False, False, 8)
    sel_vert.pack_start(l, False, False, 0)
    sel_vert.pack_start(self.match_strength, False, False, 0)
    
    l = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
    l.pack_start(Gtk.Label('Wrong place cost:'), False, False, 8)
    sel_vert.pack_start(l, False, False, 0)
    sel_vert.pack_start(self.wrong_place_cost, False, False, 0)
    
    l = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
    l.pack_start(Gtk.Label('Space multiplier:'), False, False, 8)
    sel_vert.pack_start(l, False, False, 0)
    sel_vert.pack_start(self.space_mult, False, False, 0)
    
    sel_vert.pack_start(Gtk.Separator(), False, False, 8)
    
    self.randomness = Gtk.SpinButton()
    self.randomness.set_adjustment(Gtk.Adjustment(0.1, 0.01, 1.0, 0.01, 0.1, 0.1))
    self.randomness.set_digits(2)
    
    l = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
    l.pack_start(Gtk.Label('Randomness:'), False, False, 8)
    sel_vert.pack_start(l, False, False, 0)
    sel_vert.pack_start(self.randomness, False, False, 0)
    
    self.transfer_adj_cost = Gtk.CheckButton('Transfer adjacency costs from joined up')
    self.transfer_adj_cost.set_active(True)
    sel_vert.pack_start(self.transfer_adj_cost, False, False, 0)
    
    
    # Setup the spacing tab...
    space_vert = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
    self.spacing.pack_start(space_vert, True, True, 0)
    
    self.space_alg = Gtk.ComboBoxText()
    self.space_alg.append_text('user set')
    self.space_alg.append_text('average of original')
    self.space_alg.append_text('weighted median')
    self.space_alg.append_text('weighted central draw')
    self.space_alg.set_active(3)
    
    l = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
    l.pack_start(Gtk.Label('Algorithm:'), False, False, 8)
    space_vert.pack_start(l, False, False, 0)
    space_vert.pack_start(self.space_alg, False, False, 0)
    
    space_vert.pack_start(Gtk.Separator(), False, False, 8)
    
    self.default_gap = Gtk.SpinButton()
    self.default_gap_space = Gtk.SpinButton()
    
    self.default_gap.set_adjustment(Gtk.Adjustment(0.2, -0.5, 2.0, 0.1, 0.2, 0.4))
    self.default_gap_space.set_adjustment(Gtk.Adjustment(0.6, 0.0, 4.0, 0.1, 0.2, 0.4))
    
    self.default_gap.set_digits(2)
    self.default_gap_space.set_digits(2)
    
    l = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
    l.pack_start(Gtk.Label('Default letter gap:'), False, False, 8)
    space_vert.pack_start(l, False, False, 0)
    space_vert.pack_start(self.default_gap, False, False, 0)
    
    l = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
    l.pack_start(Gtk.Label('Default space size:'), False, False, 8)
    space_vert.pack_start(l, False, False, 0)
    space_vert.pack_start(self.default_gap_space, False, False, 0)
    
    space_vert.pack_start(Gtk.Separator(), False, False, 8)
    
    self.space_match_id = Gtk.SpinButton()
    self.space_match_char = Gtk.SpinButton()
    self.space_match_type = Gtk.SpinButton()
    self.space_has_space = Gtk.SpinButton()
    
    self.space_match_id.set_adjustment(Gtk.Adjustment(10.0, 0.0, 128.0, 1.0, 2.0, 4.0))
    self.space_match_char.set_adjustment(Gtk.Adjustment(3.0, 0.0, 128.0, 1.0, 2.0, 4.0))
    self.space_match_type.set_adjustment(Gtk.Adjustment(2.0, 0.0, 128.0, 1.0, 2.0, 4.0))
    self.space_has_space.set_adjustment(Gtk.Adjustment(0.01, 0.0, 1.0, 0.01, 0.05, 0.1))
    
    self.space_match_id.set_digits(1)
    self.space_match_char.set_digits(1)
    self.space_match_type.set_digits(1)
    self.space_has_space.set_digits(3)
    
    l = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
    l.pack_start(Gtk.Label('Same glyph multiplier:'), False, False, 8)
    space_vert.pack_start(l, False, False, 0)
    space_vert.pack_start(self.space_match_id, False, False, 0)
    
    l = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
    l.pack_start(Gtk.Label('Same character multiplier:'), False, False, 8)
    space_vert.pack_start(l, False, False, 0)
    space_vert.pack_start(self.space_match_char, False, False, 0)
    
    l = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
    l.pack_start(Gtk.Label('Same type multiplier:'), False, False, 8)
    space_vert.pack_start(l, False, False, 0)
    space_vert.pack_start(self.space_match_type, False, False, 0)
    
    l = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
    l.pack_start(Gtk.Label('Edge of word penalty:'), False, False, 8)
    space_vert.pack_start(l, False, False, 0)
    space_vert.pack_start(self.space_has_space, False, False, 0)
    
    space_vert.pack_start(Gtk.Separator(), False, False, 8)
    
    self.space_amount = Gtk.SpinButton()
    self.space_amount.set_adjustment(Gtk.Adjustment(0.5, 0.0, 1.0, 0.01, 0.05, 0.1))
    self.space_amount.set_digits(3)
    
    l = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
    l.pack_start(Gtk.Label('Draw region size:'), False, False, 8)
    space_vert.pack_start(l, False, False, 0)
    space_vert.pack_start(self.space_amount, False, False, 0)
    
    space_vert.pack_start(Gtk.Separator(), False, False, 8)
    
    self.flow_gbp = Gtk.CheckButton('Tweak glyphs flow')
    self.flow_gbp.set_active(True)
    space_vert.pack_start(self.flow_gbp, False, False, 0)
    
    self.flow_pos_sd = Gtk.SpinButton()
    self.flow_pos_sd.set_adjustment(Gtk.Adjustment(1.0, 0.01, 10.0, 0.01, 0.1, 0.1))
    self.flow_pos_sd.set_digits(3)
    
    self.flow_offset_sd = Gtk.SpinButton()
    self.flow_offset_sd.set_adjustment(Gtk.Adjustment(0.2, 0.01, 10.0, 0.01, 0.1, 0.1))
    self.flow_offset_sd.set_digits(3)
    
    l = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
    l.pack_start(Gtk.Label('Position SD:'), False, False, 8)
    space_vert.pack_start(l, False, False, 0)
    space_vert.pack_start(self.flow_pos_sd, False, False, 0)
    
    l = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
    l.pack_start(Gtk.Label('Offset SD:'), False, False, 8)
    space_vert.pack_start(l, False, False, 0)
    space_vert.pack_start(self.flow_offset_sd, False, False, 0)
    
    self.flow_gbp_rf = Gtk.CheckButton('Transfer joined up to unjoined up')
    self.flow_gbp_rf.set_active(True)
    space_vert.pack_start(self.flow_gbp_rf, False, False, 0)
    
    self.flow_gbp_cp = Gtk.CheckButton('Compensate punctuation')
    self.flow_gbp_cp.set_active(True)
    space_vert.pack_start(self.flow_gbp_cp, False, False, 0)
    
    
    # Setup the chunks tab...
    chunk_vert = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
    self.chunks.pack_start(chunk_vert, True, True, 0)
    
    self.chunk_adv_match = Gtk.CheckButton('Advanced Chunk Matching')
    self.chunk_adv_match.set_active(True)
    
    self.chunk_choices = Gtk.SpinButton()
    self.chunk_memory = Gtk.SpinButton()
    self.chunk_samples = Gtk.SpinButton()
    self.chunk_direction = Gtk.SpinButton()
    self.chunk_radius = Gtk.SpinButton()
    self.chunk_density = Gtk.SpinButton()
    
    self.chunk_choices.set_adjustment(Gtk.Adjustment(8, 1, 17, 2, 1, 1))
    self.chunk_memory.set_adjustment(Gtk.Adjustment(4, 0, 17, 2, 1, 1))
    self.chunk_samples.set_adjustment(Gtk.Adjustment(4, 2, 17, 2, 1, 1))
    self.chunk_direction.set_adjustment(Gtk.Adjustment(1.0, 0.1, 17.0, 0.5, 1.0, 1.0))
    self.chunk_radius.set_adjustment(Gtk.Adjustment(2.0, 0.0, 17.0, 0.5, 1.0, 1.0))
    self.chunk_density.set_adjustment(Gtk.Adjustment(4.0, 0.0, 17.0, 0.5, 1.0, 1.0))
    
    self.chunk_choices.set_digits(1)
    self.chunk_memory.set_digits(1)
    self.chunk_samples.set_digits(1)
    self.chunk_direction.set_digits(1)
    self.chunk_radius.set_digits(1)
    self.chunk_density.set_digits(1)
    
    chunk_vert.pack_start(self.chunk_adv_match, False, False, 0)
    
    l = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
    l.pack_start(Gtk.Label('Chunk Choices:'), False, False, 8)
    chunk_vert.pack_start(l, False, False, 0)
    chunk_vert.pack_start(self.chunk_choices, False, False, 0)
    
    l = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
    l.pack_start(Gtk.Label('Repeat Avoidance Memory:'), False, False, 8)
    chunk_vert.pack_start(l, False, False, 0)
    chunk_vert.pack_start(self.chunk_memory, False, False, 0)
    
    l = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
    l.pack_start(Gtk.Label('Chunk Feature Samples:'), False, False, 8)
    chunk_vert.pack_start(l, False, False, 0)
    chunk_vert.pack_start(self.chunk_samples, False, False, 0)
    
    l = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
    l.pack_start(Gtk.Label('Chunk Direction Weight:'), False, False, 8)
    chunk_vert.pack_start(l, False, False, 0)
    chunk_vert.pack_start(self.chunk_direction, False, False, 0)
    
    l = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
    l.pack_start(Gtk.Label('Chunk Radius Weight:'), False, False, 8)
    chunk_vert.pack_start(l, False, False, 0)
    chunk_vert.pack_start(self.chunk_radius, False, False, 0)

    l = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
    l.pack_start(Gtk.Label('Chunk Density Weight:'), False, False, 8)
    chunk_vert.pack_start(l, False, False, 0)
    chunk_vert.pack_start(self.chunk_density, False, False, 0)
    
    
    # Setup the rendering algorithm tab...
    render_vert = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
    self.rendering.pack_start(render_vert, True, True, 0)
    
    self.link_type = Gtk.ComboBoxText()
    self.link_type.append_text('none')
    self.link_type.append_text('linear')
    self.link_type.append_text('sin')
    self.link_type.set_active(2)
    
    l = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
    l.pack_start(Gtk.Label('Link type:'), False, False, 8)
    render_vert.pack_start(l, False, False, 0)
    render_vert.pack_start(self.link_type, False, False, 0)
    
    render_vert.pack_start(Gtk.Separator(), False, False, 8)
    
    self.line_height = Gtk.SpinButton()
    self.line_spacing = Gtk.SpinButton()
    
    self.line_height.set_adjustment(Gtk.Adjustment(192.0, 1.0, 512.0, 8.0, 16.0, 32.0))
    self.line_spacing.set_adjustment(Gtk.Adjustment(1.5, 0.0, 8.0, 0.1, 0.2, 0.4))
    
    self.line_height.set_digits(1)
    self.line_spacing.set_digits(2)
    
    self.render_linear = Gtk.CheckButton('Linear Interpolation')
    self.render_linear.set_active(True)
    
    l = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
    l.pack_start(Gtk.Label('Line height:'), False, False, 8)
    render_vert.pack_start(l, False, False, 0)
    render_vert.pack_start(self.line_height, False, False, 0)
    
    l = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
    l.pack_start(Gtk.Label('Line spacing:'), False, False, 8)
    render_vert.pack_start(l, False, False, 0)
    render_vert.pack_start(self.line_spacing, False, False, 0)
    
    render_vert.pack_start(self.render_linear, False, False, 8)
    
    render_vert.pack_start(Gtk.Separator(), False, False, 8)
    
    self.blending = Gtk.ComboBoxText()
    self.blending.append_text('Last Layer')
    self.blending.append_text('Averaging')
    self.blending.append_text('Minimum Cut')
    self.blending.append_text('Graph Cuts')
    self.blending.set_active(3)
    
    l = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
    l.pack_start(Gtk.Label('Blending:'), False, False, 8)
    render_vert.pack_start(l, False, False, 0)
    render_vert.pack_start(self.blending, False, False, 0)
    
    self.blend_radius = Gtk.SpinButton()
    self.blend_stretch = Gtk.SpinButton()
    self.blend_edge = Gtk.SpinButton()
    self.blend_smooth = Gtk.SpinButton()
    self.blend_alpha = Gtk.SpinButton()
    self.blend_unary = Gtk.SpinButton()
    self.blend_overlap = Gtk.SpinButton()
    
    self.blend_radius.set_adjustment(Gtk.Adjustment(12.0, 0.0, 16.0, 1.0, 1.0, 1.0))
    self.blend_stretch.set_adjustment(Gtk.Adjustment(0.5, 0.0, 8.0, 0.1, 0.1, 0.1))
    self.blend_edge.set_adjustment(Gtk.Adjustment(0.5, 0.0, 8.0, 0.1, 0.1, 0.1))
    self.blend_smooth.set_adjustment(Gtk.Adjustment(2.0, 0.0, 8.0, 0.1, 0.1, 0.1))
    self.blend_alpha.set_adjustment(Gtk.Adjustment(1.0, 0.0, 8.0, 0.1, 0.1, 0.1))
    self.blend_unary.set_adjustment(Gtk.Adjustment(1.0, 0.0, 8.0, 0.1, 0.1, 0.1))
    self.blend_overlap.set_adjustment(Gtk.Adjustment(0.0, 0.0, 1.0, 0.1, 0.1, 0.1))
    
    self.blend_radius.set_digits(1)
    self.blend_stretch.set_digits(1)
    self.blend_edge.set_digits(1)
    self.blend_smooth.set_digits(1)
    self.blend_alpha.set_digits(1)
    self.blend_unary.set_digits(1)
    self.blend_overlap.set_digits(2)
    
    l = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
    l.pack_start(Gtk.Label('Extra line radius:'), False, False, 8)
    render_vert.pack_start(l, False, False, 0)
    render_vert.pack_start(self.blend_radius, False, False, 0)
    
    l = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
    l.pack_start(Gtk.Label('Stretched texture down weighting:'), False, False, 8)
    render_vert.pack_start(l, False, False, 0)
    render_vert.pack_start(self.blend_stretch, False, False, 0)
    
    l = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
    l.pack_start(Gtk.Label('Edge preservation weight:'), False, False, 8)
    render_vert.pack_start(l, False, False, 0)
    render_vert.pack_start(self.blend_edge, False, False, 0)
    
    l = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
    l.pack_start(Gtk.Label('Colour consistancy weight:'), False, False, 8)
    render_vert.pack_start(l, False, False, 0)
    render_vert.pack_start(self.blend_smooth, False, False, 0)
    
    l = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
    l.pack_start(Gtk.Label('Opaqueness bias:'), False, False, 8)
    render_vert.pack_start(l, False, False, 0)
    render_vert.pack_start(self.blend_alpha, False, False, 0)
    
    l = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
    l.pack_start(Gtk.Label('Unary cost multiplier:'), False, False, 8)
    render_vert.pack_start(l, False, False, 0)
    render_vert.pack_start(self.blend_unary, False, False, 0)
    
    l = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
    l.pack_start(Gtk.Label('Overlap weight:'), False, False, 8)
    render_vert.pack_start(l, False, False, 0)
    render_vert.pack_start(self.blend_overlap, False, False, 0)


    # Setup the glyph pane...
    self.glyph_visible = False
    self.glyph_viewer = Viewer()
    self.glyph_viewer.set_bg(1.0, 1.0, 1.0)
    self.glyph.pack_start(self.glyph_viewer, True, True, 0)
    
    self.glyph_image = TileImage(None)
    self.glyph_viewer.add_layer(self.glyph_image)
    self.code = 0
    
    self.glyph_line = None
    self.glyph_line_tiles = LineLayer()
    self.glyph_viewer.add_layer(self.glyph_line_tiles)
    
    
    # Cache of textures used for rendering, to save loading them each time...
    self.textures = TextureCache(32)
    
    # To optimise the presentation of information by the viewer...
    self.seg = None
  
  
  
  def __generate(self, widget):
    """Generates a new sample of text given all the current settings"""
    if self.glyph_db.empty():
      print 'No database avaliable - aborting.'
      return
      
    print 'Starting generation...'
    buf = self.text.get_buffer()
    txt = buf.get_text(buf.get_start_iter(), buf.get_end_iter(), False).split('\n')
    
    lines = []
    for i in xrange(len(txt)):
      if len(txt[i])==0: continue
      print 'L%i::Selecting Glyphs...'%(i+1)
      
      fetch_count = self.fetch_count.get_value()
      match_strength = self.match_strength.get_value()
      wrong_place_cost = self.wrong_place_cost.get_value()
      space_mult = self.space_mult.get_value()
      
      randomness = self.randomness.get_value()
      match_strength /= randomness
      wrong_place_cost /= randomness
      
      if self.sel_alg.get_active()==0:
        glyph_list = select_glyphs_random(txt[i], self.glyph_db)
      elif self.sel_alg.get_active()==1:
        glyph_list = select_glyphs_better_random(txt[i], self.glyph_db, fetch_count)
      elif self.sel_alg.get_active()==2:
        cfunc = costs.end_dist_cost_rf if self.transfer_adj_cost.get_active() else costs.end_dist_cost
        glyph_list = select_glyphs_dp(txt[i], self.glyph_db, fetch_count, match_strength, wrong_place_cost, space_mult, cfunc, False)
      else:
        cfunc = costs.end_dist_cost_rf if self.transfer_adj_cost.get_active() else costs.end_dist_cost
        glyph_list = select_glyphs_dp(txt[i], self.glyph_db, fetch_count, match_strength, wrong_place_cost, space_mult, cfunc, True)
    
    
      print 'L%i::Positioning Glyphs...'%(i+1)
      
      default_gap = self.default_gap.get_value()
      default_gap_space = self.default_gap_space.get_value()
      
      space_param = (self.space_match_id.get_value(), self.space_match_char.get_value(), self.space_match_type.get_value(), self.space_has_space.get_value())
      
      amount = self.space_amount.get_value()
      
      if self.space_alg.get_active()==0:
        glyph_layout = layout_fixed(glyph_list, self.glyph_db, default_gap, default_gap_space)
      elif self.space_alg.get_active()==1:
        glyph_layout = layout_source(glyph_list, self.glyph_db, default_gap, default_gap_space)
      elif self.space_alg.get_active()==2:
        glyph_layout = layout_median(glyph_list, self.glyph_db, space_param)
      else:
        glyph_layout = layout_draw(glyph_list, self.glyph_db, space_param, amount)
      
      
      if self.flow_gbp.get_active():
        print 'L%i::Adjusting flow (vertical offset)...'%(i+1)
        pos_sd = self.flow_pos_sd.get_value()
        offset_sd = self.flow_offset_sd.get_value()
        glyph_layout = layout_flow(glyph_layout, pos_sd, offset_sd, self.flow_gbp_rf.get_active(), self.flow_gbp_cp.get_active())
    
    
      print 'L%i::Joining up Glyphs...'%(i+1)
      if self.link_type.get_active()==0:
        lg_layout = stitch_noop(glyph_layout)
      else:
        lg_layout = stitch_connect(glyph_layout, self.link_type.get_active()==2, not self.chunk_db.empty(), i)
    
    
      print 'L%i::Combining to obtain final line...'%(i+1)
      line = combine_seperate(lg_layout)
    
    
      print 'L%i::Tweaking scale, moving...'%(i+1)
      hg = numpy.eye(3, dtype=numpy.float32)
      hg[1,2] = i*self.line_spacing.get_value() # Number of lines high to make each line.
      hg[2,2] /= self.line_height.get_value()
      line.transform(hg, True)
      
      lines.append(line)
    
    
    self.line = LineGraph()
    self.line.from_many(*lines)
    
    if not self.chunk_db.empty():
      print '::Changing pen style...'
      
      self.chunk_db.set_params(int(self.chunk_samples.get_value()), self.chunk_direction.get_value(), self.chunk_radius.get_value(), self.chunk_density.get_value())
      
      self.line = self.chunk_db.convert(self.line, int(self.chunk_choices.get_value()), self.chunk_adv_match.get_active(), self.textures, int(self.chunk_memory.get_value()))
    
    print '::Recording to image...'
    
    blend_radius = self.blend_radius.get_value()
    blend_stretch = self.blend_stretch.get_value()
    blend_edge = self.blend_edge.get_value()
    blend_smooth = self.blend_smooth.get_value()
    blend_alpha = self.blend_alpha.get_value()
    blend_unary = self.blend_unary.get_value()
    blend_overlap = self.blend_overlap.get_value()
    linear = self.render_linear.get_active()
    
    image, count = render(self.line, 8, self.textures, self.blending.get_active(), blend_radius, blend_stretch, blend_edge, blend_smooth, blend_alpha, blend_unary, blend_overlap, linear)
    self.image.from_array(image)
    
    if count!=0: print '  |Solved %i min cut problems'%count
    
    
    print 'Generation complete'
    self.viewer.reset_view()
    self.__line_visible(self.action_show_line)


  def __add_glyphs(self, widget):
    """Allows the user to load some glyphs from a line graph file."""
    dialog = Gtk.FileChooserDialog('Select a line graph...', self, Gtk.FileChooserAction.OPEN, (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_OPEN, Gtk.ResponseType.OK))

    filter_lg = Gtk.FileFilter()
    filter_lg.set_name('Line Graph files')
    filter_lg.add_pattern('*.line_graph')
    dialog.add_filter(filter_lg)

    filter_all = Gtk.FileFilter()
    filter_all.set_name('All files')
    filter_all.add_pattern('*')
    dialog.add_filter(filter_all)
    
    dialog.set_select_multiple(True)
    dialog.set_filename(os.path.join(os.path.abspath('.'), '.*'))

    response = dialog.run()
    if response==Gtk.ResponseType.OK:
      for fn in dialog.get_filenames():
        print 'Openning %s...'%fn
      
        count = self.glyph_db.add(fn)
        if count!=0:
          self.file_list.append([fn,count])
      
        print 'File loaded'
    
    dialog.destroy()


  def __rem_glyphs(self, widget):
    """Removes a file, and its associated set of glyphs."""
    model, treeiter = self.file_view.get_selection().get_selected()
    
    if treeiter!=None:
      self.glyph_db.rem(model[treeiter][0])
      del model[treeiter]
  
  
  def __add_chunks(self, widget):
    """Allows the user to load some chunks from a line graph file."""
    dialog = Gtk.FileChooserDialog('Select a line graph...', self, Gtk.FileChooserAction.OPEN, (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_OPEN, Gtk.ResponseType.OK))

    filter_lg = Gtk.FileFilter()
    filter_lg.set_name('Line Graph files')
    filter_lg.add_pattern('*.line_graph')
    dialog.add_filter(filter_lg)

    filter_all = Gtk.FileFilter()
    filter_all.set_name('All files')
    filter_all.add_pattern('*')
    dialog.add_filter(filter_all)
    
    dialog.set_select_multiple(True)
    dialog.set_filename(os.path.join(os.path.abspath('.'), '.*'))

    response = dialog.run()
    if response==Gtk.ResponseType.OK:
      for fn in dialog.get_filenames():
        print 'Openning %s...'%fn
      
        count = self.chunk_db.add(fn)
        if count!=0:
          self.chunk_list.append([fn,count])
      
        print 'File loaded'
    
    dialog.destroy()


  def __rem_chunks(self, widget):
    """Removes a file, and its associated set of chunks."""
    model, treeiter = self.chunk_view.get_selection().get_selected()
    
    if treeiter!=None:
      self.chunk_db.rem(model[treeiter][0])
      del model[treeiter]


  def __tab_change(self, widget, page, page_index):
    self.glyph_visible = False
    
    if page==self.stats:
      # Need to update the statistics...
      lines = []
      
      if self.glyph_db.empty():
        lines.append('Database empty')
      else:
        lines.append('Characters: (surrounded, space before, space after, seperate) = total')
        
        chars = self.glyph_db.stats()
        for char in sorted(chars.keys()):
          s = chars[char]
          lines.append('%c: (%i, %i, %i, %i) = %i' % (char, s[0], s[1], s[2], s[3], sum(s)))
      
      self.stats_text.get_buffer().set_text('\n'.join(lines))
    
    if page==self.glyph:
      self.glyph_visible = True
      self.update_glyph()
  

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
      self.glyph_image.set_interpolate(True)
    else:
      self.image.set_interpolate(False)
      self.glyph_image.set_interpolate(False)

    self.viewer.queue_draw()
    self.glyph_viewer.queue_draw()

  
  def __image_visible(self, widget = None):
    iv = self.action_image_visible.get_active()
    lv = self.action_show_line.get_active()
    
    if iv:
      if lv:
        self.image.set_tint(2.0/3.0)
        self.glyph_image.set_tint(2.0/3.0)
      else:
        self.image.set_tint(0.0)
        self.glyph_image.set_tint(0.0)
    else:
      self.image.set_tint(1.0)
      self.glyph_image.set_tint(1.0)
    
    self.viewer.queue_draw()
    self.glyph_viewer.queue_draw()


  def __line_visible(self, widget):
    """Toggles if the line is visible or not - calculates it if needed."""
    if widget.get_active():
      # Arrange for its visibility...
      self.line_tiles.set_line(self.line)
      self.glyph_line_tiles.set_line(self.glyph_line)
    else:
      # Switch off its visibility...
      self.line_tiles.set_line(None)
      self.glyph_line_tiles.set_line(None)
    
    self.__image_visible()
  
  
  def __line_render_state(self, widget = None):
    """Updates the render state of the line to match the GUI."""
    rs = self.action_render_correct.get_current_value()
    self.line_tiles.set_mode(rs)
    self.glyph_line_tiles.set_mode(rs)
    
    self.viewer.queue_draw()
    self.glyph_viewer.queue_draw()


  def __create_uimanager(self):
    # Create the actions for the file menu action group...
    action_file = Gtk.Action("file", "File", None, None)

    action_add_lg = Gtk.Action('add_lg','Add Line Graph','Adds a Line Graph to the Glyph data base', Gtk.STOCK_OPEN)
    action_add_lg.connect('activate', self.__add_glyphs)
    
    action_save_image = Gtk.Action('save_image','Save Image','Saves the generated image', Gtk.STOCK_OPEN)
    action_save_image.connect('activate', self.__save_image)
    
    action_save_many = Gtk.Action('save_many','Save Many Images','Saves lots of images, calling generate before each one, so you can quickly generate a variety of outputs', Gtk.STOCK_OPEN)
    action_save_many.connect('activate', self.__save_many)

    action_quit = Gtk.Action('quit','Quit','Terminates this application', Gtk.STOCK_QUIT)
    action_quit.connect('activate', Gtk.main_quit)

    # Create the file menu action group...
    group_file = Gtk.ActionGroup('File')
    group_file.add_action_with_accel(action_file, '')
    group_file.add_action_with_accel(action_add_lg, '<ctrl>a')
    group_file.add_action_with_accel(action_save_image, '<ctrl>s')
    group_file.add_action_with_accel(action_save_many, '<ctrl><alt>s')
    group_file.add_action_with_accel(action_quit, '<ctrl>q')


    # Create the actions for the view menu action group...
    action_view = Gtk.Action("view", "View", None, None)

    action_fullscreen = Gtk.ToggleAction('fullscreen','Fullscreen','Toggles fullscreen mode', Gtk.STOCK_FULLSCREEN)
    action_fullscreen.connect('activate', self.__fullscreen)

    action_fit_to_screen = Gtk.Action('fit_to_screen','Fit to screen','Adjusts the view so that the sctroke fills the entire screen.', Gtk.STOCK_ZOOM_FIT)
    action_fit_to_screen.connect('activate', lambda w: self.viewer.reset_view())
    action_one_to_one = Gtk.Action('one_to_one','Reset zoom','Resets the zoom to the original recording scale.', Gtk.STOCK_ZOOM_100)
    action_one_to_one.connect('activate', lambda w: self.viewer.one_to_one())
    action_zoom_in = Gtk.Action('zoom_in','Zoom In','Get closer to the image without moving your head.', Gtk.STOCK_ZOOM_IN)
    action_zoom_in.connect('activate', lambda w: self.viewer.zoom(True))
    action_zoom_out = Gtk.Action('zoom_out','Zoom Out','Get further away from the image without moving your head.', Gtk.STOCK_ZOOM_OUT)
    action_zoom_out.connect('activate', lambda w: self.viewer.zoom(False))

    action_move_up = Gtk.Action('move_up','Move Up','Travel north across the stroke', Gtk.STOCK_GO_UP)
    action_move_up.connect('activate', lambda w: self.viewer.move(0, -64))
    action_move_right = Gtk.Action('move_right','Move Right','Travel east across the stroke', Gtk.STOCK_GO_FORWARD)
    action_move_right.connect('activate', lambda w: self.viewer.move(64, 0))
    action_move_down = Gtk.Action('move_down','Move Down','Travel south across the stroke', Gtk.STOCK_GO_DOWN)
    action_move_down.connect('activate', lambda w: self.viewer.move(0, 64))
    action_move_left = Gtk.Action('move_left','Move Left','Travel west across the stroke', Gtk.STOCK_GO_BACK)
    action_move_left.connect('activate', lambda w: self.viewer.move(-64, 0))

    # Create the view menu action group...
    group_view = Gtk.ActionGroup('View')
    group_view.add_action_with_accel(action_view, '')
    group_view.add_action_with_accel(action_fullscreen, 'F11')
    group_view.add_action_with_accel(action_fit_to_screen, '<ctrl>f')
    group_view.add_action_with_accel(action_one_to_one, '<ctrl>1')
    group_view.add_action_with_accel(action_zoom_in, '<ctrl>Up')
    group_view.add_action_with_accel(action_zoom_out, '<ctrl>Down')
    group_view.add_action_with_accel(action_move_up, '<alt>Up')
    group_view.add_action_with_accel(action_move_right, '<alt>Right')
    group_view.add_action_with_accel(action_move_down, '<alt>Down')
    group_view.add_action_with_accel(action_move_left, '<alt>Left')

    
    # Create the display actions...
    action_display = Gtk.Action("display", "Display", None, None)
    
    self.action_image_visible = Gtk.ToggleAction('show_image','Show Image','Toggles if the image is visible or not', Gtk.STOCK_DND)
    self.action_image_visible.set_active(True)
    self.action_image_visible.connect('activate', self.__image_visible)
    action_interpolate = Gtk.ToggleAction('interpolate','Interpolate','Toggles if the image is interpolated or not', Gtk.STOCK_REFRESH)
    action_interpolate.set_active(True)
    action_interpolate.connect('activate', self.__interpolate)
    self.action_show_line = Gtk.ToggleAction('show_line','Show Line','Toggles if the line extracted from the text is shown or not', Gtk.STOCK_SELECT_ALL)
    self.action_show_line.connect('activate', self.__line_visible)
    
    self.action_render_correct = Gtk.RadioAction('render_correct','Render Correct Line', 'Makes the line render with the correct width.', Gtk.STOCK_PRINT, LineLayer.RENDER_CORRECT)
    self.action_render_thin = Gtk.RadioAction('render_thin','Render Thin Line', 'Makes the line render with a line of width one in viewport space.', Gtk.STOCK_PRINT_REPORT, LineLayer.RENDER_THIN)
    self.action_render_one = Gtk.RadioAction('render_one','Render Bare Line', 'Makes the line render with a line of width one in image space.', Gtk.STOCK_PRINT_PREVIEW, LineLayer.RENDER_ONE)
    self.action_render_weird = Gtk.RadioAction('render_weird','Render Weird Line', 'Makes the line render with the correct width but ignoring viewport scale.', Gtk.STOCK_PRINT_WARNING, LineLayer.RENDER_WEIRD)
    
    self.action_render_correct.set_active(True)
    self.action_render_thin.join_group(self.action_render_correct)
    self.action_render_one.join_group(self.action_render_correct)
    self.action_render_weird.join_group(self.action_render_correct)
    
    self.action_render_correct.connect('activate', self.__line_render_state)
    self.action_render_thin.connect('activate', self.__line_render_state)
    self.action_render_one.connect('activate', self.__line_render_state)
    self.action_render_weird.connect('activate', self.__line_render_state)
    
    self.action_bg_colour = Gtk.Action('bg_colour','Set Background','Sets the background colour of the display area', Gtk.STOCK_SELECT_COLOR)
    self.action_bg_colour.connect('activate', self.__bg_colour)
    
    # Create the display menu action group...
    group_display = Gtk.ActionGroup('Display')
    group_display.add_action_with_accel(action_display, '')
    group_display.add_action_with_accel(self.action_image_visible, '<ctrl>t')
    group_display.add_action_with_accel(action_interpolate, '<ctrl>i')
    group_display.add_action_with_accel(self.action_show_line, '<ctrl>l')
    
    group_display.add_action_with_accel(self.action_render_correct, 'F1')
    group_display.add_action_with_accel(self.action_render_thin, 'F2')
    group_display.add_action_with_accel(self.action_render_one, 'F3')
    group_display.add_action_with_accel(self.action_render_weird, 'F4')
    
    group_display.add_action_with_accel(self.action_bg_colour, '<ctrl>b')
    
    
    # Create the UI description string thingy...
    ui = """
    <ui>
     <menubar name='menu_bar'>
      <menu action='file'>
       <menuitem action='add_lg'/>
       <separator/>
       <menuitem action='save_image'/>
       <menuitem action='save_many'/>
       <separator/>
       <menuitem action='quit'/>
      </menu>
      <menu action='view'>
       <menuitem action='fullscreen'/>
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
      <menu action='display'>
       <menuitem action='show_image'/>
       <menuitem action='interpolate'/>
       <menuitem action='show_line'/>
       <separator/>
       <menuitem action='render_correct'/>
       <menuitem action='render_thin'/>
       <menuitem action='render_one'/>
       <menuitem action='render_weird'/>
       <separator/>
       <menuitem action='bg_colour'/>
      </menu>
     </menubar>
     <toolbar name='icon_bar'>
      <toolitem action='fullscreen'/>
      <separator/>
      <toolitem action='zoom_out'/>
      <toolitem action='fit_to_screen'/>
      <toolitem action='one_to_one'/>
      <toolitem action='zoom_in'/>
      <separator/>
      <toolitem action='show_image'/>
      <toolitem action='interpolate'/>
      <toolitem action='show_line'/>
      <separator/>
      <toolitem action='render_correct'/>
      <toolitem action='render_thin'/>
      <toolitem action='render_one'/>
      <toolitem action='render_weird'/>
      <separator/>
      <toolitem action='bg_colour'/>
     </toolbar>
    </ui>
    """

     
    # Use the various assets we have created to make and return the manager...
    ret = Gtk.UIManager()
    ret.add_ui_from_string(ui)
    ret.insert_action_group(group_file)
    ret.insert_action_group(group_view)
    ret.insert_action_group(group_display)

    return ret
  
  
  def __save_image(self, widget):
    dialog = Gtk.FileChooserDialog('Select an image file...', self, Gtk.FileChooserAction.SAVE, (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_SAVE, Gtk.ResponseType.OK))
  
    filter_png = Gtk.FileFilter()
    filter_png.set_name('Png image files')
    filter_png.add_pattern('*.png')
    dialog.add_filter(filter_png)
  
    filter_all = Gtk.FileFilter()
    filter_all.set_name('All files')
    filter_all.add_pattern('*')
    dialog.add_filter(filter_all)
  
    response = dialog.run()
    if response==Gtk.ResponseType.OK:
      fn = dialog.get_filename()
      if not fn.endswith('.png'): fn += '.png'
      print 'Saving to %s'%fn

      # Save the file...        
      if os.path.exists(fn):
        shutil.copy2(fn, fn+'~')
      
      self.image.get_original().write_to_png(fn)
      
    print 'Success!'
  
    dialog.destroy()
  
  
  def __save_many(self, widget):
    dialog = Gtk.FileChooserDialog('Select an image file template...', self, Gtk.FileChooserAction.SAVE, (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_SAVE, Gtk.ResponseType.OK))
  
    filter_png = Gtk.FileFilter()
    filter_png.set_name('Png image files')
    filter_png.add_pattern('*.png')
    dialog.add_filter(filter_png)
  
    filter_all = Gtk.FileFilter()
    filter_all.set_name('All files')
    filter_all.add_pattern('*')
    dialog.add_filter(filter_all)
    
    
    count = Gtk.SpinButton()
    count.set_adjustment(Gtk.Adjustment(8, 1, 1024, 1, 1, 1))
    count.set_value(8)
    count.set_digits(0)
    
    l = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
    l.pack_start(Gtk.Label('Image Count:'), False, False, 8)
    
    extra = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
    extra.pack_start(l, False, False, 0)
    extra.pack_start(count, False, False, 0)

    dialog.set_extra_widget(extra)
    extra.show_all()
  
    response = dialog.run()
    if response==Gtk.ResponseType.OK:
      base_fn = dialog.get_filename()
      if not base_fn.endswith('.png'): base_fn += '.png'
      base_fn, ext = os.path.splitext(base_fn)
      
      count = int(count.get_value())
      
      for i in xrange(count):
        print '::::Doing %i of %i:' % (i+1, count)
        fn = base_fn + '%04i'%i + ext
        
        # Generate a new draw...
        self.__generate(None)
        
        # Save the file...        
        if os.path.exists(fn):
          shutil.copy2(fn, fn+'~')
        self.image.get_original().write_to_png(fn)
        
        print '::::Saved to %s.' % fn
      
      print 'All done!'
  
    dialog.destroy()
  
  
  def __on_move(self, x, y):
    if self.line==None: return
    vp = self.viewer.get_viewport()
    if vp==None: return
      
    x, y =  vp.view_to_original(x, y)
    distance, edge, t = self.line.nearest(x, y)
    seg = self.line.get_segment(edge, t)
    
    if seg!=self.seg:
      self.seg = seg
      
      tags = self.line.get_tags(self.seg)
      
      self.left_link = ''
      self.key = ''
      self.right_link = ''
      self.lg_fn = ''
      self.source_x = -1.0
      self.source_y = -1.0
      
      for tag in tags:
        if len(filter(lambda c: c!='_', tag[0]))==1:
          self.key = tag[0]
        if tag[0].startswith('file:'):
          self.lg_fn = os.path.basename(tag[0][5:])
        if tag[0].startswith('link:left:'):
          self.left_link = tag[0][-1]
        if tag[0].startswith('link:right:'):
          self.right_link = tag[0][-1]
        if tag[0].startswith('code:'):
          self.code = int(tag[0][5:])
        if tag[0].startswith('source:x:'):
          self.source_x = float(tag[0][9:])
        if tag[0].startswith('source:y:'):
          self.source_y = float(tag[0][9:])
        
      if self.key=='':
        self.left_link = '-'
        self.right_link = '-'
        
      self.update_glyph()
    
    self.status.set_text('(%.1f,%.1f) | %s <%s> %s | %s(%.2f,%.2f)' % (x, y, self.left_link, self.key, self.right_link, self.lg_fn, self.source_x, self.source_y))
  
  
  def update_glyph(self):
    # Only bother if the glyph tab is visible...
    if self.glyph_visible==True:
      # Obtain the glyph object of the current glyph...
      glyph = self.glyph_db.get_single(self.code)
      
      if glyph!=None:
        # Scale the line graph to something sensible...
        self.glyph_line = LineGraph()
        self.glyph_line.from_many(glyph.get_linegraph())
        
        hg = numpy.eye(3, dtype=numpy.float32)
        hg[2,2] /= 2.0 * self.line_height.get_value()
        self.glyph_line.transform(hg, True)
        
        # Render the line graph up...
        image, _ = render(self.glyph_line, 4, self.textures, 0)
        
        self.glyph_image.from_array(image)
        self.glyph_viewer.reset_view()
        
        self.__line_visible(self.action_show_line)


  def __bg_colour(self, widget):
    dialog = Gtk.ColorSelectionDialog('Select background colour...')
    cs = dialog.get_color_selection()
    
    col = self.viewer.get_bg()
    cs.set_current_color(Gdk.Color(int(65535 * col[0]), int(65535 * col[1]), int(65535 * col[2])))
    
    response = dialog.run()
    if response==Gtk.ResponseType.OK:
      col = cs.get_current_color()
      self.viewer.set_bg(col.red / 65535.0, col.green / 65535.0, col.blue / 65535.0)
      self.viewer.queue_draw()
    
    dialog.destroy()
