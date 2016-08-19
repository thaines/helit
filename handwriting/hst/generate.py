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
import string
import random
import numpy
import numpy.linalg as la

import cairo
from gi.repository import Gdk, GdkPixbuf

from line_graph.line_graph import LineGraph

from graph_cuts import maxflow # Not actually needed by this module, but composite uses it from c, and can't compile it if its not up-to-date - doing this makes sure it is.
from composite import Composite
from texture_cache import TextureCache

from spacing import Spacing
import costs

from ddp.ddp import DDP
from gbp.gbp import GBP



def select_glyphs_random(text, glyph_db, log_func = None):
  """Given some text selects the glyphs to glue together to generate the text; returns an array of Glyph objects, one for each character, with None entries to indicate where spaces go. Glyph selection is entirly random, and ignores word position."""
  
  def do_char(c):
    if c==' ': return None
    
    glyphs = glyph_db.glyph(c)
    if len(glyphs)==0:
      if log_func!=None:
        log_func('Dropped character %s due to having no glyphs.'%c)
      return False
    
    return random.choice(glyphs)
  
  return filter(lambda x: x!=False, map(do_char, text))



def select_glyphs_better_random(text, glyph_db, fetch_count = 8, log_func = None):
  """A better random selection - takes into account word position."""
  ret = []
  
  for i in xrange(len(text)):
    if text[i] in string.whitespace: ret.append(None)
    else:
      space_before = True if i==0 or text[i-1] in string.whitespace else False
      space_after = True if i+1==len(text) or text[i+1] in string.whitespace else False
      
      key = ('_' if space_before else '') + text[i] + ('_' if space_after else '')
      glyphs = glyph_db.topup_glyph(key, fetch_count)
      
      if len(glyphs)!=0:
        ret.append(random.choice(glyphs))
      
      else:
        if log_func!=None:
          log_func('Dropped character %s due to having no glyphs.'% text[i])
          
  return ret



def select_glyphs_dp(text, glyph_db, fetch_count = 8, match_mult = 1.0, poor_fit_cost = 0.1, cost_space = 0.5, cost_func = costs.end_dist_cost, noise = False, log_func = None):
  # First pass - get a list of glyphs for each slot...
  choice = []
  
  for i in xrange(len(text)):
    if text[i] in string.whitespace: choice.append(None)
    else:
      space_before = True if i==0 or text[i-1] in string.whitespace else False
      space_after = True if i+1==len(text) or text[i+1] in string.whitespace else False
      
      key = ('_' if space_before else '') + text[i] + ('_' if space_after else '')
      glyphs = glyph_db.topup_glyph(key, fetch_count)
      
      if len(glyphs)!=0:
        choice.append(glyphs)
      
      else:
        if log_func!=None:
          log_func('Dropped character %s due to having no glyphs.'% text[i])
  
  # Second pass - calculate a constant message term, to bias towards better fitting glyphs...
  data_term = map(lambda gs: numpy.zeros(len(gs), dtype=numpy.float32) if gs!=None else None, choice)
  for i, gs in enumerate(choice):
    if gs!=None:
      space_before = True if i==0 or text[i-1] in string.whitespace else False
      space_after = True if i+1==len(text) or text[i+1] in string.whitespace else False
      
      for j in xrange(len(gs)):
        if gs[j].key.startswith('_') != space_before: data_term[i][j] += poor_fit_cost
        if gs[j].key.endswith('_') != space_after: data_term[i][j] += poor_fit_cost
        if gs[j].key.strip('_') != text[i]: data_term[i][j] += poor_fit_cost
  
  # If requested add noise to the unary term to simulate a draw...
  if noise:
    pass #####################
  
  # Third pass - calculate adjacency cost matrices for each glyph...
  smooth_term = []
  
  prev = None
  mult = match_mult
  for current in choice:
    if current==None: mult *= cost_space
    else:
      # Calculate the cost matrix between this glyph stack and the previous stack...
      if prev!=None:
        cost = numpy.empty((len(prev), len(current)), dtype=numpy.float32)
        
        for j, left in enumerate(prev):
          for i, right in enumerate(current):
            cost[j,i] = mult * cost_func(left, right)
        
        smooth_term.append(cost)
      
      # Move to next...
      prev = current
      mult = match_mult

  # Setup the dp solver...
  dp = DDP()
  
  clean_choice = filter(lambda c: c!=None, choice)
  clean_data = filter(lambda c: c!=None, data_term)
  
  dp.prepare(numpy.array([len(c) for c in clean_choice], dtype=numpy.int32))
  
  for i, data in enumerate(clean_data):
    dp.unary(i, data)
  
  for i, smooth in enumerate(smooth_term):
    dp.pairwise(i, 'full', smooth)
  
  # Solve...
  dp.solve()
  solution = dp.best()
  
  # Extract the results...
  offset = 0
  for i in xrange(len(choice)):
    if choice[i]!=None:
      choice[i] = choice[i][solution[offset]]
      offset += 1
  
  return choice


  
def layout_fixed(glyph_list, glyph_db, gap = 0.2, gap_space = 0.6, log_func = None):
  """Layout the glyphs - returns a list of (homography, glyph) pairs. This layout method does the stupid thing - constant user provided parameters."""
  
  # Go through and set the homographies each of the letters is the given distance from its neighbours...
  ret = []
  offset = 0.0
  
  for i, glyph in enumerate(glyph_list):
    if glyph==None:
      offset += gap_space - gap
      ret.append(None)
    else:
      hg = numpy.eye(3, dtype=numpy.float32)
      hg[0,2] = offset - glyph.left_x
      ret.append((hg, glyph))
      
      offset += gap + glyph.right_x - glyph.left_x

  return ret

  
  
def layout_source(glyph_list, glyph_db, gap = 0.2, gap_space = 0.6, log_func = None):
  """More sophisticated layout method - takes the gap between letters to be the average of the gaps in the source, for the characters being used. If gaps are not avaliable it falls back on the global median."""
  
  # Calculate the average distance between glyphs - used as a fallback...
  chars = string.letters + string.digits + string.punctuation
  
  diffs = glyph_db.diff(chars, chars, False)
  if len(diffs)==0: mean = gap
  else: median = numpy.median(diffs)
  
  diffs = glyph_db.diff(chars, chars, True)
  if len(diffs)==0: mean_space = gap_space
  else: median_space = numpy.median(diffs)
  
  # Iterate and process each letter in turn...
  ret = []
  offset = 0.0
    
  for i, glyph in enumerate(glyph_list):
    if glyph==None:
      offset += median_space
      ret.append(None)
      
    else:
      # Do this letter...
      hg = numpy.eye(3, dtype=numpy.float32)
      hg[0,2] = offset - glyph.left_x
      ret.append((hg, glyph))
      
      offset += glyph.right_x - glyph.left_x
      
      # Calculate a gap based on the stats, and apply it...
      next_glyph = glyph_list[i+1] if i+1<len(glyph_list) else None
      if next_glyph!=None:
        gaps = []
      
        if glyph.right!=None and not glyph.right[0].key.endswith('_'):
          gaps.append(glyph.right[0].orig_left_x() - glyph.orig_right_x())
        
      
        if next_glyph!=None and next_glyph.left!=None and not next_glyph.left[0].key.startswith('_'):
          gaps.append(next_glyph.orig_left_x() - next_glyph.left[0].orig_right_x())
      
        while len(gaps)<2: gaps.append(median)
        offset += sum(gaps) / float(len(gaps))

  return ret

  

def layout_median(glyph_list, glyph_db, space_weights = None, log_func = None):
  """Layout method that uses medians with weighted values, where the weights are calculated by the similarity to the scenario."""
  
  # Create the spacing model...
  spacing = Spacing(glyph_db)
  if space_weights!=None:
    spacing.set_weights(*space_weights)
  
  # Iterate and process each letter in turn...
  ret = []
  offset = 0.0
    
  for i, glyph in enumerate(glyph_list):
    if glyph==None:
      offset += spacing.median_space()
      ret.append(None)
      
    else:
      # Do this letter...
      hg = numpy.eye(3, dtype=numpy.float32)
      hg[0,2] = offset - glyph.left_x
      ret.append((hg, glyph))
      
      offset += glyph.right_x - glyph.left_x
      
      next_glyph = glyph_list[i+1] if i+1<len(glyph_list) else None
      if next_glyph!=None:
        offset += spacing.median(glyph, next_glyph)

  return ret



def layout_draw(glyph_list, glyph_db, space_weights = None, amount = 0.5, log_func = None):
  """Layout method that uses medians with weighted values, where the weights are calculated by the similarity to the scenario."""
  
  # Create the spacing model...
  spacing = Spacing(glyph_db)
  if space_weights!=None:
    spacing.set_weights(*space_weights)
  
  # Iterate and process each letter in turn...
  ret = []
  offset = 0.0
    
  for i, glyph in enumerate(glyph_list):
    if glyph==None:
      offset += spacing.draw_space(amount)
      ret.append(None)
      
    else:
      # Do this letter...
      hg = numpy.eye(3, dtype=numpy.float32)
      hg[0,2] = offset - glyph.left_x
      ret.append((hg, glyph))
      
      offset += glyph.right_x - glyph.left_x
      
      next_glyph = glyph_list[i+1] if i+1<len(glyph_list) else None
      if next_glyph!=None:
        offset += spacing.draw(glyph, next_glyph, amount)

  return ret



def layout_flow(layout, original_sd = 10.0, offset_sd = 5.0, use_rf = False, comp_punctuation = False):
  """Given a layout (a list of (homography, glyph) in writting order) this tweaks the height of each symbol to make for a better flow of the glyphs when they are all joined up - uses gaussian BP. Returns the replacement layout. Parameters are in the space of the input, the first being the unary terms strength the second the pairwise term. If use_rf is true it will use a random forest to guess the pairwise offset for glyphs that are not joined up."""
  
  # Get center_y values for each glyph...
  y_offset = numpy.zeros(len(layout), dtype=numpy.float32)
  for i in xrange(len(layout)):
    if layout[i]!=None:
      cx, cy = layout[i][1].get_center()
      y_offset[i] = cy
  
  # Setup the GBP object - we are inferring the vertical offset to apply *after* the homography so the unary terms are all the same...
  solver = GBP(len(layout))
  solver.unary(slice(None), y_offset, original_sd**(-2.0))
  
  # Add in the pairwise terms - they only exist when we have ligatures on both of the glyphs (We have softening when the ligatures match poorly, as we get two estimates and add the difference between them to the sd)...
  for i in xrange(len(layout)-1):
    if layout[i]==None or layout[i+1]==None:
      continue
  
    l_hg, l_glyph = layout[i]
    r_hg, r_glyph = layout[i+1]
    
    mean_sd = costs.glyph_pair_offset(l_glyph, r_glyph, offset_sd, use_rf)
    if mean_sd!=None:
      solver.pairwise(i, i+1, mean_sd[0], mean_sd[1]**(-2.0))

  # Solve for the offsets...
  solver.solve()
  
  # Duplicate the input and update the homographies with the offsets from the model...
  ret = list(layout)
  
  for i in xrange(len(layout)):
    if ret[i]!=None:
      offset = numpy.eye(3, dtype=numpy.float32)
      offset[1,2] += solver.result(i)[0] - y_offset[i]
      if comp_punctuation:
        offset[1,2] += ret[i][1].get_voffset()
      ret[i] = (offset.dot(ret[i][0]), ret[i][1])
  
  # Return the new model...
  return ret



def stitch_noop(glyph_layout):
  """Converts a glyph layout to a linegraph layout. This is usually done at the same time as stitching together glyphs to make joined up writing, but this version doesn't do that."""
  ret = []
  for pair in glyph_layout:
    if pair!=None:
      hg, glyph = pair
      ret.append((hg, glyph.lg))
  return ret



def stitch_connect(glyph_layout, soft = True, half = False, pair_base = 0):
  """Converts a glyph layout to a linegraph layout. This stitches together the glyphs when it has sufficient information to do so."""
  ret = []
  
  # First copy over the actual glyphs...
  for pair in glyph_layout:
    if pair!=None:
      hg, glyph = pair
      ret.append((hg, glyph.lg))
      
  # Now loop through and identify all pairs that can be stitched together, and stitch them...
  pair_code = 0
  for i in xrange(len(glyph_layout)-1):
    # Can't stitch spaces...
    if glyph_layout[i]!=None and glyph_layout[i+1]!=None:
      l_hg, l_glyph = glyph_layout[i]
      r_hg, r_glyph = glyph_layout[i+1]
      
      matches = costs.match_links(l_glyph, r_glyph)
        
      # Iterate and do each pairing in turn...
      for ml, mr in matches:
        # Calculate the homographies to put the two line graphs into position...
        lc_hg = numpy.dot(l_hg, numpy.dot(l_glyph.transform, la.inv(ml[0].transform)))
        rc_hg = numpy.dot(r_hg, numpy.dot(r_glyph.transform, la.inv(mr[0].transform)))

        # Copy the links, applying the homographies...
        lc = LineGraph()
        lc.from_many(lc_hg, ml[0].lg)
        
        rc = LineGraph()
        rc.from_many(rc_hg, mr[0].lg)
          
        # Extract the merge points...
        blend = [(ml[3], 0.0, mr[4]), (ml[4], 1.0, mr[3])]
          
        # Do the blending...
        lc.blend(rc, blend, soft)
          
        # Record via tagging that the two parts are the same entity...
        pair = 'duplicate:%i,%i' % (pair_base, pair_code)
        lc.add_tag(0, 0.5, pair)
        rc.add_tag(0, 0.5, pair)
        pair_code += 1
        
        # Store the pair of line graphs in the return, with identity homographies...
        ret.append((numpy.eye(3), lc))
        if not half: ret.append((numpy.eye(3), rc))
  
  return ret



def combine_seperate(lg_layout):
  """Given a line graph layout (List of (homography, line graph) pairs) this merges them all together into a single LineGraph. This version doesn't do anything clever."""
  args = []
  for hg, lg in lg_layout:
    args.append(hg)
    args.append(lg)
  
  ret = LineGraph()
  ret.from_many(*args)
  return ret



def render(lg, border = 8, textures = TextureCache(), cleverness = 0, radius_growth = 3.0, stretch_weight = 0.5, edge_weight = 0.5, smooth_weight = 2.0, alpha_weight = 1.0, unary_mult = 1.0, overlap_weight = 0.0, use_linear = True):
  """Given a line_graph this will render it, returning a numpy array that represents an image (As the first element in a tuple - second element is how many graph cut problems it solved.). It will transform the entire linegraph to obtain a suitable border. The cleverness parameter indicates how it merges the many bits - 0 means last layer (stupid), 1 means averaging; 2 selecting a border using max flow; 3 using graph cuts to take into account weight as well."""

  # Setup the compositor...
  comp = Composite()
  min_x, max_x, min_y, max_y = lg.get_bounds()
  
  do_transform = False
  offset_x = 0.0
  offset_y = 0.0
  
  if min_x<border:
    do_transform = True
    offset_x = border-min_x
    
  if min_y<border:
    do_transform = True
    offset_y = border-min_y
  
  if do_transform:
    hg = numpy.eye(3, dtype=numpy.float32)
    hg[0,2] = offset_x
    hg[1,2] = offset_y
    
    lg.transform(hg)
    
    max_x += offset_x
    max_y += offset_y
  
  comp.set_size(int(max_x+border), int(max_y+border))


  # Break the lg into segments, as each can have its own image - draw & paint each in turn...
  lg.segment()
  duplicate_sets = dict()

  for s in xrange(lg.segments):

    slg = LineGraph()
    slg.from_segment(lg, s)
    part = comp.draw_line_graph(slg, radius_growth, stretch_weight)
    
    done = False
    fn = filter(lambda t: t[0].startswith('texture:'), slg.get_tags())
    if len(fn)!=0: fn = fn[0][0][len('texture:'):]
    else: fn = None
    
    for pair in filter(lambda t: t[0].startswith('duplicate:'), slg.get_tags()):
      key = pair[0][len('duplicate:'):]
      if key in duplicate_sets: duplicate_sets[key].append(part)
      else: duplicate_sets[key] = [part]
    
    tex = textures[fn]
    
    if tex!=None:
      if use_linear:
        comp.paint_texture_linear(tex, part)
      else:
        comp.paint_texture_nearest(tex, part)
      done = True
    
    if not done:
      comp.paint_test_pattern(part)

  
  # Bias towards pixels that are opaque...
  comp.inc_weight_alpha(alpha_weight)
  
  # Arrange for duplicate pairs to have complete overlap, by adding transparent pixels, so graph cuts doesn't create a feather effect...
  if overlap_weight>1e-6:
    for values in duplicate_sets.itervalues():
      for i, part1 in enumerate(values):
        for part2 in values[i:]:
          comp.draw_pair(part1, part2, overlap_weight)
  
  # If requested use maxflow to find optimal cuts, to avoid any real blending...
  count = 0
  if cleverness==2:
    count = comp.maxflow_select(edge_weight, smooth_weight)
  elif cleverness==3:
    count = comp.graphcut_select(edge_weight, smooth_weight, unary_mult)
  
  if cleverness==0:
    render = comp.render_last()
  else:
    render = comp.render_average()

  # Return the rendered image (If cleverness==0 this will actually do some averaging, otherwise it will just create an image)...
  return render, count
