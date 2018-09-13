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

import numpy
import numpy.linalg as la

from collections import defaultdict

import costs

from ply2 import ply2
from line_graph.line_graph import LineGraph



def gen_bias(lg, hg):
  """Helper function - creates and returns a bias object for use when creating Glyphs. Basically weights each line with the amount of ink on it, so when a writter uses every other line it strongly biases towards letters being assigned to the lines they wrote on."""
  bias = defaultdict(float)
  
  # Transform the line graph to line space...
  ls_lg = LineGraph()
  ls_lg.from_many(lg)
  
  ihg = la.inv(hg)
  ls_lg.transform(ihg, True)
  
  # Add weight from all of the line segments...
  for ei in xrange(ls_lg.edge_count):
    edge = ls_lg.get_edge(ei)
    
    vf = ls_lg.get_vertex(edge[0])
    vt = ls_lg.get_vertex(edge[1])
    
    dx = vt[0] - vf[0]
    dy = vt[1] - vf[1]
    
    mass = (vf[5] + vt[5]) * numpy.sqrt(dx*dx + dy*dy)
    line = int(numpy.floor(0.5 * (vt[1] + vf[1])))
    
    bias[line] += mass
  
  # Normalise and return...
  maximum = max(bias.values())
  
  for key in bias.keys():
    bias[key] /= maximum
  
  return bias



class Glyph:
  """Represents a glyph, that has been transformed into a suitable coordinate system; includes connectivity information."""
  def __init__(self, lg, seg, hg, extra = 0.4, bias = None):
    """Given a segmented LineGraph and segment number this extracts it, transforms it into the standard coordinate system and stores the homography used to get there. (hg transforms from line space, where there is a line for each y=integer, to the space of the original pixels.) Also records its position on its assigned line and line number so it can be ordered suitably. Does not store connectivity information - that is done later. extra is used for infering the line position, and is extra falloff to have either side of a line voting for it - a smoothing term basically. bias is an optional dictionary indexed by line number that gives a weight to assign to being assigned to that line - used to utilise the fact that data collection asks the writter to use every-other line, which helps avoid misassigned dropped j's for instance."""
    if lg is None: return

    # Extract the line graph...
    self.lg = LineGraph()
    self.adjacent = self.lg.from_segment(lg, seg)
    self.seg = seg
    
    # Tranform it to line space...
    ihg = la.inv(hg)
    self.lg.transform(ihg, True)
    
    # Check if which line its on is tagged - exists as an override for annoying glyphs...
    line = None
    for tag in self.lg.get_tags():
      if tag[0]=='line':
        # We have a tag of line - its position specifies the line the glyph is on...
        point = self.lg.get_point(tag[1], tag[2])
        line = int(numpy.floor(point[1]))
        break
    
    # Record which line it is on and its position along the line...
    # (Works by assuming that the line is the one below the space where most of the mass of the glyph is. Takes it range to be that within the space, so crazy tails are cut off.)
    min_x, max_x, min_y, max_y = self.lg.get_bounds()
    self.source = (0.5 * (min_x + max_x), 0.5 * (min_y + max_y))
    
    if line is None:   
      best_mass = 0.0
      self.left_x = min_x
      self.right_x = max_x
      line = 0
    
      start = int(numpy.trunc(min_y))
      for pl in xrange(start, int(numpy.ceil(max_y))):
        mass = 0.0
        low_y = float(pl) - extra
        high_y = float(pl+1) + extra
      
        left_x = None
        right_x = None
      
        for es in self.lg.within(min_x, max_x, low_y, high_y):
          for ei in xrange(*es.indices(self.lg.edge_count)):
            edge = self.lg.get_edge(ei)
            vf = self.lg.get_vertex(edge[0])
            vt = self.lg.get_vertex(edge[1])
          
            if vf[1]>low_y and vf[1]<high_y and vt[1]>low_y and vt[1]<high_y:
              dx = vt[0] - vf[0]
              dy = vt[1] - vf[1]
              mass += (vf[5] + vt[5]) * numpy.sqrt(dx*dx + dy*dy)
            
              if left_x is None: left_x = min(vf[0], vt[0])
              else: left_x = min(vf[0], vt[0], left_x)
            
              if right_x is None: right_x = max(vf[0], vt[0])
              else: right_x = max(vf[0], vt[0], right_x)
      
        mass *= 1.0/(1.0+pl - start) # Bias to choosing higher, for tails.
        
        if bias is not None:
          mass *= bias[pl]
      
        if mass>best_mass:
          best_mass = mass
          self.left_x = left_x
          self.right_x = right_x
          line = pl
    
    # Transform it so it is positioned to be sitting on line 1 of y, store the total homography that we have applied...
    self.offset_x = -min_x
    self.offset_y = -line
    
    hg = numpy.eye(3, dtype=numpy.float32)
    hg[0,2] = self.offset_x
    hg[1,2] = self.offset_y
    
    self.left_x += self.offset_x
    self.right_x += self.offset_x
    
    self.lg.transform(hg)
    
    self.transform = numpy.dot(hg, ihg)
    
    # Set as empty its before and after glyphs - None if there is no adjacency, or a tuple if there is: (glyph, list of connecting (link glyph, shared vertex in this, shared vertex in glyph, vertex in link glyph on this side, vertex in link glyph on glyph side), empty if none.)...
    self.left = None
    self.right = None
    
    # Extract the character this glyph represents...
    tags = self.lg.get_tags()
    codes = [t[0] for t in tags if len(filter(lambda c: c!='_', t[0]))==1]
    self.key = codes[0] if len(codes)!=0 else None
    
    self.code = -id(self)
    
    # Cache stuff...
    self.mass = None
    self.center = None
    self.feat = None
    self.v_offset = None


  def clone(self):
    """Returns a clone of this Glyph."""
    ret = Glyph(None, None, None)
    
    ret.lg = self.lg
    ret.adjacent = self.adjacent
    ret.seg = self.seg
    
    ret.source = self.source
    
    ret.left_x = self.left_x
    ret.right_x = self.right_x
    
    ret.offset_x = self.offset_x
    ret.offset_y = self.offset_y
    ret.transform = self.transform
    
    ret.left = self.left
    ret.right = self.right
    ret.key = self.key
    
    ret.code = self.code
    
    ret.mass = None if self.mass is None else self.mass.copy()
    ret.center = None if self.center is None else self.center.copy()
    ret.feat = None if self.feat is None else map(lambda a: a.copy(), self.feat)
    ret.v_offset = self.v_offset
    
    return ret
    
    
  def get_linegraph(self):
    return self.lg
  
  
  def orig_left_x(self):
    return self.left_x - self.offset_x
  
  def orig_right_x(self):
    return self.right_x - self.offset_x

  
  def get_mass(self):
    """Returns a vector of [average density, average radius] - used for matching adjacent glyphs."""
    if self.mass is None:
      self.mass = numpy.zeros(2, dtype=numpy.float32)
      weight = 0.0
      for i in xrange(self.lg.vertex_count):
        info = self.lg.get_vertex(i)
        
        weight += 1.0
        self.mass += (numpy.array([info[6], info[5]]) - self.mass) / weight
    
    return self.mass


  def get_center(self):
    """Returns the 'center' of the glyph - its density weighted in an attempt to make it robust to crazy tails."""
    if self.center is None:
      self.center = numpy.zeros(2, dtype=numpy.float32)
      weight = 0.0
    
      for i in xrange(self.lg.vertex_count):
        info = self.lg.get_vertex(i)
        w = info[5] * info[5] * info[6] # Radius squared * density - proportional to quantity of ink, assuming (correctly as rest of system currently works) even sampling.
        if w>1e-6:
          weight += w
          mult = w / weight
          self.center[0] += (info[0] - self.center[0]) * mult
          self.center[1] += (info[1] - self.center[1]) * mult
    
    return self.center
  
  
  def get_voffset(self):
    """Calculates and returns the vertical offset to apply to the glyph that corrects for any systematic bias in its flow calculation."""
    
    if self.v_offset is None:
      self.v_offset = 0.0
      weight = 0.0
      
      truth = self.get_center()[1]
      
      # Calculate the estimated offsets from the left side and update the estimate, correctly factoring in the variance of the offset...
      if self.left is not None:
        diff, sd = costs.glyph_pair_offset(self.left[0], self, 0.2, True)
        estimate = self.left[0].get_center()[1] + diff
        offset = truth - estimate
        
        est_weight = 1.0 / (sd**2.0)
        weight += est_weight
        self.v_offset += (offset - self.v_offset) * est_weight / weight
    
      # Again from the right side...
      if self.right is not None:
        diff, sd = costs.glyph_pair_offset(self, self.right[0], 0.2, True)
        estimate = self.right[0].get_center()[1] - diff
        offset = truth - estimate
        
        est_weight = 1.0 / (sd**2.0)
        weight += est_weight
        self.v_offset += (offset - self.v_offset) * est_weight / weight
  
    return self.v_offset


  def most_left(self):
    """Returns the coordinate of the furthest left vertex in the glyph."""
    
    info = self.lg.get_vertex(0)
    best_x = info[0]
    best_y = info[1]
    
    for i in xrange(1,self.lg.vertex_count):
      info = self.lg.get_vertex(0)
      if info[0]<best_x:
        best_x = info[0]
        best_y = info[1]
    
    return (best_x, best_y)

  def most_right(self):
    """Returns the coordinate of the furthest right vertex in the glyph."""
    
    info = self.lg.get_vertex(0)
    best_x = info[0]
    best_y = info[1]
    
    for i in xrange(1,self.lg.vertex_count):
      info = self.lg.get_vertex(0)
      if info[0]>best_x:
        best_x = info[0]
        best_y = info[1]
    
    return (best_x, best_y)
  
  
  def get_feat(self):
    """Calculates and returns a feature for the glyph, or, more accuratly two features, representing (left, right), so some tricks can be done to make their use side dependent (For learning a function for matching to adjacent glyphs.)."""
    if self.feat is None:
      # First build a culumative distribution over the x axis range of the glyph...
      min_x, max_x, min_y, max_y = self.lg.get_bounds()
      culm = numpy.ones(32, dtype=numpy.float32)
      culm *= 1e-2
      
      min_x -= 1e-3
      max_x += 1e-3
      
      for i in xrange(self.lg.vertex_count):
        info = self.lg.get_vertex(i)
        w = info[5] * info[5] * info[6]
        t = (info[0] - min_x) / (max_x - min_x)
        
        t *= (culm.shape[0]-1)
        low = int(t)
        high = low + 1
        t -= low
        culm[low] += (1.0 - t) * w
        culm[high] += t * w
      
      culm /= culm.sum()
      culm = numpy.cumsum(culm)
      
      # Now extract all the per sample features...
      feat_param = {'dir_travel':0.1, 'travel_max':1.0, 'travel_bins':6, 'travel_ratio':0.8, 'pos_bins':3, 'pos_ratio':0.9, 'radius_bins':1, 'density_bins':3}
      fv = self.lg.features(**feat_param)
      
      # Combine them into the two halves, by weighting by the culumative; include density and radius as well...
      left = numpy.zeros(fv.shape[1]+2, dtype=numpy.float32)
      right = numpy.zeros(fv.shape[1]+2, dtype=numpy.float32)
      
      left_total = 0.0
      right_total = 0.0
      
      for i in xrange(self.lg.vertex_count):
        info = self.lg.get_vertex(i)
        w = info[5] * info[5] * info[6]
        t = (info[0] - min_x) / (max_x - min_x)
        
        t *= (culm.shape[0]-1)
        low = int(t)
        high = low + 1
        t -= low
        right_w = (1.0-t) * culm[low] + t * culm[high]
        left_w = 1.0 - right_w
        
        left[0] += left_w * info[5]
        right[0] += right_w * info[5]
        
        left[1] += left_w * info[6]
        right[1] += right_w * info[6]
        
        left[2:] += w * left_w * fv[i,:]
        right[2:] += w * right_w * fv[i,:]
        
        left_total += left_w
        right_total += right_w
      
      left[:2] /= left_total
      right[:2] /= right_total
      left[2:] /= max(left[2:].sum(), 1e-6)
      right[2:] /= max(right[2:].sum(), 1e-6)
      
      self.feat = (left, right)
    
    return self.feat


  def __str__(self):
    l = self.left[0].key if self.left is not None else 'None'
    r = self.right[0].key if self.right is not None else 'None'
    return 'Glyph %i: key = %s (%s|%s)' % (self.code, self.key, l, r)



class GlyphDB:
  """A database of glyphs, as extracted from tagged text - typically you load in a bunch of LineGraph objects (It gets the filename from the meta data) and then you have a list of linegraphs for each character it has obtained from the line graph. Includes adjacency information, so you can fetch the adjacent segments, to get the connecting bits. Each segment comes such that it sits on the y=1 line, with the next line up at y=0. Adjacency information includes the homography required to put the adjacent segment into the same coordinate system (Due to the way it works its often the identity however)."""
  def __init__(self):
    self.db = dict() # Lists of glyphs, indexed by their token (e.g. 'a', '_B'...).
    self.fnl = [] # List of filenames that glyphs have been loaded from.
    
    self.by_code = [] # List of glyphs, for unique access - index into list is stored in the glyph itself.
    
  
  def empty(self):
    """Returns True if there is nothing in the db."""
    return len(self.db)==0
    
    
  def add(self, fn):
    """Given the filename for a LineGraph file this loads it in and splits it into Glyphs, which it dumps into the db. Does nothing if the file is already loaded. returns the number of glyphs added."""
    
    if fn in self.fnl: return 0
    self.fnl.append(fn)
    
    # Load the LineGraph from the given filename, and get the homography...
    f = open(fn, 'r')
    data = ply2.read(f)
    f.close()
    
    lg = LineGraph()
    lg.from_dict(data)
    lg.segment()
    
    hg = data['element']['homography']['v'].reshape((3,3))
    
    texture = os.path.normpath(os.path.join(os.path.dirname(fn), data['meta']['image']))
    
    
    # Create a line bias object to be used in the next step...
    bias = gen_bias(lg, hg)
    
    # First pass - create each glyph object...
    glyphs = []
    for s in xrange(lg.segments):
      g = Glyph(lg, s, hg, bias = bias)
      glyphs.append(g if '_' not in map(lambda t: t[0], g.lg.get_tags()) else None)


    # Second pass - fill in the connectivity information supported by adjacency...
    link_glyphs = []
    for seg, glyph in enumerate(glyphs):
      if glyph is None: continue
      if glyph.key is None: continue
      
      # Brute force search to find a left partner...
      if glyph.left is None:
        best = None
        for g in glyphs:
          # Check it satisfies the conditions...
          if g is None: continue
          if g.key is None: continue;
          if id(g)==id(glyph): continue
          if g.offset_y!=glyph.offset_y: continue
          if (g.right_x - g.offset_x) > (glyph.right_x - glyph.offset_x): continue
          
          # Check its better than the current best...
          if best is None or (best.right_x - best.offset_x) < (g.right_x - g.offset_x):
            best = g
        
        if best is not None:
          glyph.left = (best, [])
      
      # Brute force search to find a right partner...
      if glyph.right is None:
        best = None
        for g in glyphs:
          # Check it satisfies the conditions...
          if g is None: continue
          if g.key is None: continue;
          if id(g)==id(glyph): continue
          if g.offset_y!=glyph.offset_y: continue
          if (g.left_x - g.offset_x) < (glyph.left_x - glyph.offset_x): continue
          
          # Check its better than the current best...
          if best is None or (best.left_x - best.offset_x) > (g.left_x - g.offset_x):
            best = g
        
        if best is not None:
          glyph.right = (best, [])


      # Now we have the best matches find common glyphs to link them, and record them...
      for other, out in [g for g in [glyph.left, glyph.right] if g is not None]:
        shared_seg = set([a[1] for a in glyph.adjacent]) & set([a[1] for a in other.adjacent])
        
        for seg in shared_seg:
          g = glyphs[seg]
          if g is None: continue
          
          # We have a linking glyph - extract the information...
          glyph_vert = [a[0] for a in glyph.adjacent if a[1]==seg]
          other_vert = [a[0] for a in other.adjacent if a[1]==seg]
          
          link_glyph = [a[0] for a in g.adjacent if a[1]==glyph.seg]
          link_other = [a[0] for a in g.adjacent if a[1]==other.seg]
          
          # Check if we have a multi-link scenario - if so choose links...
          if len(glyph_vert)>1 or len(other_vert)>1:
            gv_y = map(lambda v: glyph.lg.get_vertex(v)[1], glyph_vert)
            ov_y = map(lambda v: other.lg.get_vertex(v)[1], other_vert)
            
            if (max(gv_y) - min(ov_y)) > (max(ov_y) - min(gv_y)):
              glyph_vert = glyph_vert[numpy.argmax(gv_y)]
              other_vert = other_vert[numpy.argmin(ov_y)]
            else:
              glyph_vert = glyph_vert[numpy.argmin(gv_y)]
              other_vert = other_vert[numpy.argmax(ov_y)]
            
            lg_y = map(lambda v: g.lg.get_vertex(v)[1], link_glyph)
            lo_y = map(lambda v: g.lg.get_vertex(v)[1], link_other)

            if (max(lg_y) - min(lo_y)) > (max(lo_y) - min(lg_y)):
              link_glyph = link_glyph[numpy.argmax(lg_y)]
              link_other = link_other[numpy.argmin(lo_y)]
            else:
              link_glyph = link_glyph[numpy.argmin(lg_y)]
              link_other = link_other[numpy.argmax(lo_y)]

          else:
            # Simple scenario...
            glyph_vert = glyph_vert[0]
            other_vert = other_vert[0]
            
            link_glyph = link_glyph[0]
            link_other = link_other[0]
          
          # Recreate the link as a simple path - its the only way to be safe!..
          try:
            g = g.clone()
            nlg = LineGraph()
            link_glyph, link_other = nlg.from_path(g.lg, link_glyph, link_other)
            g.lg = nlg
            link_glyphs.append(g)
          except:
            continue # Line is broken in the centre - don't use it.
          
          # Add the tuple to the storage...
          out.append((g, glyph_vert, other_vert, link_glyph, link_other))


    # Third pass - enforce consistancy...
    for glyph in glyphs:
      if glyph is None: continue
      
      if glyph.left is not None:
        if glyph.left[0].right is None or id(glyph.left[0].right[0])!=id(glyph):
          # Inconsistancy - break it...
          glyph.left[0].right = None
          glyph.left = None
          
      if glyph.right is not None:
        if glyph.right[0].left is None or id(glyph.right[0].left[0])!=id(glyph):
          # Inconsistancy - break it...
          glyph.right[0].left = None
          glyph.right = None
    
    
    # Forth pass - add filename tags and add to db (Count and return the glyph count)...
    count = 0
    
    for glyph in (glyphs+link_glyphs):
      if glyph is None: continue
      
      glyph.lg.add_tag(0, 0.1, 'file:%s'%fn)
      glyph.lg.add_tag(0, 0.2, 'texture:%s'%texture)
      
      glyph.lg.add_tag(0, 0.3, 'link:left:%s'%('n' if glyph.left is None else ('g' if len(glyph.left[1])==0 else 'l')))
      glyph.lg.add_tag(0, 0.4, 'link:right:%s'%('n' if glyph.right is None else ('g' if len(glyph.right[1])==0 else 'l')))
      
      glyph.lg.add_tag(0, 0.5, 'source:x:%f' % glyph.source[0])
      glyph.lg.add_tag(0, 0.6, 'source:y:%f' % glyph.source[1])
      
      if glyph.key is not None:
        count += 1
        if glyph.key in self.db: self.db[glyph.key].append(glyph)
        else: self.db[glyph.key] = [glyph]
      
      glyph.code = len(self.by_code)
      glyph.lg.add_tag(0, 0.5, 'code:%i'%glyph.code)
      self.by_code.append(glyph)
    
    return count


  def rem(self, fn):
    """Given a filename this removes all Glyphs that were loaded from that file."""
    self.fnl = filter(lambda f: f!=fn, self.fnl)
    
    def die_glyph_die(glyph):
      """Returns False to kill the glyph, True to keep it. Removes from glyph_db when it returns False"""
      tags = glyph.get_linegraph().get_tags()
      for tag in tags:
        if tag[0]==('file:%s'%fn):
          for t in filter(lambda t: t[0][:5]=='code:', tags):
            self.by_code[int(t[0][5:])] = None
          return False
      return True
    
    for key in self.db.iterkeys():
      self.db[key] = filter(die_glyph_die, self.db[key])


  def filenames(self):
    """Returns a list of all loaded files."""
    return self.fnl
  
  
  def get_single(self, code):
    """Given a code for a glyph (Stored within using a tag 'code:<integer>') this returns the actual glyph object, or None if it doesn't exist/has been deleted from the db."""
    if code<len(self.by_code):
      return self.by_code[code]
    else:
      return None
  
  def get_all(self):
    """Returns a list of all the glyphs."""
    ret = []
    
    for value in self.db.itervalues():
      ret += value
    
    return ret


  def glyph(self, char):
    """Given a character this returns all entrys of that glyph in the db; can be an empty list. Will include versions that start/end with a _, i.e. have a space before/after."""
    ret = []
    
    keys = [char, '_'+char, char+'_', '_'+char+'_']
    for key in keys:
      if key in self.db:
        ret += self.db[key]
    
    return ret


  def topup_glyph(self, key, minimum):
    """Given a key, including '_' to indicate where spaces go, this returns a list of matching glyphs. In the event it cant get the minimum (parameter) number it tops up using samples with different spacing; if that is not enough it considers alternate case."""
    ret = []
    char = filter(lambda c: c!='_', key)
    
    # What we want...
    if key in self.db: ret += self.db[key]
    
    # Alternate spacing...
    alt = [char, '_'+char, char+'_', '_'+char+'_']
    for a in alt:
      if len(ret)<minimum and a!=key and a in self.db:
        ret += self.db[a]

    # If its uppercase consider substituting a lowercase...
    if len(ret)<minimum and char in string.uppercase:
      ret += self.topup_glyph(key.lower(), minimum-len(ret))
    
    return ret


  def stats(self):
    """Returns a dictionary of statistics about the data within - indexed by each char to a 4-tuple of counts, of how many of char c exist of the following form: (c, _c, c_, _c_)"""
    ret = dict()
    
    for key, value in self.db.iteritems():
      true_key = filter(lambda c: c!='_', key)
      
      if true_key not in ret: ret[true_key] = [0,0,0,0]
      
      index = 0
      if len(key)==3: index = 3
      elif key[0]=='_': index = 1
      elif key[-1]=='_': index = 2
      
      ret[true_key][index] += len(value)
    
    return ret
  
  
  def diff(self, char_left, char_right, space = False):
    """Returns a list of all the differences between characters that appear in the database, such that they satisfy the provided conditions - the character on the left is in char_left, the character on the right is in char_right, and if space is True at least one of them will be marked as having a space in the required direction, whilst False and neither can."""
    ret = []
    for key, value in self.db.iteritems():
      space_left = key.endswith('_')
      if space==False and space_left==True:
        continue

      if filter(lambda c: c!='_', key) in char_left:
        
        for left_glyph in value:
          if left_glyph.right is not None:
            right_glyph = left_glyph.right[0]
            if filter(lambda c: c!='_', right_glyph.key) in char_right:
              
              space_right = right_glyph.key.startswith('_')
              if space==False and space_right==True:
                continue
              if space==True and space_left==False and space_right==False:
                continue
              
              left_pos = left_glyph.right_x - left_glyph.offset_x
              right_pos = right_glyph.left_x - right_glyph.offset_x
              delta = right_pos - left_pos
              ret.append(delta)
      
    return ret
