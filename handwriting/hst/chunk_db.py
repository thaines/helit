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

import numpy
import random
import scipy.spatial

from ply2 import ply2
from line_graph.line_graph import LineGraph

from texture_cache import TextureCache

from graph_cuts import maxflow # Not actually needed by this module, but composite uses it from c, and can't compile it if its not up-to-date - doing this makes sure it is.
from composite import Composite



class ChunkDB:
  """Given line graphs this chops them up and stores all the resulting chunks; it then has the ability to replace a line graph with chunks from its database. Basically does line replacement, so you can swap e.g. pen for crayon."""
  def __init__(self):
    self.chunks = [] # List of chunks, as (line graph, median radius)
    self.kdtree = None # So we can quickly find chunks that match a given chunk.
    
    self.fnl = [] # List of filenames from which chunks have been loaded from, to avoid duplication.
    
    self.dist = 16.0 / 5.0 # Length it aims to make the chunks, noting that we measure length as multiples of the typical radius for a line segment in the source data set, to normalise.
    self.factor = 0.5 # Step size in chunk length, to control overlap.
    
    self.samples = 8 # Number of bins to sample density/radius/orientation when making feature vectors.
    self.radius_mult = 1.0
    self.density_mult = 1.0
  
  
  def feature_vect(self, chunk, median_radius):
    """Given a line graph representing a chunk returns a 1D numpy array that is used as a feature vector."""
    return chunk.chain_feature(self.samples, self.radius_mult / median_radius, self.density_mult)
  
  
  def empty(self):
    """Returns True if there is nothing in the db."""
    return len(self.chunks)==0

  
  def set_params(self, samples = 8, angle_weight=1.0, radius_weight=1.0, density_weight=1.0):
    """Sets the chunk matching parameters - note that this resets the KD tree it has to build, so next convert will be computationally expensive."""
    self.samples = samples
    self.radius_mult = radius_weight / angle_weight
    self.density_mult = density_weight / angle_weight
    
    self.kdtree = None

  
  def add(self, fn):
    """Given a filename for a linegraph file this loads it, extracts all chunks and stores them in the db."""
    if fn in self.fnl: return 0
    self.fnl.append(fn)
    self.kdtree = None
    
    # Load the LineGraph from the given filename...
    data = ply2.read(fn)
    
    lg = LineGraph()
    lg.from_dict(data)
    
    texture = os.path.normpath(os.path.join(os.path.dirname(fn), data['meta']['image']))
    
    # Calculate the radius scaler and distance for this line graph, by calculating the median radius...
    rads = map(lambda i: lg.get_vertex(i)[5], xrange(lg.vertex_count))
    rads.sort()
    median_radius = rads[len(rads)//2]
    radius_mult = 1.0 / median_radius
    
    dist = self.dist * median_radius
    
    # Chop it up into chains, extract chunks and store them in the database...
    ret = 0
    
    for raw_chain in lg.chains():
      for chain in filter(lambda c: len(c)>1, [raw_chain, raw_chain[::-1]]):
        head = 0
        tail = 0
        length = 0.0
        
        while True:
          # Move tail so its long enough, or has reached the end...
          while length<dist and tail+1<len(chain):
            tail += 1
            v1 = lg.get_vertex(chain[tail-1])
            v2 = lg.get_vertex(chain[tail])
            length += numpy.sqrt((v1[0]-v2[0])**2 + (v1[1]-v2[1])**2)
          
          # Create the chunk...
          chunk = LineGraph()
          chunk.from_vertices(lg, chain[head:tail+1])
          
          # Tag it...
          chunk.add_tag(0, 0.1, 'file:%s'%fn)
          chunk.add_tag(0, 0.2, 'texture:%s'%texture)
          
          # Store it...
          self.chunks.append((chunk, median_radius))
          ret += 1
          
          # If tail is at the end exit the loop...
          if tail+1 >= len(chain): break
          
          # Move head along for the next chunk...
          to_move = dist * self.factor
          while to_move>0.0 and head+2<len(chain):
            head += 1
            v1 = lg.get_vertex(chain[head-1])
            v2 = lg.get_vertex(chain[head])
            offset = numpy.sqrt((v1[0]-v2[0])**2 + (v1[1]-v2[1])**2)
            length -= offset
            to_move -= offset
            
    return ret
  
  
  def rem(self, fn):
    """Given a filename removes all chunks that were extracted from it from the database."""
    if fn in self.fnl:
      self.fnl = filter(lambda f: f!=fn, self.fnl)
      self.kdtree = None
      
      def die_chunk_die(c):
        """Returns False to kill the chunk, True to keep it"""
        tags = c[0].get_tags()
        for tag in tags:
          if tag[0]==('file:%s'%fn): return False
        return True
      
      self.chunks = filter(die_chunk_die, self.chunks)


  def filenames(self):
    """Returns a list of all loaded files."""
    return self.fnl
  
  
  def convert(self, lg, choices = 1, adv_match = False, textures = TextureCache(), memory = 0):
    """Given a line graph this chops it into chunks, matches each chunk to the database of chunks and returns a new line graph with these chunks instead of the original. Output will involve heavy overlap requiring clever blending. choices is the number of options it select from the db - it grabs this many closest to the requirements and then randomly selects from them. If adv_match is True then instead of random selection from the choices it does a more advanced match, and select the best match in terms of colour distance from already-rendered chunks. This option is reasonably expensive. memory is how many recently use chunks to remember, to avoid repetition."""
    if memory > (choices - 1):
      memory = choices - 1

    # If we have no data just return the input...
    if self.empty(): return lg
    
    # Check if the indexing structure is valid - if not create it...
    if self.kdtree==None:
      data = numpy.array(map(lambda p: self.feature_vect(p[0], p[1]), self.chunks), dtype=numpy.float)
      self.kdtree = scipy.spatial.cKDTree(data, 4)
      
    # Calculate the radius scaler and distance for this line graph, by calculating the median radius...
    rads = map(lambda i: lg.get_vertex(i)[5], xrange(lg.vertex_count))
    rads.sort()
    median_radius = rads[len(rads)//2]
    radius_mult = 1.0 / median_radius
    
    dist = self.dist * median_radius
    
    # Create the list into which we dump all the chunks that will make up the return...
    chunks = []
    temp = LineGraph()
    
    # List of recently used chunks, to avoid obvious patterns...
    recent = []
    
    # If advanced match we need a Composite of the image thus far, to compare against...
    if adv_match:
      canvas = Composite()
      min_x, max_x, min_y, max_y = lg.get_bounds()
      canvas.set_size(int(max_x+8), int(max_y+8))
    
    # Iterate the line graph, choping it into chunks and matching a chunk to each chop...
    for chain in lg.chains():
      head = 0
      tail = 0
      length = 0.0
        
      while True:
        # Move tail so its long enough, or has reached the end...
        while length<dist and tail+1<len(chain):
          tail += 1
          v1 = lg.get_vertex(chain[tail-1])
          v2 = lg.get_vertex(chain[tail])
          length += numpy.sqrt((v1[0]-v2[0])**2 + (v1[1]-v2[1])**2)

        # Extract a feature vector for this chunk...
        temp.from_vertices(lg, chain[head:tail+1])
        fv = self.feature_vect(temp, median_radius)
        
        # Select a chunk from the database...
        if choices==1:
          selected = self.kdtree.query(fv)[1]
          orig_chunk = self.chunks[selected]
        else:
          options = list(self.kdtree.query(fv, choices)[1])
          options = filter(lambda v: v not in recent, options)
          if not adv_match:
            selected = random.choice(options)
            orig_chunk = self.chunks[selected]
          else:
            cost = 1e64 * numpy.ones(len(options))
            
            for i, option in enumerate(options):
              fn = filter(lambda t: t[0].startswith('texture:'), self.chunks[option][0].get_tags())
              if len(fn)!=0:
                fn = fn[0][0][len('texture:'):]
                tex = textures[fn]
                
                chunk = LineGraph()
                chunk.from_many(self.chunks[option][0])
                chunk.morph_to(lg, chain[head:tail+1])
              
                part = canvas.draw_line_graph(chunk)
                cost[i] = canvas.cost_texture_nearest(tex, part)
            
            selected = options[numpy.argmin(cost)]
            orig_chunk = self.chunks[selected]
        
        # Update recent list...
        recent.append(selected)
        if len(recent)>memory:
          recent.pop(0)

        # Distort it to match the source line graph...
        chunk = LineGraph()
        chunk.from_many(orig_chunk[0])
        chunk.morph_to(lg, chain[head:tail+1])
        
        # Record it for output...
        chunks.append(chunk)
        
        # If advanced matching is on write it out to canvas, so future choices will take it into account...
        if adv_match:
          fn = filter(lambda t: t[0].startswith('texture:'), chunk.get_tags())
          if len(fn)!=0:
            fn = fn[0][0][len('texture:'):]
            tex = textures[fn]

            part = canvas.draw_line_graph(chunk)
            canvas.paint_texture_nearest(tex, part)
         
        # If tail is at the end exit the loop...
        if tail+1 >= len(chain): break
          
        # Move head along for the next chunk...
        to_move = dist * self.factor
        while to_move>0.0 and head+2<len(chain):
          head += 1
          v1 = lg.get_vertex(chain[head-1])
          v2 = lg.get_vertex(chain[head])
          offset = numpy.sqrt((v1[0]-v2[0])**2 + (v1[1]-v2[1])**2)
          length -= offset
          to_move -= offset

    # Return the final line graph...
    ret = LineGraph()
    ret.from_many(*chunks)
    return ret
