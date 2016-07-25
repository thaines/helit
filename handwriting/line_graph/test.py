#! /usr/bin/env python

# Copyright 2016 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

from line_graph import LineGraph

import numpy
from ply2 import ply2

import tempfile
import unittest



class TestLineGraph(unittest.TestCase):
  """Some unit tests of the LineGraph object - hardly exhaustive, but at least makes sure the basics and i/o are not broken."""
  
  def test_new_delete(self):
    lg = LineGraph()
    del lg
  


  def make_circle(self):
    mask = numpy.zeros((512,512), dtype=numpy.bool)
    
    centre = (mask.shape[0]//2, mask.shape[1]//2)
    radius = 128 + 64
    
    # Stupid approach to rendering a circle, but its quick to write...
    for y in xrange(centre[0]-radius, centre[0]+radius+1):
      x_off = numpy.sqrt(radius**2 - (y - centre[0])**2)
      x_off = int(x_off+0.5)
      
      mask[y, centre[1]-x_off] = True
      mask[y, centre[1]+x_off] = True
    
    for x in xrange(centre[1]-radius, centre[1]+radius+1):
      y_off = numpy.sqrt(radius**2 - (x - centre[1])**2)
      y_off = int(y_off+0.5)
      
      mask[centre[0]-y_off, x] = True
      mask[centre[0]+y_off, x] = True

    # Mask to line graph...
    lg = LineGraph()
    lg.from_mask(mask)
    return lg


  def test_from_mask_circle(self):
    lg = self.make_circle()
    
    # A circle should have the same number of vertices and edges...
    self.assertTrue(lg.vertex_count==lg.edge_count)
    
    # Should only have one segment...
    lg.segment()
    self.assertTrue(lg.segments==1)
    
    del lg



  def make_grid(self):
    # Setup a grid, with varying radii....
    mask = numpy.zeros((512,512), dtype=numpy.bool)
    
    mask[:,::32] = True
    mask[::32,:] = True
    
    # Create an arbitrary radius field...
    radius = numpy.empty(mask.shape, dtype=numpy.float32)
    radius[:,:] = 1.5 + numpy.sin(0.1*numpy.arange(radius.shape[0]))[:,None]

    # Mask to line graph...
    lg = LineGraph()
    lg.from_mask(mask, radius)
    
    # Add some splits...
    for y in xrange(0, 512, 32):
      distance, edge_index, edge_t = lg.nearest(240, y)
      lg.add_split(edge_index, edge_t)
    
    # Return...
    return lg


  def test_from_mask_grid(self):
    lg = self.make_grid()
    
    # Should have two segments due to splits...
    lg.segment()
    self.assertTrue(lg.segments==2)
    
    # Check radius is varying...
    radii = numpy.empty(lg.vertex_count, dtype=numpy.float32)
    for i in xrange(radii.shape[0]):
      radii[i] = lg.get_vertex(i)[5]

    self.assertTrue(radii.min()<1.6)
    self.assertTrue(radii.max()>2.4)
    self.assertTrue(radii.mean()>1.4)
    self.assertTrue(radii.mean()<1.6)
    
    del lg
  
  
  
  def make_squares(self):
    # Create 4 squares....
    mask = numpy.zeros((128,128), dtype=numpy.bool)
    
    mask[8:57,8] = True
    mask[8:57,56] = True
    mask[8,8:57] = True
    mask[56,8:57] = True
    
    mask[72:121,8] = True
    mask[72:121,56] = True
    mask[72,8:57] = True
    mask[120,8:57] = True
    
    mask[8:57,72] = True
    mask[8:57,120] = True
    mask[8,72:121] = True
    mask[56,72:121] = True
    
    mask[72:121,72] = True
    mask[72:121,120] = True
    mask[72,72:121] = True
    mask[120,72:121] = True
    
    # Create arbitrary radius and density fields...
    radius = numpy.empty(mask.shape, dtype=numpy.float32)
    radius[:,:] = 1.5 + numpy.sin(0.1*numpy.arange(radius.shape[0]))[:,None]
    
    density = numpy.empty(mask.shape, dtype=numpy.float32)
    density[:,:] = 1.5 + numpy.sin(0.05*numpy.arange(radius.shape[1]))[None,:]
    
    # Mask to line graph...
    lg = LineGraph()
    lg.from_mask(mask, radius, density)
    
    # Tag them...
    distance, edge_index, edge_t = lg.nearest(8, 8)
    lg.add_tag(edge_index, edge_t, 'A')
    
    distance, edge_index, edge_t = lg.nearest(8, 72)
    lg.add_tag(edge_index, edge_t, 'B')
    
    distance, edge_index, edge_t = lg.nearest(72, 8)
    lg.add_tag(edge_index, edge_t, 'C')
    
    distance, edge_index, edge_t = lg.nearest(72, 72)
    lg.add_tag(edge_index, edge_t, 'D')
    
    # Return...
    return lg
    
    
  def test_from_mask_squares(self):
    lg = self.make_squares()
    
    # Segment per square...
    lg.segment()
    self.assertTrue(lg.segments==4)
    
    # Verify we have the right set of tags...
    expected = ['A', 'B', 'C', 'D']
    for i in xrange(lg.segments):
      tags = lg.get_tags(i)
      self.assertTrue(len(tags)==1)
      self.assertTrue(tags[0][0] in expected)
      expected = [t for t in expected if t!=tags[0][0]]
    
    del lg
  
  
  
  def make_text(self):
    # Create something that approximates what real text looks like....
    mask = numpy.zeros((32,100), dtype=numpy.bool)
    
    # Create an 'O'...
    for y in xrange(16-12, 16+12+1):
      x_off = numpy.sqrt(12**2 - (y - 16)**2)
      x_off = int(x_off+0.5)
      
      mask[y, 16-x_off] = True
      mask[y, 16+x_off] = True
    
    for x in xrange(16-12, 16+12+1):
      y_off = numpy.sqrt(12**2 - (x - 16)**2)
      y_off = int(y_off+0.5)
      
      mask[16-y_off, x] = True
      mask[16+y_off, x] = True
    
    # Create an 'i'...
    mask[4:8,36] = True
    mask[12:29,36] = True
    
    # Create a 'n'...
    mask[28,36:53] = True
    mask[12:29,52] = True
    
    for x in xrange(52, 69):
      y = int(16 - 4.0 * numpy.sin(0.25*(x - 52)) + 0.5)
      mask[y,x] = True
    
    mask[20:29,68] = True
    
    # Create a 'k'...
    mask[28,68:85] = True
    mask[4:28,84] = True
    
    for x in xrange(84, 84+12+1):
      os = x- 84
      mask[16+os,x] = True
      mask[16-os,x] = True
    
    # Generate all three maps...
    radius = numpy.empty(mask.shape, dtype=numpy.float32)
    radius[:,:] = numpy.linspace(2.0, 0.5, mask.shape[0])[:,None]
    
    density = numpy.empty(mask.shape, dtype=numpy.float32)
    density[:,:] = numpy.linspace(1.2, 0.8, mask.shape[0])[:,None]
    
    weight = numpy.empty(mask.shape, dtype=numpy.float32)
    weight[:,:] = numpy.linspace(1.0, 0.1, mask.shape[1])[None,:]
    
    # Mask to line graph...
    lg = LineGraph()
    lg.from_mask(mask, radius, density, weight)
    
    # Add splits...
    distance, edge_index, edge_t = lg.nearest(37, 28)
    lg.add_split(edge_index, edge_t)
    
    distance, edge_index, edge_t = lg.nearest(51, 28)
    lg.add_split(edge_index, edge_t)
    
    distance, edge_index, edge_t = lg.nearest(69, 28)
    lg.add_split(edge_index, edge_t)
    
    distance, edge_index, edge_t = lg.nearest(83, 28)
    lg.add_split(edge_index, edge_t)
    
    # Add a link for the 'i'...
    distance, edge_index_a, edge_t_a = lg.nearest(36, 7)
    distance, edge_index_b, edge_t_b = lg.nearest(36, 12)
    lg.add_link(edge_index_a, edge_t_a, edge_index_b, edge_t_b)
    
    # Add tags...
    distance, edge_index, edge_t = lg.nearest(16, 4)
    lg.add_tag(edge_index, edge_t, '_O')
    
    distance, edge_index, edge_t = lg.nearest(36, 4)
    lg.add_tag(edge_index, edge_t, 'i')
    
    distance, edge_index, edge_t = lg.nearest(52, 12)
    lg.add_tag(edge_index, edge_t, 'n')
    
    distance, edge_index, edge_t = lg.nearest(84, 4)
    lg.add_tag(edge_index, edge_t, 'k_')
    
    # Return...
    return lg


  def test_from_mask_text(self):
    lg = self.make_text()
    
    # Extract the letters and ligatures and check its all as expected, with the right tags...
    lg.segment()
    self.assertTrue(lg.segments==6)
    
    for i in xrange(lg.segments):
      tags = lg.get_tags(i)
      adj = lg.adjacent(i)
      
      if len(tags)==0:
        # Ligature...
        self.assertTrue(len(adj)==2)
        self.assertTrue(len(lg.get_tags(adj[0][0]))==1)
        self.assertTrue(len(lg.get_tags(adj[1][0]))==1)
      
      else:
        # Glyph...
        if tags[0][0]=='_O':
          self.assertTrue(len(adj)==0)
        
        elif tags[0][0]=='i':
          self.assertTrue(len(adj)==1)
        
        elif tags[0][0]=='n':
          self.assertTrue(len(adj)==2)
        
        elif tags[0][0]=='k_':
          self.assertTrue(len(adj)==1)
        
        else:
          self.assertTrue(False)
    
    del lg
    
    
    
  def identical(self, a, b):
    a.segment()
    b.segment()
    
    self.assertTrue(a.vertex_count==b.vertex_count)
    self.assertTrue(a.edge_count==b.edge_count)
    self.assertTrue(a.segments==b.segments)
    
    a_min_x, a_max_x, a_min_y, a_max_y = a.get_bounds()
    b_min_x, b_max_x, b_min_y, b_max_y = b.get_bounds()
    self.assertTrue(numpy.fabs(a_min_x-b_min_x)<1e-12)
    self.assertTrue(numpy.fabs(a_max_x-b_max_x)<1e-12)
    self.assertTrue(numpy.fabs(a_min_y-b_min_y)<1e-12)
    self.assertTrue(numpy.fabs(a_max_y-b_max_y)<1e-12)
    
    for i in xrange(a.vertex_count):
      va = a.get_vertex(i)
      vb = b.get_vertex(i)
      for j in xrange(7):
        self.assertTrue(numpy.fabs(va[j]-vb[j])<1e-12)
    
    for i in xrange(a.edge_count):
      ea = a.get_edge(i)
      eb = b.get_edge(i)
      self.assertTrue(ea[0]==eb[0])
      self.assertTrue(ea[1]==eb[1])
    
    tags_a = a.get_tags()
    tags_b = b.get_tags()
    for ta, tb in zip(tags_a, tags_b):
      self.assertTrue(len(ta)==len(tb))
      for va, vb in zip(ta, tb):
        if isinstance(va, float):
          self.assertTrue(numpy.fabs(va-vb)<1e-12)
        else:
          self.assertTrue(va==vb)
    
    splits_a = a.get_splits()
    splits_b = b.get_splits()
    for sa, sb in zip(splits_a, splits_b):
      self.assertTrue(len(sa)==len(sb))
      for va, vb in zip(sa, sb):
        if isinstance(va, float):
          self.assertTrue(numpy.fabs(va-vb)<1e-12)
        else:
          self.assertTrue(va==vb)


  def test_io(self):
    # Circle...
    temp = tempfile.TemporaryFile('w+b')
      
    before = self.make_circle()
    ply2.write(temp, before.as_dict())
      
    temp.seek(0)
    after = LineGraph()
    after.from_dict(ply2.read(temp))
      
    self.identical(before, after)
    temp.close()
    
    # Grid...
    temp = tempfile.TemporaryFile('w+b')
      
    before = self.make_grid()
    ply2.write(temp, before.as_dict())
      
    temp.seek(0)
    after = LineGraph()
    after.from_dict(ply2.read(temp))
      
    self.identical(before, after)
    temp.close()
    
    # Squares...
    temp = tempfile.TemporaryFile('w+b')
      
    before = self.make_squares()
    ply2.write(temp, before.as_dict())
      
    temp.seek(0)
    after = LineGraph()
    after.from_dict(ply2.read(temp))
      
    self.identical(before, after)
    temp.close()
    
    # Text...
    temp = tempfile.TemporaryFile('w+b')
      
    before = self.make_text()
    ply2.write(temp, before.as_dict())
      
    temp.seek(0)
    after = LineGraph()
    after.from_dict(ply2.read(temp))
      
    self.identical(before, after)
    temp.close()



# Run unit tests...
unittest.main()
