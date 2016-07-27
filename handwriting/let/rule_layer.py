# Copyright 2016 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import numpy
import numpy.linalg as la

from hg.homography import match as match_hg

from line_graph.utils_gui.viewport_layer import *



class RuleLayer(Layer):
  """Provides a rule - renders it and provides the ability to drag them around to get them to match the lines on the original paper. Has ply2 file i/o as well. The internal homography goes from line space to image space, as does the one saved to disk. Never liked this code - the homography editting is terrible, and goes wrong all the time. Should really be replaced with a 'list of matched points' system, where the user can edit/delete matches."""
  def __init__(self):    
    self.show = True
    self.grid = False
    
    self.reset()
  
  
  def get_size(self):
    """Dummy - required to satisfy the interface."""
    return (240, 320)
  
  
  def reset(self):
    """For if the homography goes pear shaped."""
    # Setup a default homography, from lines on the y=integers to image space - initialise with lines that are 192 pixels high, as that is about right for normal ruled paper scanned at 600dpi...
    self.homography = numpy.eye(3, dtype=numpy.float32)
    self.homography[0,0] *= 192.0
    self.homography[1,1] *= 192.0
    
    # Default set of matches - set nice and far away...
    self.match = [] # List of last 4 matches, as ((source x, source y), (dest x, dest y))
    
    for x in xrange(2):
      for y in xrange(2):
        px = x - 0.5
        py = y - 0.5
        
        p = numpy.dot(self.homography, numpy.array((px, py, 1.0), dtype=numpy.float32))
        p /= p[2]
        
        self.match.append(((px, py), (p[0], p[1]), True)) # line space, image space, False for user provided, True for default.
  
  
  def ply2_save(self, dic):
    """Given a dictionary representing a ply2 file this adds the homography details to it."""
    if 'element' not in dic:
      dic['element'] = dict()
      
    if 'homography' not in dic['element']:
      dic['element']['homography'] = dict()
      
    dic['element']['homography']['v'] = self.homography.flatten()
  
  
  def ply2_load(self, dic):
    """Given a dictionary representing a ply2 file this extracts the homography details to it. Fails silently if not avaliable."""
    if 'element' not in dic:
      return
    
    if 'homography' not in dic['element']:
      return
    
    if 'v' in dic['element']['homography']:
      data = dic['element']['homography']['v']
      
      if len(data.shape)==1 and data.shape[0]==9:
        self.homography = data.reshape((3,3)).astype(numpy.float32)
  
  
  def image_slices(self, dims, spacing = 0.0):
    """Given the tuple (height, width) this uses the Homography to define a set of lines over the implied image - it returns a list of line indices and bounding boxes (line, low_y,  high_y), only of the lines that are completly in the image. It includes the given spacing value of extra space on either side, measures in line space."""
    
    # We are going to need the inverse...
    inv_homography = la.inv(self.homography)
    
    # Find the range of lines to consider, by extracting the minimum/maximum from the inverse homography applied to the four corners...
    pnt = numpy.empty((4,3), dtype=numpy.float32)
    pnt[0,:] = numpy.dot(inv_homography, numpy.array((0, 0, 1), dtype=numpy.float32))
    pnt[1,:] = numpy.dot(inv_homography, numpy.array((dims[1], 0, 1), dtype=numpy.float32))
    pnt[2,:] = numpy.dot(inv_homography, numpy.array((0, dims[0], 1), dtype=numpy.float32))
    pnt[3,:] = numpy.dot(inv_homography, numpy.array((dims[1], dims[0], 1), dtype=numpy.float32))
    
    pnt /= pnt[:,2, numpy.newaxis]

    min_line = int(numpy.ceil(pnt[:,1].min()))
    max_line = int(numpy.floor(pnt[:,1].max()))

    # Iterate the lines and calculate the boudn for each - quite involved...
    ret = []
    
    for line in xrange(min_line, max_line):
      # Get the 4 points that define the line, including the spacing term...
      pnt[0,:] = numpy.dot(self.homography, numpy.array((0.0, line - spacing, 1.0), dtype=numpy.float32))
      pnt[1,:] = numpy.dot(self.homography, numpy.array((1.0, line - spacing, 1.0), dtype=numpy.float32))
      pnt[2,:] = numpy.dot(self.homography, numpy.array((0.0, line + 1.0 + spacing, 1.0), dtype=numpy.float32))
      pnt[3,:] = numpy.dot(self.homography, numpy.array((1.0, line + 1.0 + spacing, 1.0), dtype=numpy.float32))
      
      pnt /= pnt[:,2, numpy.newaxis]
      
      # Project them so they are on the edge of the image...
      lims = []
      
      for base in [0,2]:
        nx = pnt[base+1,0] - pnt[base,0]
        ny = pnt[base+1,1] - pnt[base,1]
        
        if (numpy.fabs(nx)>1e-6):
          d = (0.0 - pnt[base,0]) / nx
          p = (pnt[base,0] + nx*d, pnt[base,1] + ny*d)
          if p[1]>=0.0 and p[1]<dims[0]: lims.append(p)
      
          d = (dims[1] - pnt[base,0]) / nx
          p = (pnt[base,0] + nx*d, pnt[base,1] + ny*d)
          if p[1]>=0.0 and p[1]<dims[0]: lims.append(p)
      
        if (numpy.fabs(ny)>1e-6):
          d = (0.0 - pnt[base,1]) / ny
          p = (pnt[base,0] + nx*d, pnt[base,1] + ny*d)
          if p[0]>=0.0 and p[0]<dims[1]: lims.append(p)
      
          d = (dims[0] - pnt[base,1]) / ny
          p = (pnt[base,0] + nx*d, pnt[base,1] + ny*d)
          if p[0]>=0.0 and p[0]<dims[1]: lims.append(p)
      
      # Take the y range as the output, checking its valid...
      if len(lims)==4:
        lims = numpy.array(lims)
        ret.append((line, lims[:,1].min(), lims[:,1].max()))
    
    return ret
  
  
  def drag(self, sx, sy, ex, ey):
    """Used to alter the rule, to match the lines the text was written on."""
    source = numpy.empty((4,2), dtype=numpy.float32)
    dest   = numpy.empty((4,2), dtype=numpy.float32)
    
    # Use the provided point to create a pair...
    inv_homography = la.inv(self.homography)
    
    p = numpy.dot(inv_homography, numpy.array((sx, sy, 1.0), dtype=numpy.float32))
    p /= p[2]
    
    new_match = ((p[0], p[1]), (ex, ey), False)
    
    user_matches = len(filter(lambda m: m[2]==False, self.match))
    
    # Check if they are tweaking a pre-existing point - if so we update replacing that one, rather than doing anything clever...
    for mi in xrange(4):
      dx = self.match[mi][0][0] - new_match[0][0]
      dy = self.match[mi][0][1] - new_match[0][1]
      dist = numpy.sqrt(dx*dx + dy*dy)
      
      if dist<0.3 and self.match[2]==False:
        if user_matches==1 and mi==0:
          user_matches = 0
          break # Its a correction for the first use - make it do that again.
        
        if user_matches==2 and mi==1:
          user_matches = 1
          break # Same as above
        
        if user_matches==3 and mi==2:
          user_matches = 2
          break # Ditto
        
        self.match[mi] = new_match
        
        # Fill in the source and dest with the matches...
        for r in xrange(4):
          source[r,0] = self.match[r][0][0]
          source[r,1] = self.match[r][0][1]
      
          dest[r,0] = self.match[r][1][0]
          dest[r,1] = self.match[r][1][1]

        # Use the matches to learn the relevant homography...
        self.homography = match_hg(source, dest)
        
        # We are done - no need for the complex solution, so return...
        return
    
    
    # Special case if this is the first user provided match - make it a straight offset of the existing matches, dropping the match into the first position...
    if user_matches==0:
      # Put the first match into the first bucket - doesn't really matter...
      self.match[0] = new_match

      # Redo the other matches to be an offset...
      for mi in xrange(1, 4):
        offset = (self.match[mi][1][0] + ex - sx, self.match[mi][1][1] + ey - sy)
        self.match[mi] = (self.match[mi][0], offset, True)
      
      # Fill in the source and dest with the matches...
      for r in xrange(4):
        source[r,0] = self.match[r][0][0]
        source[r,1] = self.match[r][0][1]
      
        dest[r,0] = self.match[r][1][0]
        dest[r,1] = self.match[r][1][1]

      # Use the matches to learn the relevant homography...
      self.homography = match_hg(source, dest)
      return
    
    
    # Special case the second time the user provides input - make that set the scale and orientation, which means the user should only have to tweak thereafter for minor issues...
    if user_matches==1:
      self.match[1] = new_match
      
      # Redo the last two matches using vector math - basically make it into a square grid with the scale and rotation implied by the first two points...
      vec0 = numpy.array(self.match[1][0]) - numpy.array(self.match[0][0])
      vec1 = numpy.array(self.match[1][1]) - numpy.array(self.match[0][1])
      
      vec0 = (vec0[1], -vec0[0])
      vec1 = (vec1[1], -vec1[0])
      
      s = 1e-2
      self.match[2] = ((self.match[0][0][0] + s*vec0[0], self.match[0][0][1] + s*vec0[01]), (self.match[0][1][0] + s*vec1[0], self.match[0][1][1] + s*vec1[1]), True)
      self.match[3] = ((self.match[1][0][0] + s*vec0[0], self.match[1][0][1] + s*vec0[01]), (self.match[1][1][0] + s*vec1[0], self.match[1][1][1] + s*vec1[1]), True)
      
      # Fill in the source and dest with the matches...
      for r in xrange(4):
        source[r,0] = self.match[r][0][0]
        source[r,1] = self.match[r][0][1]
      
        dest[r,0] = self.match[r][1][0]
        dest[r,1] = self.match[r][1][1]

      # Use the matches to learn the relevant homography...
      self.homography = match_hg(source, dest)
      return
    
    
    # Special case the third time - make it so we only take the skew they imply by their vertex...
    if user_matches==2:
      self.match[2] = new_match
      
      # Redo the last match using vector maths, to get the sanest possible grid...
      vec0 = numpy.array(self.match[1][0]) - numpy.array(self.match[0][0])
      vec1 = numpy.array(self.match[1][1]) - numpy.array(self.match[0][1])
      
      s = 1e-2
      self.match[3] = ((self.match[2][0][0] + s*vec0[0], self.match[2][0][1] + s*vec0[01]), (self.match[2][1][0] + s*vec1[0], self.match[2][1][1] + s*vec1[1]), True)
      
      # Fill in the source and dest with the matches...
      for r in xrange(4):
        source[r,0] = self.match[r][0][0]
        source[r,1] = self.match[r][0][1]
      
        dest[r,0] = self.match[r][1][0]
        dest[r,1] = self.match[r][1][1]

      # Use the matches to learn the relevant homography...
      self.homography = match_hg(source, dest)
      return
      return
    

    # Time for something complex - start by iterate replacing every match in the array with the new one, storing the resulting homography for each...
    hhl = []
    
    for kill in xrange(4):
      match = self.match[:]
      match[kill] = new_match
    
      # Fill in the source and dest with the matches...
      for r in xrange(4):
        source[r,0] = match[r][0][0]
        source[r,1] = match[r][0][1]
      
        dest[r,0] = match[r][1][0]
        dest[r,1] = match[r][1][1]

      # Use the matches to learn the relevant homography...
      hhl.append(match_hg(source, dest))
    
    # Go through and select the most stable homography...
    best = 0
    best_score = 0.0
    
    for i, hg in enumerate(hhl):
      east = numpy.dot(hg,  [1.0,0.0,0.0])[:2]
      west = numpy.dot(hg, [-1.0,0.0,0.0])[:2]
      north = numpy.dot(hg,  [0.0,1.0,0.0])[:2]
      south = numpy.dot(hg, [0.0,-1.0,0.0])[:2]
      
      east /= la.norm(east)
      west /= la.norm(west)
      north /= la.norm(north)
      south /= la.norm(south)
      
      score = 4.0
      score -= numpy.fabs(numpy.dot(north, east))
      score -= numpy.fabs(numpy.dot(north, west))
      score -= numpy.fabs(numpy.dot(south, east))
      score -= numpy.fabs(numpy.dot(south, west))
      
      if match[i][2]: score += 1.0
      
      if score>best_score:
        best = i
        best_score = score

    # Record the best...
    del self.match[best]
    self.match.append(new_match)
    self.homography = hhl[best]

  
  def set_visible(self, value):
    """Allows you to toggle the visibility of this layer"""
    self.show = value
    
  def set_grid(self, value):
    """If True shows itself as a grid rather than just lines."""
    self.grid = value
  
  
  def draw(self, ctx, vp):
    """Draw the rules, as light green slightly transparent lines."""
    if self.show==False: return
    
    # First work out the range of the y axis in the image plane...
    inv_homography = la.inv(self.homography)
    
    cp1 = numpy.dot(inv_homography, numpy.array((vp.start_x, vp.start_y, 1.0), dtype=numpy.float32))
    cp2 = numpy.dot(inv_homography, numpy.array((vp.start_x, vp.end_y, 1.0), dtype=numpy.float32))
    cp3 = numpy.dot(inv_homography, numpy.array((vp.end_x, vp.start_y, 1.0), dtype=numpy.float32))
    cp4 = numpy.dot(inv_homography, numpy.array((vp.end_x, vp.end_y, 1.0), dtype=numpy.float32))
    
    cp1 /= cp1[2]
    cp2 /= cp2[2]
    cp3 /= cp3[2]
    cp4 /= cp4[2]
    
    low_y  = min(cp1[1], cp2[1], cp3[1], cp4[1])
    high_y = max(cp1[1], cp2[1], cp3[1], cp4[1])

    # Iterate all integers in the range, each being a line, and render it...
    for i in xrange(int(numpy.ceil(low_y)), int(numpy.floor(high_y))+1):
      # Get two points on the line...
      p1 = numpy.dot(self.homography, numpy.array((0.0, i, 1.0), dtype=numpy.float32))
      p2 = numpy.dot(self.homography, numpy.array((1.0, i, 1.0), dtype=numpy.float32))
      
      p1 /= p1[2]
      p2 /= p2[2]
      
      # Convert to screen coordinates...
      sx, sy = vp.original_to_view(p1[0], p1[1])
      ex, ey = vp.original_to_view(p2[0], p2[1])
      
      # Extend them so that they fill the screen...
      nx = ex - sx
      ny = ey - sy
      
      ## Intercept with each side of the screen, as possible...
      points = []
      
      if (numpy.fabs(nx)>1e-6):
        d = (0.0 - sx) / nx
        p = (sx + nx*d, sy + ny*d)
        if p[1]>=0.0 and p[1]<vp.height: points.append(p)
      
        d = (vp.width - sx) / nx
        p = (sx + nx*d, sy + ny*d)
        if p[1]>=0.0 and p[1]<vp.height: points.append(p)
      
      if (numpy.fabs(ny)>1e-6):
        d = (0.0 - sy) / ny
        p = (sx + nx*d, sy + ny*d)
        if p[0]>=0.0 and p[0]<vp.width: points.append(p)
      
        d = (vp.height - sy) / ny
        p = (sx + nx*d, sy + ny*d)
        if p[0]>=0.0 and p[0]<vp.width: points.append(p)
      
      # Render...
      if len(points)==2:
        ctx.set_line_width(1.0)
        ctx.set_source_rgba(0.26, 0.62, 0.34, 0.8)
      
        ctx.move_to(points[0][0], points[0][1])
        ctx.line_to(points[1][0], points[1][1])
        ctx.stroke()
    
    # If requested draw the x lines as well, to make a grid...
    if self.grid:
      low_x  = min(cp1[0], cp2[0], cp3[0], cp4[0])
      high_x = max(cp1[0], cp2[0], cp3[0], cp4[0])
      
      for i in xrange(int(numpy.ceil(low_x)), int(numpy.floor(high_x))+1):
        # Get two points on the line...
        p1 = numpy.dot(self.homography, numpy.array((i, 0.0, 1.0), dtype=numpy.float32))
        p2 = numpy.dot(self.homography, numpy.array((i, 1.0, 1.0), dtype=numpy.float32))
      
        p1 /= p1[2]
        p2 /= p2[2]
      
        # Convert to screen coordinates...
        sx, sy = vp.original_to_view(p1[0], p1[1])
        ex, ey = vp.original_to_view(p2[0], p2[1])
      
        # Extend them so that they fill the screen...
        nx = ex - sx
        ny = ey - sy
      
        ## Intercept with each side of the screen, as possible...
        points = []
      
        if (numpy.fabs(nx)>1e-6):
          d = (0.0 - sx) / nx
          p = (sx + nx*d, sy + ny*d)
          if p[1]>=0.0 and p[1]<vp.height: points.append(p)
      
          d = (vp.width - sx) / nx
          p = (sx + nx*d, sy + ny*d)
          if p[1]>=0.0 and p[1]<vp.height: points.append(p)
      
        if (numpy.fabs(ny)>1e-6):
          d = (0.0 - sy) / ny
          p = (sx + nx*d, sy + ny*d)
          if p[0]>=0.0 and p[0]<vp.width: points.append(p)
      
          d = (vp.height - sy) / ny
          p = (sx + nx*d, sy + ny*d)
          if p[0]>=0.0 and p[0]<vp.width: points.append(p)
      
        # Render...
        if len(points)==2:
          ctx.set_line_width(1.0)
          ctx.set_source_rgba(0.36, 0.52, 0.34, 0.5)
      
          ctx.move_to(points[0][0], points[0][1])
          ctx.line_to(points[1][0], points[1][1])
          ctx.stroke()
