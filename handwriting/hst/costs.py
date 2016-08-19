#! /usr/bin/env python

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

import numpy

from frf import frf



# Global used by rf cost method - simply so it doesn't have to reload the random forest each time...
cost_proxy = None



def end_dist_cost(left_g, right_g, mass_weight = 1.0):
  """Given two glyphs returns the cost of putting them side by side - height difference between end points of ligaments basically. It hallucinates end points if there are no ligaments, which is stupid."""
  
  # Identify the match points - if the glyphs have links then it is these, if not find the closest point on the x axis in each case and assume its where the line ends...
  if left_g.right==None:
    # No partner - select a horizontal link...
    left_left = [left_g.most_right()[1]]
    left_right = left_left
  elif len(left_g.right[1])==0:
    # It has a partner, but no link - choose some link points...
    left_left  = [left_g.most_right()[1]]
    left_right = [left_g.right[0].most_left()[1]]
  else:
    # We have an actual link we can use, to get real values...
    left_left  = map(lambda l: l[0].lg.get_vertex(l[3])[1], left_g.right[1])
    left_right = map(lambda l: l[0].lg.get_vertex(l[4])[1], left_g.right[1])
  
  if right_g.left==None:
    # No partner - select a horizontal link...
    right_right = [right_g.most_left()[1]]
    right_left = right_right
  elif len(right_g.left[1])==0:
    # It has a partner, but no link - choose some link points...
    right_right = [right_g.most_left()[1]]
    right_left  = [right_g.left[0].most_right()[1]]
  else:
    # We have an actual link we can use, to get real values...
    right_right = map(lambda l: l[0].lg.get_vertex(l[3])[1], right_g.left[1])
    right_left  = map(lambda l: l[0].lg.get_vertex(l[4])[1], right_g.left[1])
    
  # Cost is the height differences for the match points on each side - multiple match points is a possibility...
  ret = 0.0
  ret += numpy.fabs(numpy.array(left_left).reshape((-1,1)) - numpy.array(right_left).reshape((1,-1))).min()
  ret += numpy.fabs(numpy.array(left_right).reshape((-1,1)) - numpy.array(right_right).reshape((1,-1))).min()
  
  # Also add in the absolute difference in average mass and radius...
  dr_diff = numpy.fabs(left_g.get_mass() - right_g.get_mass())
  ret += mass_weight * dr_diff.sum()
  
  return ret



def glyph_pair_feat(left_g, right_g):
  """Given two glyphs this returns their relationship feature - makes use of the average/absolute difference trick to make it appropriate for distance learning, and also matches the far vectors and near vectors for the sides of the glyphs, to make it glyph order dependent."""
  
  # Extract the features, naming them according to the relationship being explored...
  left_far, left_near = left_g.get_feat()
  right_near, right_far = right_g.get_feat()
  
  # Calculate the final feature...
  near_avg = 0.5 * (left_near + right_near)
  near_diff = numpy.fabs(left_near - right_near)
  far_avg = 0.5 * (left_far + right_far)
  far_diff = numpy.fabs(left_far - right_far)
  
  # Concatenate and return...
  return numpy.concatenate((near_avg, near_diff, far_avg, far_diff), axis=0)



def end_dist_cost_rf(left_g, right_g, mass_weight = 1.0):
  """Given two glyphs returns the cost of putting them side by side - height difference between end points of ligaments basically. It hallucinates end points if there are no ligaments, which is stupid."""
  # Check we have a random forest loaded and ready to go...
  global cost_proxy
  if cost_proxy==None:
    cost_proxy = frf.load_forest('cost_proxy.rf')
  
  # Identify the match points - if the glyphs have links then it is these, if not we are going to use a random forest to guess, so give up...
  joined_up = True
  if left_g.right==None:
    joined_up = False
  elif len(left_g.right[1])==0:
    joined_up = False
  else:
    # We have an actual link we can use, to get real values...
    left_left  = map(lambda l: l[0].lg.get_vertex(l[3])[1], left_g.right[1])
    left_right = map(lambda l: l[0].lg.get_vertex(l[4])[1], left_g.right[1])
  
  if right_g.left==None:
    joined_up = False
  elif len(right_g.left[1])==0:
    joined_up = False
  else:
    # We have an actual link we can use, to get real values...
    right_right = map(lambda l: l[0].lg.get_vertex(l[3])[1], right_g.left[1])
    right_left  = map(lambda l: l[0].lg.get_vertex(l[4])[1], right_g.left[1])
    
  # Cost calculation depends if the letters are joined up or not...
  if joined_up:
    # Cost is the height differences for the match points on each side - multiple match points is a possibility...
    ret = 0.0
    ret += numpy.fabs(numpy.array(left_left).reshape((-1,1)) - numpy.array(right_left).reshape((1,-1))).min()
    ret += numpy.fabs(numpy.array(left_right).reshape((-1,1)) - numpy.array(right_right).reshape((1,-1))).min()
  else:
    # Not joined up - fall back to a random forest...
    feat = glyph_pair_feat(left_g, right_g)
    ret = cost_proxy.predict(feat[numpy.newaxis,:], 0)[0]['mean']
  
  # Also add in the absolute difference in average mass and radius...
  dr_diff = numpy.fabs(left_g.get_mass() - right_g.get_mass())
  ret += mass_weight * dr_diff.sum()
  
  return ret



def match_links(l_glyph, r_glyph):
  """Given two glyphs this matches up their links and returns a list of link pairs to be matched. Greedy matching based on distance between match points."""
  
  # Can only stitch if we have tails on both glyphs...
  if l_glyph.right!=None and len(l_glyph.right[1])!=0 and r_glyph.left!=None and len(r_glyph.left[1])!=0:
    l_links = l_glyph.right[1]
    r_links = r_glyph.left[1]
        
    # Match up the links, to choose the best pairings (Greedy)...
    if len(l_links)==1 and len(r_links)==1:
      return [(l_links[0], r_links[0])]
    else:
      matches = []
          
      cost = numpy.zeros((len(l_links), len(r_links)), dtype=numpy.float32)
      for il in xrange(cost.shape[0]):
        for ir in xrange(cost.shape[1]):
          yl = l_links[il][0].lg.get_vertex(l_links[il][3])[1]
          yr = r_links[ir][0].lg.get_vertex(r_links[ir][4])[1]
          cost[il,ir] += numpy.fabs(yl - yr)
              
          yl = l_links[il][0].lg.get_vertex(l_links[il][4])[1]
          yr = r_links[ir][0].lg.get_vertex(r_links[ir][3])[1]
          cost[il,ir] += numpy.fabs(yl - yr)
           
      while True:
        il, ir = numpy.unravel_index(numpy.argmin(cost), cost.shape)
        if cost[il, ir]>1e99: break
             
        matches.append((l_links[il], r_links[ir]))
             
        cost[il,:] = 1e100
        cost[:,ir] = 1e100
      
      return matches
  else:
    return []



def glyph_pair_offset(left_g, right_g, offset_sd, fallback = False):
  """Returns the vertical offset between two glyphs, or None if there is no information to use. Never returns None if you set fallback to True, as it will then use a random forest."""
  matches = match_links(left_g, right_g)
    
  prec_mean = 0.0
  prec = 0.0
  
  for ml, mr in matches:
    yl = ml[0].lg.get_vertex(ml[3])[1]
    yr = mr[0].lg.get_vertex(mr[4])[1]
    offset1 = yr - yl
              
    yl = ml[0].lg.get_vertex(ml[4])[1]
    yr = mr[0].lg.get_vertex(mr[3])[1]
    offset2 = yr - yl
      
    mean = 0.5 * (offset1 + offset2)
    p = 1.0 / (offset_sd**2)
    p += 1.0 / max((0.5*numpy.fabs(offset2 - offset1))**2, 1e-5)
    
    prec_mean += mean * p
    prec += p
  
  if prec>1e-3:
    left_c = left_g.get_center()
    right_c = right_g.get_center()
  
    return ((prec_mean / prec) + (right_c[1] - left_c[1]), numpy.sqrt(1.0/prec))
    
  elif fallback:
    # Check the random forest is loaded...
    global cost_proxy
    if cost_proxy==None:
      cost_proxy = frf.load_forest('cost_proxy.rf')

    # Calculate the feature to be fed into the forest...
    feat = glyph_pair_feat(left_g, right_g)
    
    # Evaluate the random forest...
    res = cost_proxy.predict(feat[numpy.newaxis,:], 0)[1]
    
    # Do some sd weirdness (convolve rf error with requested uncertainty) and return...
    prec = 1.0 / (offset_sd**2)
    prec += 1.0 / res['var']
    
    return (res['mean'], numpy.sqrt(1.0/prec))
    
  else:
    return None
