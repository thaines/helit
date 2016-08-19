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

import string
import numpy
import numpy.random



class Spacing:
  """Provides estimates of how much space should exist between glyphs, learned from a glyph_db. Includes caching of answers, and makes use of statistics. Makes use of weighted medians, for robustness."""
  def __init__(self, glyph_db):
    """You provide a glyph_db that is its data source."""
    
    # Build the database - go through and record the spacing for every glyph pair, as a list of tuples (id left glyph, left key, space, right key, id right glyph)...
    self.db = []
    
    glyphs = glyph_db.get_all()
    for glyph in glyphs:
      if glyph.right!=None:
        id_left = id(glyph)
        key_left = glyph.key
        space = glyph.right[0].orig_left_x() - glyph.orig_right_x()
        key_right = glyph.right[0].key
        id_right = id(glyph.right[0])
        
        self.db.append((id_left, key_left, space, key_right, id_right))
    
    # Default weights...
    self.set_weights()
    
    # Caches...
    self.median_space_cache = None
  
  
  def set_weights(self, match_id = 10.0, match_char = 3.0, match_type = 2.0, has_space = 0.01):
    """Given two chars this class calculates spacing based on the median of weighted samples - which depend on the relationship between the sample and the request. Essentially the weight of each starts at 1, and is multiplied by each weight if it matches its conditions."""
    self.match_id = match_id
    self.match_char = match_char
    self.match_type = match_type
    self.has_space = has_space

    
  def __weighted_median(self, value, weight):
    """Calculates and returns the weighted median of the provided arrays of value and weight."""
    value = numpy.asarray(value)
    weight = numpy.asarray(weight)
    
    order = numpy.argsort(value)
    value = value[order]
    weight = weight[order]
    
    cum_w = numpy.append([0.0], numpy.cumsum(weight))
    
    middle = 0.5 * weight.sum()
    low = numpy.argmax(cum_w[cum_w<middle])
    if low+1 >= value.shape[0]: return value[low]
    
    t = (middle - cum_w[low]) / (cum_w[low+1] - cum_w[low])
    
    return (1.0-t) * value[low] + t * value[low+1]
  
  
  def __weighted_draw(self, value, weight, amount = 0.5):
    """Treats it as a set of samples to draw from, using the supplied weights. Only considers values in the interquartile range around the center defined by amount, such that it avoids outliers."""
    # Sort the arrays, after making sure they are numpy...
    value = numpy.asarray(value)
    weight = numpy.asarray(weight)
    
    order = numpy.argsort(value)
    value = value[order]
    weight = weight[order]
    
    # Find the range to gra from...
    cum_w = numpy.append([0.0], numpy.cumsum(weight))
    
    low = (0.5 - amount*0.5) * weight.sum()
    high = (0.5 + amount*0.5) * weight.sum()
    
    # Get the range of base values...
    options = numpy.nonzero(numpy.logical_and(low<cum_w, cum_w<high))[0]
    if options.shape[0]==0: return numpy.mean(low, high)
    
    # Draw an option...
    draw = numpy.random.multinomial(1, weight[options] / weight[options].sum())
    index = options[numpy.nonzero(draw)][0]
    
    # Do a uniform draw around the option, and return it...
    below = index-1 if index-1>=0 else 0
    above = index+1 if index+1<value.shape[0] else index
    
    below = 0.5 * (value[below] + value[index])
    above = 0.5 * (value[index] + value[above])
    
    if above-below>1e-3: return numpy.random.uniform(below, above)
    else: return 0.5 * (below + above)

    
  def __type(self, char):
    """For internal use - returns an integer representing the type of a char. Strongly biased towards English."""
    if char in string.ascii_lowercase: return 0
    if char in string.ascii_uppercase: return 1
    if char in string.digits: return 2
    if char in string.punctuation: return 3
    return -1
    
    
  def __weight(self, entry, left, right):
    """Internal method - returns the weight assigned to the given entry in the db for the given left & right."""
    w = 1.0
    
    if entry[0]==id(left): w *= self.match_id
    if entry[4]==id(right): w *= self.match_id
    
    l_char = filter(lambda c: c!='_', left.key)
    r_char = filter(lambda c: c!='_', right.key)
    le_char = filter(lambda c: c!='_', entry[1])
    re_char = filter(lambda c: c!='_', entry[3])
    
    if l_char==le_char: w *= self.match_char    
    if r_char==re_char: w *= self.match_char
    
    if self.__type(l_char)==self.__type(le_char): w *= self.match_type
    if self.__type(r_char)==self.__type(re_char): w *= self.match_type
    
    if entry[1].endswith('_'): w *= self.has_space
    if entry[3].startswith('_'): w *= self.has_space
    
    return 1.0


  def median_space(self):
    """Returns the size of a space - the median seen in the db."""
    if self.median_space_cache!=None: return self.median_space_cache
    
    nums = []
    for entry in self.db:
      if entry[1].endswith('_') and entry[3].startswith('_'):
        nums.append(entry[2])
    
    if len(nums)==0: self.median_space_cache = 0.6
    else: self.median_space_cache = numpy.median(nums)
    
    return self.median_space_cache
  
  
  def draw_space(self, amount = 0.5):
    """Same as median space, but with a bit of noise, in the same style as the draw method for normal spacing."""
    nums = []
    for entry in self.db:
      if entry[1].endswith('_') and entry[3].startswith('_'):
        nums.append(entry[2])
    if len(nums)==0: return 0.6
    
    return self.__weighted_draw(nums, [1.0]*len(nums), amount)
    
    
  def median(self, left, right):
    """Returns the weighted median of the dataset, where the weights have been adjusted for the provided glyph pair."""
    
    values = map(lambda e: e[2], self.db)
    weights = map(lambda e: self.__weight(e, left, right), self.db)
    
    return self.__weighted_median(values, weights)

    
  def draw(self, left, right, amount = 0.5):
    """Same as median, except it treats the values as defining a probability distribution, and draws from the middle amount (parameter) of the weight mass. Basically adds some noise, without the risk of sampling outliers."""
    
    value = map(lambda e: e[2], self.db)
    weight = map(lambda e: self.__weight(e, left, right), self.db)
    
    return self.__weighted_draw(value, weight, amount)
