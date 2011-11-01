# Copyright 2011 Tom SF Haines

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.



import numpy



class FlagIndexArray:
  """Provides a register for flag lists - given a list of true/false flags gives a unique number for each combination. Requesting the numebr associated with a combination that has already been entered will always return the same number. All flag lists should be the same length and you can obtain a numpy matrix of {0,1} valued unsigned chars where each row corresponds to the flag list with that index. Also has a function to add the flags for each case of only one flag being on, which if called before anything else puts them so the index of the flag and the index of the flag list correspond - a trick required by the rest of the system."""
  def __init__(self, length, addSingles = False):
    """Requires the length of the flag lists. Alternativly it can clone another FlagIndexArray. Will call the addSingles method for you if the flag is set."""
    if isinstance(length, FlagIndexArray):
      self.length = length.length
      self.flags = dict(length.flags)
    else:
      self.length = length
      self.flags = dict() # Dictionary from flag lists to integers. Flag lists are represented with tuples of {0,1}.
      if addSingles: self.addSingles()

  def getLength(self):
    """Return the length that all flag lists should be."""
    return self.length

  def addSingles(self):
    """Adds the entries where only a single flag is set, with the index of the flag list set to match the index of the flag that is set. Must be called first, before flagIndex is ever called."""
    for i in xrange(self.length):
      t = tuple([0]*i + [1] + [0]*(self.length-(i+1)))
      self.flags[t] = i

  def flagIndex(self, flags, create = True):
    """Given a flag list returns its index - if it has been previously supplied then it will be the same index, otherwise a new one. Can be passed any entity that can be indexed via [] to get the integers {0,1}. Returns a natural. If the create flag is set to False in the event of a previously unseen flag list it will raise an exception instead of assigning it a new natural."""
    f = [0]*self.length
    for i in xrange(self.length):
      if flags[i]!=0: f[i] = 1
    f = tuple(f)

    if f in self.flags: return self.flags[f]
    if create==False: raise Exception('Unrecognised flag list')

    index = len(self.flags)
    self.flags[f] = index
    return index

  def addFlagIndexArray(self, fia, remap = None):
    """Given a flag index array this merges its flags into the new flags, returning a dictionary indexed by fia's indices that converts them to the new indices in self. remap is optionally a dictionary converting flag indices in fia to flag indexes in self - remap[fia index] = self index."""
    def adjust(fi):
      fo = [0]*self.length
      for i in xrange(fia.length):
        fo[remap[i]] = fi[i]
      return tuple(fo)

    ret = dict()

    for f, index in fia.flags.iteritems():
      if remap: f = adjust(f)
      ret[index] = self.flagIndex(f)

    return ret

  def flagCount(self):
    """Returns the number of flag lists that are in the system."""
    return len(self.flags)

  def getFlagMatrix(self):
    """Returns a 2D numpy array of type numpy.uint8 containing {0,1}, indexed by [flag index,flag entry] - basically all the flags stacked into a single matrix and indexed by the entries returned by flagIndex. Often refered to as a 'flag index array' (fia)."""
    ret = numpy.zeros((len(self.flags),self.length), dtype=numpy.uint8)
    for flags,row in self.flags.iteritems():
      for col in xrange(self.length):
        if flags[col]!=0: ret[row,col] = 1
    return ret
