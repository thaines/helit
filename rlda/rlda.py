# -*- coding: utf-8 -*-

# Copyright 2010 Tom SF Haines

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



from params import Params
from document import Document
from corpus import Corpus



# Load in a suitable solver - autodetect the most powerful supported...
try:
  from solve_weave_mp import fit, fitDoc
  __fitter = 'multiprocess weave'
except:
  try:
    from solve_weave import fit, fitDoc
    __fitter = 'weave'
  except:
    raise Exception('All of the rLDA solvers failed to load.')



def getAlgorithm():
  """Returns a text string indicating which implimentation of the fitting algorithm is being used."""
  global __fitter
  return __fitter