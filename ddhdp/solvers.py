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



# Loads solvers....

# Load the most basic solver, but also load a mp one if possible...
try:
  from solve_weave import gibbs_all, gibbs_doc, leftRightNegLogProbWord
  __fitter = 'weave'
except:
  raise Exception('Could not load basic weave solver')

try:
  from solve_weave_mp import gibbs_all_mp, gibbs_doc_mp
  __fitter = 'multiprocess weave'
except:
  pass



# Function to get the best fitter avaliable...
def getAlgorithm():
  """Returns a text string indicating which implimentation of the fitting algorithm is being used by default, which will be the best avaliable."""
  global __fitter
  return __fitter
