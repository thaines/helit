# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



# Loads solvers....

# Load the most basic solver, but also load a mp one if possible...
try:
  from solve_weave import gibbs_all, gibbs_doc
  __fitter = 'weave'
except:
  raise
  #raise Exception('Could not load basic weave solver')

try:
  from solve_weave_mp import gibbs_all_mp, gibbs_doc_mp
  __fitter = 'multiprocess weave'
except:
  pass



def getAlgorithm():
  """Returns a text string indicating which implimentation of the fitting algorithm is being used by default, which will be the best avaliable."""
  global __fitter
  return __fitter
