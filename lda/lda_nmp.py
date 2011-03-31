# -*- coding: utf-8 -*-

# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



# This loads in the entire libarary and provide the interface - only import needed by a user.
# Version that doesn't consider using a multiprocess solver.

# Load in all the data structure types...
from document import Document
from topic import Topic
from corpus import Corpus



# Get the shared solvers params object...
from solve_shared import Params



# Load in a suitable solver - autodetect the most powerful supported...
try:
  from solve_weave import fit,fitDoc
  __fitter = 'weave'
except:
  try:
    from solve_python import fit,fitDoc
    __fitter = 'python'  
  except:
    raise Exception('All of the lda solvers failed to load.')



def getAlgorithm():
  """Returns a text string indicating which implimentation of the fitting algorithm is being used."""
  global __fitter
  return __fitter
  