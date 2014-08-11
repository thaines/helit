#! /usr/bin/env python

# Copyright 2013 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import ms

from utils import doc_gen



# Setup...
doc = doc_gen.DocGen('ms', 'Mean Shift', 'Mean shift, plus kernel density estimation and subspace constrained mean shift.')
doc.addFile('readme.txt', 'Overview')



# Pull in information about the supported kernels...
text = []
for kernel in ms.MeanShift.kernels():
  text.append(kernel)
  
  d = ms.MeanShift.info(kernel)
  text.append(d)
  
  c = ms.MeanShift.info_config(kernel)
  if c==None:
    text.append('Kernel does not require configuring')
  else:
    text.append(c)
  text.append('')

doc.addOther('\n'.join(text), "Kernels", False)



# Pull in information about the supported spatial data structures...
text = []
for spatial in ms.MeanShift.spatials():
  text.append(spatial)
  
  d = ms.MeanShift.info(spatial)
  text.append(d)
  text.append('')

doc.addOther('\n'.join(text), "Spatial Indexing Structures", False)



# Pull in information about the supported ball detection methods...
text = []
for ball in ms.MeanShift.balls():
  text.append(ball)
  
  d = ms.MeanShift.info(ball)
  text.append(d)
  text.append('')

doc.addOther('\n'.join(text), "Cluster Convergence Detection Methods (balls)", False)



# Classes...
doc.addClass(ms.MeanShift)
