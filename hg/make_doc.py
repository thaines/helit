#! /usr/bin/env python

# Copyright 2016 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import hg

from utils import doc_gen



# Setup...
doc = doc_gen.DocGen('hg', 'Homography', 'Library for constructing and applying homographies.')
doc.addFile('readme.txt', 'Overview')



# Functions...
doc.addFunction(hg.translate)
doc.addFunction(hg.rotate)
doc.addFunction(hg.scale)

doc.addFunction(hg.match)

doc.addFunction(hg.bounds)
doc.addFunction(hg.fit)
doc.addFunction(hg.scaling)

doc.addFunction(hg.fillmasked)
doc.addFunction(hg.transform)

doc.addFunction(hg.sample)
doc.addFunction(hg.offsets)
doc.addFunction(hg.rotsets)

doc.addFunction(hg.Gaussian)
