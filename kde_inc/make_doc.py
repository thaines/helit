#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



# This script file generates a kde_inc.html file, which contains all the information needed to use the system.

import kde_inc

from utils import doc_gen



# Setup...
doc = doc_gen.DocGen('kde_inc', 'Incrimental kernel density estimation', 'Kernel density estimation with Gaussian kernels and greedy merging beyond a cap')
doc.addFile('readme.txt', 'Overview')


# Classes...
doc.addClass(kde_inc.PrecisionLOO)
doc.addClass(kde_inc.SubsetPrecisionLOO)
doc.addClass(kde_inc.GMM)
doc.addClass(kde_inc.KDE_INC)
