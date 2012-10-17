#! /usr/bin/env python

# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import p_cat

from utils import doc_gen



# Setup...
doc = doc_gen.DocGen('p_cat', 'Probabilistic Classification', 'Standardised interface to probabilistic classifiers with features for active learning')
doc.addFile('readme.txt', 'Overview')


# Classes...
doc.addClass(p_cat.ProbCat)
doc.addClass(p_cat.ClassifyGaussian)
doc.addClass(p_cat.ClassifyKDE)
doc.addClass(p_cat.ClassifyDPGMM)
doc.addClass(p_cat.ClassifyDF)
