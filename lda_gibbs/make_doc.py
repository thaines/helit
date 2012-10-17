#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import lda

from utils import doc_gen



# Setup...
doc = doc_gen.DocGen('lda_gibbs', 'Latent Dirichlet Allocation (Gibbs)', 'Gibbs sampling implimentation of latent Dirichlet allocation')
doc.addFile('readme.txt', 'Overview')


# Functions...
doc.addFunction(lda.getAlgorithm)


# Classes...
doc.addClass(lda.Document)
doc.addClass(lda.Topic)
doc.addClass(lda.Corpus)
doc.addClass(lda.Params)
