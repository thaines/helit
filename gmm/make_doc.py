#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import gmm

from utils import doc_gen



# Setup...
doc = doc_gen.DocGen('gmm', 'Gaussian Mixture Model (plus K-means)', 'Gaussian mixture model, with EM, plus assorted k-means implimentations')
doc.addFile('readme.txt', 'Overview')


# Function that removes the methods that start with 'do' from a class - to hide them in the documentation...
def pruneClassOfDo(cls):
  methods = dir(cls)
  for method in methods:
    if method[:2]=='do':
      delattr(cls, method)

pruneClassOfDo(gmm.KMeansShared)
pruneClassOfDo(gmm.KMeans0)
pruneClassOfDo(gmm.KMeans1)
pruneClassOfDo(gmm.KMeans2)
pruneClassOfDo(gmm.KMeans3)
pruneClassOfDo(gmm.Mixture)
pruneClassOfDo(gmm.IsotropicGMM)


# Variables...
doc.addVariable('KMeans', 'The prefered k-means implimentation can be referenced as KMeans')
doc.addVariable('KMeansShort', 'The prefered k-means implimentation is choosen on the assumption of long feature vectors - if the feature vectors are in fact short then this is a synonym of a more appropriate fitter. (By short think less than 20, though this is somewhat computer dependent.)')


# Functions...
doc.addFunction(gmm.modelSelectBIC)


# Classes...
doc.addClass(gmm.KMeansShared)
doc.addClass(gmm.KMeans0)
doc.addClass(gmm.KMeans1)
doc.addClass(gmm.KMeans2)
doc.addClass(gmm.KMeans3)
doc.addClass(gmm.Mixture)
doc.addClass(gmm.IsotropicGMM)
