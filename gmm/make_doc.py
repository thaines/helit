#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import gmm
from kmeans_shared import KMeansShared
from mixture import Mixture
import pydoc

doc = pydoc.HTMLDoc()


# Open the document...
out = open('gmm.html','w')
out.write('<html>\n')
out.write('<head>\n')
out.write('<title>Gaussian Mixture Model (plus K-means)</title>\n')
out.write('</head>\n')
out.write('<body>\n')


# Function that removes the methods that start with 'do' from a class - to hide them in the documentation...
def pruneClassOfDo(cls):
  methods = dir(cls)
  for method in methods:
    if method[:2]=='do':
      delattr(cls, method)

pruneClassOfDo(KMeansShared)
pruneClassOfDo(gmm.KMeans1)
pruneClassOfDo(gmm.KMeans2)
pruneClassOfDo(gmm.KMeans3)
pruneClassOfDo(Mixture)
pruneClassOfDo(gmm.IsotropicGMM)



# Openning blob...
readme = open('readme.txt','r').read()
readme = readme.replace('\n','<br/>')
out.write(doc.bigsection('Overview','#ffffff','#7799ee',readme))


# Variables...
variables = ''
variables += '<strong>KMeans</strong> = <strong>KMeans3</strong><br/>'
variables += 'The prefered k-means implimentation can be referenced as KMeans<br/><br/>'
variables += '<strong>KMeansShort</strong> = <strong>KMeans1</strong><br/>'
variables += 'The prefered k-means implimentation is choosen on the assumption of long feature vectors - if the feature vectors are in fact short then this is a synonym of a more appropriate fitter. (By short think less than 20, though this is very computer dependent.)<br/>'
variables = variables.replace('&nbsp;',' ')
out.write(doc.bigsection('Synonyms','#ffffff','#8d50ff',variables))


# Functions...
funcs = ''
funcs += doc.docroutine(gmm.modelSelectBIC)
funcs = funcs.replace('&nbsp;',' ')
out.write(doc.bigsection('Functions','#ffffff','#eeaa77',funcs))


# Classes...
classes = ''
classes += doc.docclass(KMeansShared)
classes += doc.docclass(gmm.KMeans1)
classes += doc.docclass(gmm.KMeans2)
classes += doc.docclass(gmm.KMeans3)
classes += doc.docclass(Mixture)
classes += doc.docclass(gmm.IsotropicGMM)
classes = classes.replace('&nbsp;',' ')
out.write(doc.bigsection('Classes','#ffffff','#ee77aa',classes))


# Close the document...
out.write('</body>\n')
out.write('</html>\n')
out.close()
