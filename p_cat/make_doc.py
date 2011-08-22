#! /usr/bin/env python

# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import p_cat
import pydoc

doc = pydoc.HTMLDoc()


# Open the document...
out = open('p_cat.html','w')
out.write('<html>\n')
out.write('<head>\n')
out.write('<title>Probabilistic Classification</title>\n')
out.write('</head>\n')
out.write('<body>\n')


# Openning blob...
readme = open('readme.txt','r').read()
readme = readme.replace('\n','<br/>')
out.write(doc.bigsection('Overview','#ffffff','#7799ee',readme))


# Classes...
classes = ''
classes += doc.docclass(p_cat.ProbCat)
classes += doc.docclass(p_cat.ClassifyGaussian)
classes += doc.docclass(p_cat.ClassifyKDE)
classes += doc.docclass(p_cat.ClassifyDPGMM)
classes = classes.replace('&nbsp;',' ')
out.write(doc.bigsection('Classes','#ffffff','#ee77aa',classes))


# Close the document...
out.write('</body>\n')
out.write('</html>\n')
out.close()
