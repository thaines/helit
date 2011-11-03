#! /usr/bin/env python

# Copyright 2011 Tom SF Haines

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.



import ddhdp
import pydoc

doc = pydoc.HTMLDoc()


# Open the document...
out = open('ddhdp.html','w')
out.write('<html>\n')
out.write('<head>\n')
out.write('<title>Delta-Dual Hierarchical Dirichlet Processes</title>\n')
out.write('</head>\n')
out.write('<body>\n')


# Openning blob...
readme = open('readme.txt','r').read()
readme = readme.replace('\n','<br/>')
out.write(doc.bigsection('Overview','#ffffff','#7799ee',readme))


# Functions...
funcs = doc.docroutine(ddhdp.getAlgorithm)
funcs = funcs.replace('&nbsp;',' ')
out.write(doc.bigsection('Functions','#ffffff','#eeaa77',funcs))


# Classes...
classes = ''
classes += doc.docclass(ddhdp.PriorConcDP)
classes += doc.docclass(ddhdp.Params)
classes += doc.docclass(ddhdp.Document)
classes += doc.docclass(ddhdp.Corpus)
classes += doc.docclass(ddhdp.DocSample)
classes += doc.docclass(ddhdp.Sample)
classes += doc.docclass(ddhdp.Model)
classes += doc.docclass(ddhdp.DocModel)
classes = classes.replace('&nbsp;',' ')
out.write(doc.bigsection('Classes','#ffffff','#ee77aa',classes))


# Close the document...
out.write('</body>\n')
out.write('</html>\n')
out.close()
