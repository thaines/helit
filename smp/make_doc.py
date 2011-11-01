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



import smp
import pydoc

doc = pydoc.HTMLDoc()


# Open the document...
out = open('smp.html','w')
out.write('<html>\n')
out.write('<head>\n')
out.write('<title>Sparse Multinomial Posterior</title>\n')
out.write('</head>\n')
out.write('<body>\n')


# Openning blob...
readme = open('readme.txt','r').read()
readme = readme.replace('\n','<br/>')
out.write(doc.bigsection('Overview','#ffffff','#7799ee',readme))



# Variables...
variables = ''
variables += '<strong>smp_code</strong><br/>'
variables += '<br/>String containing the C++ code that does the actual work for the system.<br/><br/>'
variables = variables.replace('&nbsp;',' ')
out.write(doc.bigsection('Variables','#ffffff','#8d50ff',variables))



# Classes...
classes = ''
classes += doc.docclass(smp.SMP)
classes += doc.docclass(smp.FlagIndexArray)
classes = classes.replace('&nbsp;',' ')
out.write(doc.bigsection('Classes','#ffffff','#ee77aa',classes))


# Close the document...
out.write('</body>\n')
out.write('</html>\n')
out.close()
