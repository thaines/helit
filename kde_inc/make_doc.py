#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



# This script file generates a kde_inc.html file, which contains all the information needed to use the system.

import kde_inc
import pydoc

doc = pydoc.HTMLDoc()


# Open the document...
out = open('kde_inc.html','w')
out.write('<html>\n')
out.write('<head>\n')
out.write('<title>Incrimental capped KDE</title>\n')
out.write('</head>\n')
out.write('<body>\n')


# Openning blob...
readme = open('readme.txt','r').read()
readme = readme.replace('\n','<br/>')
out.write(doc.bigsection('Overview','#ffffff','#7799ee',readme))



# Classes...
classes = ''
classes += doc.docclass(kde_inc.PrecisionLOO)
classes += doc.docclass(kde_inc.SubsetPrecisionLOO)
classes += doc.docclass(kde_inc.GMM)
classes += doc.docclass(kde_inc.KDE_INC)
classes = classes.replace('&nbsp;',' ')
out.write(doc.bigsection('Classes','#ffffff','#ee77aa',classes))


# Close the document...
out.write('</body>\n')
out.write('</html>\n')
out.close()
