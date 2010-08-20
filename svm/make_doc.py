#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2010 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



# This script file generates a svm.html file, which contains all the information needed to use the svm system.

import svm
import pydoc

doc = pydoc.HTMLDoc()


# Open the document...
out = open('svm.html','w')
out.write('<html>\n')
out.write('<head>\n')
out.write('<title>pysvm</title>\n')
out.write('</head>\n')
out.write('<body>\n')


# Openning blob...
readme = open('readme.txt','r').read()
readme = readme.replace('\n','<br/>')
out.write(doc.bigsection('Overview','#ffffff','#7799ee',readme))


# lda functions...
funcs = ''
funcs += doc.docroutine(svm.solvePair)
funcs += doc.docroutine(svm.looPair)
funcs += doc.docroutine(svm.looPairSelect)
funcs = funcs.replace('&nbsp;',' ')
out.write(doc.bigsection('Functions','#ffffff','#eeaa77',funcs))


# Classes...
classes = ''
classes += doc.docclass(svm.Kernel)
classes += doc.docclass(svm.Params)
classes += doc.docclass(svm.ParamsRange)
classes += doc.docclass(svm.ParamsSet)
classes += doc.docclass(svm.Dataset)
classes += doc.docclass(svm.Model)
classes += doc.docclass(svm.MultiModel)
classes = classes.replace('&nbsp;',' ')
out.write(doc.bigsection('Classes','#ffffff','#ee77aa',classes))


# Close the document...
out.write('</body>\n')
out.write('</html>\n')
out.close()
