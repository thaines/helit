#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2011, Tom SF Haines
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#  * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



import cvarray
import mp_map
import numpy_help_cpp
import prog_bar
import python_obj_cpp
import setProcName
import start_cpp

import pydoc


doc = pydoc.HTMLDoc()


# Open the document...
out = open('utils.html','w')
out.write('<html>\n')
out.write('<head>\n')
out.write('<title>Utilities/Miscellaneous</title>\n')
out.write('</head>\n')
out.write('<body>\n')



# Openning blob...
readme = open('readme.txt','r').read()
readme = readme.replace('\n','<br/>')
out.write(doc.bigsection('Overview','#ffffff','#7799ee',readme))


# Variables...
variables = ''
variables += '<strong>numpy_help_cpp.numpy_util_code</strong><br/>'
variables += 'Assorted utility functions for accessing numpy arrays within scipy.weave C++ code.<br/><br/>'
variables += '<strong>python_obj_cpp.python_obj_code</strong><br/>'
variables += 'Assorted utility functions for interfacing with python objects from scipy.weave C++ code.<br/><br/>'
variables = variables.replace('&nbsp;',' ')
out.write(doc.bigsection('Synonyms','#ffffff','#8d50ff',variables))


# Functions...
funcs = ''
funcs += doc.docroutine(cvarray.cv2array)
funcs += doc.docroutine(cvarray.array2cv)
funcs += doc.docroutine(mp_map.repeat)
funcs += doc.docroutine(mp_map.mp_map)
funcs += doc.docroutine(setProcName.setProcName)
funcs += doc.docroutine(start_cpp.start_cpp)
funcs = funcs.replace('&nbsp;',' ')
out.write(doc.bigsection('Functions','#ffffff','#eeaa77',funcs))


# Classes...
classes = ''
classes += doc.docclass(prog_bar.ProgBar)
classes = classes.replace('&nbsp;',' ')
out.write(doc.bigsection('Classes','#ffffff','#ee77aa',classes))


# Close the document...
out.write('</body>\n')
out.write('</html>\n')
out.close()
