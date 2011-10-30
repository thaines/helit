#! /usr/bin/env python

# Copyright (c) 2011, Tom SF Haines
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#  * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



import dp_utils

import pydoc


doc = pydoc.HTMLDoc()


# Open the document...
out = open('dp_utils.html','w')
out.write('<html>\n')
out.write('<head>\n')
out.write('<title>Dirichlet Process Utilities</title>\n')
out.write('</head>\n')
out.write('<body>\n')



# Openning blob...
readme = open('readme.txt','r').read()
readme = readme.replace('\n','<br/>')
out.write(doc.bigsection('Overview','#ffffff','#7799ee',readme))


# Variables...
variables = ''
variables += '<strong>funcs_code</strong><br/>'
variables += '<br/>Defines a selection of gamma-related functions, specifically ln-gamma, di-gamma and tri-gamma.<br/><br/>'
variables += '<strong>sampling_code</strong><br/>'
variables += '<br/>Code for sampling from various distributions - uniform, Gaussian, gamma and beta.<br/><br/>'
variables += '<strong>conc_code</strong><br/>'
variables += '<br/>Contains code to sample a concentration parameter and two classes - one to represent the status of a concentration parameter - its prior and its estimated value, and another to do the same thing for when a concentration parameter is shared between multiple Dirichlet processes.<br/><br/>'
variables += '<strong>dir_est_code</strong><br/>'
variables += '<br/>Contains a class for doing maximum likelihood estimation of a Dirichlet distrbution given multinomials that have been drawn from it.<br/><br/>'
variables += '<strong>linked_list_code</strong><br/>'
variables += '<br/>A linked list implimentation - doubly linked, adds data via templated inheritance.<br/><br/>'
variables += '<strong>linked_list_gc_code</strong><br/>'
variables += '<br/>A linked list with reference counting and garabge collection for its entries. Happens to be very good at representing a Dirichlet process.<br/><br/>'
variables += '<strong>dp_utils_code</strong><br/>'
variables += '<br/>Combines all of the code provided in this module into a single variable.<br/><br/>'
variables = variables.replace('&nbsp;',' ')
out.write(doc.bigsection('Variables','#ffffff','#8d50ff',variables))


# Close the document...
out.write('</body>\n')
out.write('</html>\n')
out.close()
