#! /usr/bin/env python

# Copyright (c) 2011, Tom SF Haines
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#  * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



import dp_utils

from utils import doc_gen



# Setup...
doc = doc_gen.DocGen('dp_utils', 'Dirichlet Process Utilities', 'Utility library for handling Dirichlet processes')
doc.addFile('readme.txt', 'Overview')


# Variables...
doc.addVariable('sampling_code', 'Code for sampling from various distributions - uniform, Gaussian, gamma and beta.')
doc.addVariable('conc_code', 'Contains code to sample a concentration parameter and two classes - one to represent the status of a concentration parameter - its prior and its estimated value, and another to do the same thing for when a concentration parameter is shared between multiple Dirichlet processes.')
doc.addVariable('dir_est_code', 'Contains a class for doing maximum likelihood estimation of a Dirichlet distrbution given multinomials that have been drawn from it.')
doc.addVariable('linked_list_code', 'A linked list implimentation - doubly linked, adds data via templated inheritance.')
doc.addVariable('linked_list_gc_code', 'A linked list with reference counting and garabge collection for its entries. Happens to be very good at representing a Dirichlet process.')
doc.addVariable('dp_utils_code', 'Combines all of the code provided in this module into a single variable.')
