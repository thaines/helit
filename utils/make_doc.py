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
import prog_bar
import numpy_help_cpp
import python_obj_cpp
import matrix_cpp
import gamma_cpp
import setProcName
import start_cpp
import make

import doc_gen



# Setup...
doc = doc_gen.DocGen('utils', 'Utilities/Miscellaneous', 'Library of miscellaneous stuff - most modules depend on this.')
doc.addFile('readme.txt', 'Overview')


# Variables...
doc.addVariable('numpy_help_cpp.numpy_util_code', 'Assorted utility functions for accessing numpy arrays within scipy.weave C++ code.')
doc.addVariable('python_obj_cpp.python_obj_code', 'Assorted utility functions for interfacing with python objects from scipy.weave C++ code.')
doc.addVariable('matrix_cpp.matrix_code', 'Matrix manipulation routines for use in scipy.weave C++')
doc.addVariable('gamma_cpp.gamma_code', 'Gamma and related functions for use in scipy.weave C++')


# Functions...
doc.addFunction(make.make_mod)
doc.addFunction(cvarray.cv2array)
doc.addFunction(cvarray.array2cv)
doc.addFunction(mp_map.repeat)
doc.addFunction(mp_map.mp_map)
doc.addFunction(setProcName.setProcName)
doc.addFunction(start_cpp.start_cpp)
doc.addFunction(make.make_mod)


# Classes...
doc.addClass(prog_bar.ProgBar)
doc.addClass(doc_gen.DocGen)
