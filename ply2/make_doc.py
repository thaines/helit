#! /usr/bin/env python

# Copyright (c) 2016, Tom SF Haines
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#  * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import ply2

from utils import doc_gen



# Setup...
doc = doc_gen.DocGen('ply2', 'Ply 2', 'Simple generic data file format, for filling in the gap between json and hdf5.')
doc.addFile('readme.txt', 'Overview')



# Add all of the functions...
doc.addFunction(ply2.write)
doc.addFunction(ply2.read)

doc.addFunction(ply2.create)
doc.addFunction(ply2.verify)

doc.addFunction(ply2.encoding_to_dtype)
doc.addFunction(ply2.array_to_encoding)

doc.addFunction(ply2.to_meta_line)
doc.addFunction(ply2.read_meta_line)
doc.addFunction(ply2.to_element_line)



# Throw in the specification in for completeness...
doc.addFile('specification.txt', 'Specification')
