# -*- coding: utf-8 -*-

# Copyright (c) 2010, Tom SF Haines
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#  * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



import inspect
import hashlib



def start_cpp(hash_str = None):
  """This method does two things - firstly it adds the correct line numbers to scipy.weave code (Good for debugging) and secondly it can optionaly inserts a hash code of some other code into the code. This latter feature is useful for working around the fact the scipy.weave only recompiles if the hash of the code changes, but ignores the support_code - passing the support_code into start_cpp avoids this problem by putting its hash into the code and forcing a recompile when that code changes. Usage is <code variable> = start_cpp([support_code variable]) + <3 quotations to start big comment with code in, typically going over many lines.>"""
  frame = inspect.currentframe().f_back
  info = inspect.getframeinfo(frame)
  if hash_str==None:
    return '#line %i "%s"\n'%(info[1],info[0])
  else:
    h = hashlib.md5()
    h.update(hash_str)
    hash_val = h.hexdigest()
    return '#line %i "%s" // %s\n'%(info[1],info[0],hash_val)
