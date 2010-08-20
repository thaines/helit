#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2010, Tom SF Haines
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#  * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



from ctypes import *



def setProcName(name):
  """Sets the process name, linux only - useful for those programs where you might want to do a killall, but don't want to slaughter all the other python processes. Note that there are multiple mechanisms, and that the given new name can be shortened by differing amounts in differing cases."""
  # Call the process control function...
  libc = cdll.LoadLibrary('libc.so.6')
  libc.prctl(15, c_char_p(name), 0, 0, 0)

  # Update argv...
  charPP = POINTER(POINTER(c_char))
  argv = charPP.in_dll(libc,'_dl_argv')
  size = libc.strlen(argv[0])
  libc.strncpy(argv[0],c_char_p(name),size)



if __name__=='__main__':
  # Quick test that it works...
  import os
  ps1 = 'ps'
  ps2 = 'ps -f'

  os.system(ps1)
  os.system(ps2)
  setProcName('wibble_wobble')
  os.system(ps1)
  os.system(ps2)
