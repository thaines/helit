# -*- coding: utf-8 -*-

# Copyright (c) 2010, Tom SF Haines
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#  * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



import sys
import time



class ProgBar:
  """Simple console progress bar class. Note that object creation and destruction matter, as they indicate when processing starts and when it stops."""
  def __init__(self, width = 60, onCallback = None):
    self.start = time.time()
    self.fill = 0
    self.width = width
    self.onCallback = onCallback

    sys.stdout.write(('_'*self.width)+'\n')
    sys.stdout.flush()
    

  def __del__(self):
    self.end = time.time()
    self.__show(self.width)
    sys.stdout.write('\nDone - '+str(self.end-self.start)+' seconds\n\n')
    sys.stdout.flush()

  def callback(self, nDone, nToDo):
    """Hand this into the callback of methods to get a progress bar - it works by users repeatedly calling it to indicate how many units of work they have done (nDone) out of the total number of units required (nToDo)."""
    if self.onCallback:
      self.onCallback()
    n = int(float(self.width)*float(nDone)/float(nToDo))
    n = min((n,self.width))
    if n>self.fill:
      self.__show(n)

  def __show(self,n):
    sys.stdout.write('|'*(n-self.fill))
    sys.stdout.flush()
    self.fill = n
    