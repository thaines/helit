# Copyright 2016 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import os
import os.path
import random
import time



class LockFile:
  """Rather simple lock file system, based on lock directories (as directory creation is atomic) - whilst crude this is about as safe a locking system as its possible to create."""
  def __init__(self, fn, mode='r'):
    self.fn = fn
    self.mode = mode
  
  
  def __enter__(self):
    # Keep trying to create a lock directory - this is safe on all platforms as directory creation is always atomic...
    # Calculate lock dircectory name...
    head, tail = os.path.split(self.fn)
    self.lock_dir = os.path.join(head, '.%s_lock' % tail)
    
    # Get the lock...
    while True:
      try:
        os.mkdir(self.lock_dir)
        break
      except OSError:
        # Its locked - sleep for a slightly random period of milliseconds...
        time.sleep(1e-3 * random.randrange(1,9))
    
    # We are safe - open the file...
    try:
      self.f = open(self.fn, self.mode)
    except IOError:
      self.f = None
    
    return self.f
  
  
  def __exit__(self, etype, value, traceback):
    # Close file...
    if self.f!=None:
      self.f.close()
    
    # Terminate lock...
    os.rmdir(self.lock_dir)
    
    return etype==None



if '__main__'==__name__:
  with LockFile('test1.txt','w') as f:
    f.write('Hello\n')
  
  try:
    with LockFile('test2.txt','w') as f:
      f.write('Hello again\n')
      raise LookupError
  except LookupError:
    print 'Lookup error received - success'
