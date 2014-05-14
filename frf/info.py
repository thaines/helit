#! /usr/bin/env python

# Copyright 2014 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import frf



print 'Summary types:'
for summary in frf.Forest.summary_list():
  print '  ' + summary['name'] + ':'
  print '    ' + 'code = ' + summary['code']
  
  d = summary['description']
  for i in xrange(0, len(d), 60):
    print '    ' + d[i:i+60].strip()
    
  print
print



print 'Information types:'
for info in frf.Forest.info_list():
  print '  ' + info['name'] + ':'
  print '    ' + 'code = ' + info['code']
  
  d = info['description']
  for i in xrange(0, len(d), 60):
    print '    ' + d[i:i+60].strip()
    
  print
print



print 'Learner types:'
for learner in frf.Forest.learner_list():
  print '  ' + learner['name'] + ':'
  print '    ' + 'code = ' + learner['code']
  print '    ' + 'test = ' + learner['test']
  
  d = learner['description']
  for i in xrange(0, len(d), 60):
    print '    ' + d[i:i+60].strip()
    
  print
print
