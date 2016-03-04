#! /usr/bin/env python

# Copyright 2016 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import frf

from utils import doc_gen



# Setup...
doc = doc_gen.DocGen('frf', 'Fast Random Forest', 'Straight random forest implementation, for when you just want a reasonable classifier/regressor.')
doc.addFile('readme.txt', 'Overview')



# Pull in information about the 'types'...
text = []
for summary in frf.Forest.summary_list():
  text.append(summary['name'])
  text.append('code = ' + summary['code'])
  
  text.append(summary['description'])
  text.append('')

doc.addOther('\n'.join(text), 'Summary types', False)


text = []
for summary in frf.Forest.info_list():
  text.append(summary['name'])
  text.append('code = ' + summary['code'])
  
  text.append(summary['description'])
  text.append('')

doc.addOther('\n'.join(text), 'Information types', False)


text = []
for summary in frf.Forest.learner_list():
  text.append(summary['name'])
  text.append('code = ' + summary['code'])
  
  text.append(summary['description'])
  text.append('')

doc.addOther('\n'.join(text), 'Learner types', False)



# Functions...
doc.addFunction(frf.save_forest)
doc.addFunction(frf.load_forest)



# Classes...
doc.addClass(frf.Forest)
doc.addClass(frf.Tree)

