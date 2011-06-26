#! /usr/bin/env python

# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



from utils.prog_bar import ProgBar

from swood import SWood

import test_model as mod



# Tests the stocahstic woodland class on the model contained within test_model.py

# Parameters...
tree_count = 256
option_count = 4



# Get trainning data...
int_dm, real_dm, cats, weight = mod.generate_train()



# Train...
p = ProgBar()
sw = SWood(int_dm, real_dm, cats, tree_count = tree_count, option_count = option_count, weight = weight, callback=p.callback)
del p

print 'Out-of-bag success rate = %.2f%%'%(100.0*sw. oob_success())
print



# Test...
mod.test(sw.classify)
