#! /usr/bin/env python

# Copyright 2011 Tom SF Haines

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.



import smp

from utils import doc_gen



# Setup...
doc = doc_gen.DocGen('smp', 'Sparse Multinomial Posterior', 'Estimate a multinomial distribution, given sparse draws')
doc.addFile('readme.txt', 'Overview')


# Variables...
doc.addVariable('smp_code', 'String containing the C++ code that does the actual work for the system.')


# Classes...
doc.addClass(smp.SMP)
doc.addClass(smp.FlagIndexArray)
