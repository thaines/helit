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



import ddhdp

from utils import doc_gen



# Setup...
doc = doc_gen.DocGen('ddhdp', 'Delta-Dual Hierarchical Dirichlet Processes', 'Semi-supervised topic model, with clustering')
doc.addFile('readme.txt', 'Overview')


# Functions...
doc.addFunction(ddhdp.getAlgorithm)


# Classes...
doc.addClass(ddhdp.PriorConcDP)
doc.addClass(ddhdp.Params)
doc.addClass(ddhdp.Document)
doc.addClass(ddhdp.Corpus)
doc.addClass(ddhdp.DocSample)
doc.addClass(ddhdp.Sample)
doc.addClass(ddhdp.Model)
doc.addClass(ddhdp.DocModel)
