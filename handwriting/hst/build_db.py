#! /usr/bin/env python

# Copyright 2016 Tom SF Haines

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import sys
import numpy

from utils.prog_bar import ProgBar
from frf import frf

from glyph_db import GlyphDB
from costs import *



# Handle command line input of where to search...
if len(sys.argv)<2:
  print 'Creates a random forest trained on glyph relationships from joined up handwriting, so the relationships can be inferred for print handwriting. Outputs cost_proxy.rf in this directory, so hst knows where to find it.'
  print 'Usage:'
  print 'python build_db.py <dir to search for line graphs>'
  print
  sys.exit(1)

root_dir = sys.argv[1]



# Parameters...
rand_mult = 64
trees = 64



# Get the filenames of each line graph file...
lg_fn = []
for root, _, files in os.walk(root_dir):
  for fn in [fn for fn in files if fn.endswith('.line_graph')]:
    lg_fn.append(os.path.join(root, fn))

if len(lg_fn)==0:
  print 'Failed to find any line graphs in the given directory'
  sys.exit(1)
print 'Found %i line graphs' % len(lg_fn)



# Build a database of features, extracted from each line graph...
train = []

for fn_num, fn in enumerate(lg_fn):
  print 'Processing %s: (%i of %i)' % (fn, fn_num+1, len(lg_fn))
  
  # Load a glyph DB that contains just this database...
  gdb = GlyphDB()
  gdb.add(fn)
  
  # Iterate and select a number of pairs in the file (adjacent, but also do some random ones as well)...
  glyphs = gdb.get_all()
  pairs = dict()
  
  ## Pairs...
  for glyph in glyphs:
    if glyph.right!=None and len(glyph.right[1])!=0 and len(glyph.right[0].left[1])!=0:
      pairs[(id(glyph), id(glyph.right[0]))] = (glyph, glyph.right[0])
  
  ## Random...
  for _ in xrange(len(glyphs)*rand_mult):
    # Random selection...
    left_g = glyphs[numpy.random.randint(len(glyphs))]
    right_g = glyphs[numpy.random.randint(len(glyphs))]
    
    # If they are not the same glyph and tests are passed record...
    if id(left_g)!=id(right_g) and left_g.right!=None and len(left_g.right[1])!=0 and right_g.left!=None and len(right_g.left[1])!=0:
      pairs[(id(left_g), id(right_g))] = (left_g, right_g)
  
  # Go through the selected pairs calculate the feature and the output then add that to the database...
  for left_g, right_g in pairs.itervalues():
    # Calculate the feature...
    feat = glyph_pair_feat(left_g, right_g)
    
    # Calculate the cost and offset...
    cost = end_dist_cost(left_g, right_g, 0.0)
    offset = glyph_pair_offset(left_g, right_g, 1.0)[0]
    
    # Store ready to be fed to the forest of monsters...
    train.append((feat, cost, offset))
print



# Convert into the straight data matrix format...
train_in = numpy.empty((len(train), train[0][0].shape[0]), dtype=numpy.float32)
train_out = numpy.empty((len(train), 2), dtype=numpy.float32)

for i, exemplar in enumerate(train):
  train_in[i,:] = exemplar[0]
  train_out[i,0] = exemplar[1]
  train_out[i,1] = exemplar[2]

print '%i exemplars, containing %i features each' % (train_in.shape[0], train_in.shape[1])
print 'cost range = %.3f - %.3f' % (train_out[:,0].min(), train_out[:,0].max())
print 'offset range = %.3f - %.3f' % (train_out[:,1].min(), train_out[:,1].max())
print


# Learn a random forest, going to both the adjacency cost and the vertical offset...
forest = frf.Forest()
forest.configure('GG', 'GG', 'S' * train_in.shape[1])
forest.opt_features = int(numpy.sqrt(train_in.shape[1]))
forest.set_ratios(numpy.array([[1.0, 0.5]]))

print 'Learning:'
pb = ProgBar()
oob = forest.train(train_in, train_out, trees, pb.callback)
del pb

print 'Tree trained: oob cost error = %.3f, oob offset error = %.3f' % (oob[0], oob[1])



# Save it to disk...
frf.save_forest('cost_proxy.rf', forest)
print 'Saved and done'
print
