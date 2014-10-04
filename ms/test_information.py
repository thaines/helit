#! /usr/bin/env python

# Copyright 2014 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import numpy
from ms import MeanShift

import cv
from utils.cvarray import *



# Some letters, because I feel like having some fun, or at least being silly...
alphabet = dict()

alphabet['A'] = ['  #  ',
                 ' # # ',
                 '#   #',
                 '#   #',
                 '#####',
                 '#   #',
                 '#   #']

alphabet['B'] = ['#### ',
                 '#   #',
                 '#   #',
                 '#### ',
                 '#   #',
                 '#   #',
                 '#### ']

alphabet['C'] = [' ### ',
                 '#   #',
                 '#    ',
                 '#    ',
                 '#    ',
                 '#   #',
                 ' ### ']
                 
alphabet['D'] = ['#### ',
                 '#   #',
                 '#   #',
                 '#   #',
                 '#   #',
                 '#   #',
                 '#### ']

alphabet['E'] = ['#####',
                 '#    ',
                 '#    ',
                 '###  ',
                 '#    ',
                 '#    ',
                 '#####']

alphabet['F'] = ['#####',
                 '#    ',
                 '#    ',
                 '#### ',
                 '#    ',
                 '#    ',
                 '#    ']

alphabet['G'] = [' ### ',
                 '#   #',
                 '#    ',
                 '#    ',
                 '#  ##',
                 '#   #',
                 ' ### ']

alphabet['H'] = ['#   #',
                 '#   #',
                 '#   #',
                 '#####',
                 '#   #',
                 '#   #',
                 '#   #']

alphabet['I'] = [' ### ',
                 '  #  ',
                 '  #  ',
                 '  #  ',
                 '  #  ',
                 '  #  ',
                 ' ### ']

alphabet['J'] = [' ####',
                 '    #',
                 '    #',
                 '    #',
                 '    #',
                 ' #  #',
                 '  ## ']

alphabet['K'] = ['#   #',
                 '#  # ',
                 '# #  ',
                 '###  ',
                 '#  # ',
                 '#   #',
                 '#   #']
                 
alphabet['L'] = [' #   ',
                 ' #   ',
                 ' #   ',
                 ' #   ',
                 ' #   ',
                 ' #   ',
                 ' ####']

alphabet['M'] = ['#   #',
                 '## ##',
                 '# # #',
                 '# # #',
                 '#   #',
                 '#   #',
                 '#   #']
                 
alphabet['N'] = ['#   #',
                 '##  #',
                 '##  #',
                 '# # #',
                 '#  ##',
                 '#  ##',
                 '#   #']
                 
alphabet['O'] = [' ### ',
                 '#   #',
                 '#   #',
                 '#   #',
                 '#   #',
                 '#   #',
                 ' ### ']

alphabet['P'] = ['#### ',
                 '#   #',
                 '#   #',
                 '#### ',
                 '#    ',
                 '#    ',
                 '#    ']

alphabet['Q'] = [' ### ',
                 '#   #',
                 '#   #',
                 '#   #',
                 '# # #',
                 ' ### ',
                 '    #']

alphabet['R'] = ['#### ',
                 '#   #',
                 '#   #',
                 '#### ',
                 '# #  ',
                 '#  # ',
                 '#   #']

alphabet['S'] = [' ### ',
                 '#   #',
                 '#    ',
                 ' ### ',
                 '    #',
                 '#   #',
                 ' ### ']

alphabet['T'] = ['#####',
                 '  #  ',
                 '  #  ',
                 '  #  ',
                 '  #  ',
                 '  #  ',
                 '  #  ']

alphabet['U'] = ['#   #',
                 '#   #',
                 '#   #',
                 '#   #',
                 '#   #',
                 '#   #',
                 ' ### ']

alphabet['V'] = ['#   #',
                 '#   #',
                 '#   #',
                 '#   #',
                 ' # # ',
                 ' # # ',
                 '  #  ']

alphabet['W'] = ['#   #',
                 '#   #',
                 '#   #',
                 '#   #',
                 '# # #',
                 '# # #',
                 ' # # ']

alphabet['X'] = ['#   #',
                 '#   #',
                 ' # # ',
                 '  #  ',
                 ' # # ',
                 '#   #',
                 '#   #']

alphabet['Y'] = ['#   #',
                 '#   #',
                 '#   #',
                 ' ####',
                 '    #',
                 '#   #',
                 ' ### ']

alphabet['Z'] = ['#####',
                 '    #',
                 '   # ',
                 '  #  ',
                 ' #   ',
                 '#    ',
                 '#####']

alphabet[' '] = ['     ',
                 '     ',
                 '     ',
                 '     ',
                 '     ',
                 '     ',
                 '     ']

alphabet['.'] = ['     ',
                 '     ',
                 '     ',
                 '     ',
                 '     ',
                 '     ',
                 '   # ']



# Create KDEs for each entry...
blank_weight = 0.001

def to_kde(grid):
  data = numpy.empty((7, 5), dtype=numpy.float32)
  for y in xrange(7):
    for x in xrange(5):
      data[y,x] = 1.0 if grid[y][x]!=' ' else blank_weight
  
  ret = MeanShift()
  ret.set_data(data, 'bb', 2)
  ret.set_kernel('triangular')
  ret.set_spatial('kd_tree')
  ret.scale_loo_nll()
  
  return ret

distributions = dict()
for key, value in alphabet.iteritems():
  distributions[key] = to_kde(value)

print 'Generated glyphs'



# Draw from the distributions to create a new set of denser KDEs...
samples = 256

def resample(kde):
  data = kde.draws(samples)
  
  ret = MeanShift()
  ret.set_data(data, 'df')
  ret.set_kernel('triangular')
  ret.set_spatial('kd_tree')
  ret.scale_loo_nll()
  
  return ret

glyphs = dict()
for key, kde in distributions.iteritems():
  glyphs[key] = resample(kde)

print 'Resampled to create glyphs'



# Render them out... because I want to see what they look like... ;-)
pangram = 'Zelda might fix the job growth plans very quickly on Monday.'
pangram = pangram.upper()

data = []
for i, char in enumerate(pangram):
  d = distributions[char].draws(samples)
  d[:,1] += i * 7.0
  data.append(d)

data = numpy.concatenate(data, axis=0)

scale = 8.0
low_y = data[:,0].min() - 1.0
high_y = data[:,0].max() + 1.0
low_x = data[:,1].min() - 1.0
high_x = data[:,1].max() + 1.0

height = int(scale * (high_y - low_y))
width = int(scale * (high_x - low_x))

image = numpy.zeros((height, width, 3), dtype=numpy.float32)
image[numpy.asarray(scale * (data[:,0] - low_y), dtype=numpy.int32), numpy.asarray(scale * (data[:,1] - low_x), dtype=numpy.int32), :] = 1.0

image = array2cv(image*255.0)
cv.SaveImage('pangram.png', image)

print 'Saved pangram image'
print



# Calculate the entropy of each letter - no idea what this means, but I know what theses values should like like, so its a good test of entropy...
print 'Letter entropies:'
for letter, kde in distributions.iteritems():
  rekde = glyphs[letter]
  print "  '%s': entropy (base) = %.3f nats, entropy (redraw) = %.3f nats" % (letter, kde.entropy(), rekde.entropy())
print



# Calculate the KL-divergence for every letter combination, printing out for each letter the one that best approximates the given...
print 'Most similar letter by KL-divergence:'
for char in glyphs.iterkeys():
  best = None
  
  for partner in glyphs.iterkeys():
    if partner==char:
      continue
    
    kl = glyphs[char].kl(glyphs[partner])
    
    if best==None or best_kl>kl:
      best = partner
      best_kl = kl
  
  print "'%s': Most similar by KL = '%s', with a KL of %f nats" % (char, best, best_kl)
print
