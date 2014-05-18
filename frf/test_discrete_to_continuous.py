#! /usr/bin/env python

# Copyright 2014 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import frf

import numpy



# Toy problem, based on estimating the review score of a film given discrete attributes...



attributes = [('nolan', 0.2, 1.8), ('whedon', 0.3, 1.5), ('zimmer', 0.8, 0.0), ('boobies', 0.5, 0.9), ('nudity', 0.4, 0.3), ('sex_scene', 0.2, 0.4), ('bad_dialog', 0.4, -2.5), ('violence', 0.9, 0.2), ('gandalf', 0.1, 2.0), ('spaceship', 0.3, 1.5), ('explosions', 0.7, 0.5), ('car_chase', 0.6, 1.0), ('zombies', 0.5, 0.3), ('high_fashion', 0.2, 0.0), ('decapitation', 0.1, 0.2), ('exploding_heads', 0.01, 0.5), ('witty_midget', 0.3, 2.0), ('hitlar_in_a_leotard', 0.001, -3.0), ('crappy_acting', 0.3, -2.0)]

attr_index = map(lambda p: p[0], attributes)



def draw_film():
  """Simply draw 0 for not in the film, 1 for in the film, return feature vector.  Blah."""
  ret = numpy.zeros(len(attributes), dtype=numpy.int32)
  
  for i in xrange(ret.shape[0]):
    if numpy.random.random() < attributes[i][1]:
      ret[i] = 1
  
  # Some modifications - certain combinations are not allowed...
  if ret[attr_index.index('sex_scene')]!=0:
    ret[attr_index.index('nudity')] = 1
  
  if ret[attr_index.index('zombies')]!=0:
    ret[attr_index.index('violence')] = 1

  if ret[attr_index.index('exploding_heads')]!=0:
    ret[attr_index.index('explosions')] = 1
    ret[attr_index.index('decapitation')] = 1
  
  if ret[attr_index.index('spaceship')]!=0:
    ret[attr_index.index('explosions')] = 1
    
  if ret[attr_index.index('whedon')]!=0:
    ret[attr_index.index('bad_dialog')] = 0

  return ret


  
extra_scoring = {('car_chase', 'decapitation') : 0.3, ('bad_dialog', 'spaceship') : 1.0, ('boobies', 'crappy_acting') : 2.0, ('sex_scene', 'gandalf') : -1.0, ('witty_midget', 'sex_scene') : 0.5, ('whedon', 'exploding_heads', 'hitlar_in_a_leotard') : 15.0}



def rate_film(film):
  """My totally logical film rating system... happens to be the worst kind of system to throw at a random forest."""
  ret = 0.0
  
  # Unary terms...
  for i, attr in enumerate(attributes):
    if film[i]!=0:
      ret += attr[2]
  
  # Extra terms...
  for terms, offset in extra_scoring.iteritems():
    do_it = True
    for term in terms:
      i = attr_index.index(term)
      if film[i]==0:
        do_it = False
        break
    if do_it:
      ret += offset

  return ret

  

# Create data set...
dm = numpy.empty((1024, len(attributes)), dtype=numpy.int32)
rating = numpy.empty(dm.shape[0], dtype=numpy.float32)

for i in xrange(dm.shape[0]):
  dm[i, :] = draw_film()
  rating[i] = rate_film(dm[i,:])

print 'Rating: min = %f, mean = %f, max = %f' % (rating.min(), rating.mean(), rating.max())


# Train a random forest...
forest = frf.Forest()
forest.configure('G', 'G', 'O' * len(attributes))
forest.min_exemplars = 2

oob = forest.train(dm, rating, 16)

print 'Made forest:'
for i in xrange(len(forest)):
  if oob!=None:
    extra = ', oob = %s' % str(oob[i,:])
  else:
    extra = ''
    
  print '  Tree %i: %i bytes, %i nodes%s' % (i, forest[i].size, forest[i].nodes(), extra)
print



# Test...
dm = numpy.empty((256, len(attributes)), dtype=numpy.int32)
rating = numpy.empty(dm.shape[0], dtype=numpy.float32)

for i in xrange(dm.shape[0]):
  dm[i, :] = draw_film()
  rating[i] = rate_film(dm[i,:])

  
print 'Test average error:'
res = forest.predict(dm)[0]
print numpy.fabs(rating - res['mean']).mean()
