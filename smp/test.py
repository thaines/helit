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



import random
import numpy
import numpy.random
import numpy.random.mtrand

from smp import FlagIndexArray, SMP



# Parameters...
size = 4
n_to_keep = 2
draws = 30000 # Number of draws in each draw taken from the multinomial.
samples = 8 # Number of draws from multinomial.
hold_first = True
samCount = 1024


# Generate a multinomial...
mn = numpy.random.mtrand.dirichlet(numpy.ones(size))
print 'Actual multinomial ='
print mn



# Draw some samples, with only a finite number of entrys each...
sam = []
for _ in xrange(samples):
  pos_to_use = range(size)
  if hold_first:
    pos_to_use = pos_to_use[1:]
    random.shuffle(pos_to_use)
    pos_to_use = [0] + pos_to_use
  else:
    random.shuffle(pos_to_use)
  pos_to_use = pos_to_use[:n_to_keep]

  draw_mn = mn[numpy.array(pos_to_use)]
  draw_mn /= draw_mn.sum()
  draw = numpy.random.multinomial(draws,draw_mn)

  counts = numpy.zeros(size,dtype=numpy.int32)
  flags = numpy.zeros(size,dtype=numpy.uint8)

  for a,b in enumerate(pos_to_use):
    counts[b] = draw[a]
    flags[b] = 1

  sam.append((counts,flags))



# Construct the flag index array, put relevant index into each sample...
fia = FlagIndexArray(size)
fia.addSingles()

for i in xrange(len(sam)):
  ind = fia.flagIndex(sam[i][1])
  sam[i] = (sam[i][0],sam[i][1],ind)



# Construct the SMP object...
smp = SMP(fia)
smp.setSampleCount(samCount)

for s in sam:
  smp.add(s[2],s[0])



# Get and print out the mean and its distance from the actual multinomial...
mean = smp.mean()

print 'Mean =', mean
print 'error =', numpy.fabs(mn-mean).sum()/mn.shape[0]
print 'cError =\n', numpy.fabs(mn-mean)
