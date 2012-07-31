#! /usr/bin/env python

# Copyright 2012 Tom SF Haines

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



import sys
from optparse import OptionParser

import video


# Handle options...
parser = OptionParser(usage='usage: %prog video_file', version='%prog 0.1')

parser.add_option('-s', '--start', type='int', dest='start', help='Frame to start playback on', metavar='FRAME', default=0)

(options, args) = parser.parse_args()
if len(args)!=1:
  parser.print_help()
  sys.exit(1)

inFN = args[0]



# Build the node tree...
man = video.Manager()


p = video.Play(inFN)

fc = video.FrameCrop(p, options.start)
man.add(fc)

rw = video.RenderWord(video.five_word_colours)
rw.source(0,fc)
man.add(rw)

winWord = video.ViewCV('Words')
winWord.source(0,rw)
man.add(winWord)



# go...
man.run()
