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
import os.path
from optparse import OptionParser
import time

import video



# Handle the command line agruments...
parser = OptionParser(usage='usage: %prog [options] video_file', version='%prog 0.1')

parser.add_option('-o', '--out', dest='outFN', help="Overide default output filename. (Default is the input filename with its extension replaced by '.word'.", metavar='FILE')
parser.add_option('-d', '--deinterlace', action='store_true', dest='deinterlace', default=False, help='Deinterlace the input.')
parser.add_option('-e', '--even-first', action='store_true', dest='even_first', default=False, help='When deinterlacing assume even-first rather than odd-first fields.')
parser.add_option('-s', '--sequence', action='store_true', dest='sequence', default=False, help='Treats the filename as a sequence, where a # indicates where a file number should be (Of arbitrary length, with preceding zeros if you want.) - all files in the sequence will be considered as one long video file, in numerical order (It will not complain about gaps).')
parser.add_option('-f', '--frames', type='int', dest='frames', help='Overide number of frames, incase you only want to do part of the file. Note the frame count will not be updated - mostly good for testing.', metavar='FRAMES')
parser.add_option('-c', '--clip', type='int', dest='clip', help='Clips off the top of the image, remove such areas from the foreground mask. Useful for removing problematic skys and distant irrelevant activities.', metavar='PIXELS', default=0)
parser.add_option('-q', '--quater', action='store_true', dest='half', default=False, help='Halfs the resolution of the video before procesing (If requested deinterlacing will be done first, obviously.), resulting in their being a quarter of the data.')

(options, args) = parser.parse_args()
if len(args)!=1:
  parser.print_help()
  sys.exit(1)

inFN = args[0]
outFN = os.path.splitext(inFN)[0] + '.word'
if options.outFN!=None: outFN = options.outFN



# Build the node tree...
man = video.Manager()


if options.sequence: vid = video.num_to_seq(inFN, video.ReadCV)
else: vid = video.ReadCV(inFN)
man.add(vid)


if options.deinterlace:
  ivid = vid
  vid = video.DeinterlaceEV(not options.even_first)
  vid.source(0,ivid)
  man.add(vid)

if options.half:
  jvid = vid
  vid = video.Half()
  vid.source(0,jvid)
  man.add(vid)


lc = video.LightCorrectMS()
lc.source(0,vid)
man.add(lc)

bs = video.BackSubDP()
bs.source(0,vid)
bs.source(1,lc,0)
man.add(bs)

lc.source(1,bs,2) # Calculate lighting change relative to current background estimate

cm = video.ClipMask(top = options.clip)
cm.source(0,bs)
man.add(cm)

of = video.OpticalFlowLK()
of.source(0,vid)
of.source(2,cm)
man.add(of)

mf = video.MaskFlow()
mf.source(0,of)
mf.source(1,cm)
man.add(mf)

fw = video.FiveWord()
fw.source(0,of)
fw.source(1,cm)
man.add(fw)

r = video.Record(fw, outFN)
man.add(r)



# Run....
frames = vid.frameCount()
if options.frames!=None: frames = options.frames

lastTime = time.clock()
for i in xrange(frames):
  if man.nextFrame()!=True:
    print 'Error getting next frame'
    break

  now = time.clock()
  print 'Frame %i of %i, time = %f'%(i+1,frames,now-lastTime)
  lastTime = now
