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

parser.add_option('-o', '--out', dest='outFN', help="Overide default output filenames. Default is '<video_file without extension>/#.png'.", metavar='FILE')
parser.add_option('-c', '--com_count', dest='ccFN', help="Sets an output file into which component count visualisation is writen - must include a #.", metavar='FILE')
parser.add_option('--com_count_log', action='store_true', dest='ccl', default=False, help='Log the average component count for each frame into a csv file, called com_count_log.csv')
parser.add_option('-d', '--deinterlace', action='store_true', dest='deinterlace', default=False, help='Deinterlace the input.')
parser.add_option('-e', '--even-first', action='store_true', dest='even_first', default=False, help='When deinterlacing assume even-first rather than odd-first fields.')
parser.add_option('-s', '--sequence', action='store_true', dest='sequence', default=False, help='Treats the filename as a sequence, where a # indicates where a file number should be (Of arbitrary length, with preceding zeros if you want.) - all files in the sequence will be considered as one long video file, in numerical order (It will not complain about gaps).')
parser.add_option('-f', '--frames', type='int', dest='frames', help='Overide number of frames, incase you only want to do part of the file.', metavar='FRAMES')
parser.add_option('-q', '--quarter', action='store_true', dest='half', default=False, help='Halfs the resolution of the video before procesing (If requested deinterlacing will be done first, obviously.), resulting in their being a quarter of the data.')

(options, args) = parser.parse_args()
if len(args)!=1:
  parser.print_help()
  sys.exit(1)

inFN = args[0]
outFN = filter(lambda c: c!='#', os.path.splitext(inFN)[0]) + '/#.png'
if options.outFN!=None: outFN = options.outFN

ccFN = options.ccFN
if ccFN!=None and '#' not in ccFN: ccFN = None



lumScale = 0.5
noiseFloor = 0.2
baseFrame = 1



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


cb = video.ColourBias(lumScale, noiseFloor, man.getCL())
cb.source(0,vid)
man.add(cb)

lc = video.LightCorrectMS()
lc.source(0,cb)
man.add(lc)

bs = video.BackSubDP(man.getCL())
bs.source(0,cb)
bs.source(1,lc,0)
man.add(bs)

lc.source(1,bs,2) # Calculate lighting change relative to current background estimate


bs.setDP(comp=6, conc=0.01, cap=128.0)
bs.setHackDP(min_weight = 0.0005)
bs.setBP(threshold = 0.4, half_life = 0.05, iters = 2)
bs.setExtraBP(cert_limit = 0.005, change_limit = 0.001, min_same_prob = 0.99, change_mult = 3.0)
bs.setOnlyCL(minSize = 64, maxLayers = 8, itersPerLevel = 2)


mr = video.RenderMask()
mr.source(0,bs)
man.add(mr)

wf = video.WriteFramesCV(outFN, start_frame = baseFrame)
wf.source(0,mr)
man.add(wf)


if ccFN!=None:
  wcc = video.WriteFramesCV(ccFN, start_frame = baseFrame)
  wcc.source(0,bs,3)
  man.add(wcc)


if options.ccl:
  ccl = video.RecordAverage('com_count_log.csv', '%(frame)i, %(r)f\n', 'Frame #, Component count\n', 6.0)
  ccl.source(0,bs,3)
  man.add(ccl)


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
