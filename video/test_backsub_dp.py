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
import video

fn = 'test.avi'
if len(sys.argv)>1: fn = sys.argv[1]



lumScale = 0.7
noiseFloor = 0.01
use_rgb = False


man = video.Manager(useCL = True)

if '#' in fn:
  vid = video.ReadCV_IS(fn)
else:
  vid = video.ReadCV(fn)
man.add(vid)

print 'Resolution:', vid.width(), 'X', vid.height()

cb = video.ColourBias(lumScale, noiseFloor, man.getCL())
cb.source(0,vid)
man.add(cb)

if use_rgb: cb = vid

lc = video.LightCorrectMS()
lc.source(0,cb)
man.add(lc)

ucb_lc = video.ColourUnBias(lumScale, noiseFloor, man.getCL())
ucb_lc.source(0,lc,3)
man.add(ucb_lc)

if use_rgb: ucb_lc = lc

diff = video.RenderDiff(64.0)
diff.source(0,vid)
diff.source(1,ucb_lc,3)
man.add(diff)

bs = video.BackSubDP(man.getCL())
bs.source(0,cb)
bs.source(1,lc,0)
man.add(bs)


if 'bw' in sys.argv: bs.setLumOnly()
lc.source(1,bs,2) # Calculate lighting change relative to current background estimate.


mr = video.RenderMask(bgColour=(0.0,0.0,1.0))
mr.source(0,bs)
mr.source(1,vid)
man.add(mr)


ucb_bg = video.ColourUnBias(lumScale, noiseFloor, man.getCL())
ucb_bg.source(0,bs,2)
man.add(ucb_bg)

if use_rgb: ucb_bg = bs


winIn = video.ViewCV('Input')
winIn.move(0,0)
winIn.source(0,vid)
man.add(winIn)

winProb = video.ViewCV('Probability')
winProb.move(vid.width()+5,vid.height()+50)
winProb.source(0,bs,1)
man.add(winProb)

winFore = video.ViewCV('Foreground')
winFore.move(0,vid.height()+50)
winFore.source(0,mr)
man.add(winFore)

winBack = video.ViewCV('Background')
winBack.move(vid.width()+5,0)
winBack.source(0,ucb_bg,2)
man.add(winBack)

winLC = video.ViewCV('Lighting Difference x64')
winLC.source(0,diff)
winLC.move(vid.width()*2+10,0)
man.add(winLC)


if len(sys.argv)>2: # Any parameter beyond a filename and we save the output.
  cg = video.CombineGrid(3,2)
  cg.source(0,vid)
  cg.source(1,bs,2)
  cg.source(2,diff)
  cg.source(3,mr)
  cg.source(4,bs,1)
  man.add(cg)

  out = video.WriteCV('test_backsub_dp.avi')
  out.source(0,cg)
  man.add(out)


man.run(profile=True)
