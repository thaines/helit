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



lumScale = 0.5
noiseFloor = 0.2


man = video.Manager()

vid = video.ReadCV('test.avi')
man.add(vid)

cb = video.ColourBias(lumScale, noiseFloor)
cb.source(0,vid)
man.add(cb)

cb = vid # Uncoment to switch off the colour model.

lc = video.LightCorrectMS()
lc.source(0,cb)
man.add(lc)

ucb_lc = video.ColourUnBias(lumScale, noiseFloor)
ucb_lc.source(0,lc,3)
man.add(ucb_lc)

ucb_lc = lc # Uncoment to switch off the colour model.

diff = video.RenderDiff(128.0)
diff.source(0,vid)
diff.source(1,ucb_lc,3)
man.add(diff)

bs = video.BackSubDP(man.getCL())
bs.source(0,cb)
bs.source(1,lc,0)
man.add(bs)

#bs.setConComp(16)
#bs.setBP(iters = 0)


lc.source(1,bs,2) # Calculate lighting change relative to current background estimate.

cm = video.ClipMask(top = 40) # For the mile end data set chops off the problamatic sky.
cm.source(0,bs)
man.add(cm)

mr = video.RenderMask(bgColour=(0.0,0.0,1.0))
mr.source(0,cm)
mr.source(1,vid)
man.add(mr)


ucb_bg = video.ColourUnBias(lumScale, noiseFloor)
ucb_bg.source(0,bs,2)
man.add(ucb_bg)

ucb_bg = bs # Uncoment to switch off the colour model.


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

winLC = video.ViewCV('Lighting Difference x128')
winLC.source(0,diff)
winLC.move(vid.width()*2+10,0)
man.add(winLC)


if len(sys.argv)>1: # Any parameter and we save the output.
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
