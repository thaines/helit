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



lumScale = 0.7
noiseFloor = 0.01
quarter = True


man = video.Manager()


vid = video.ReadCamCV()
man.add(vid)

if quarter:
  vid_old = vid
  vid = video.Half()
  vid.source(0,vid_old)
  man.add(vid)
  


cb = video.ColourBias(lumScale, noiseFloor, man.getCL())
cb.source(0,vid)
man.add(cb)

cb_h = video.Half()
cb_h.source(0,cb)
man.add(cb_h)

lc = video.LightCorrectMS()
lc.source(0,cb_h)
man.add(lc)

bs = video.BackSubDP(man.getCL())
bs.source(0,cb)
bs.source(1,lc,0)
man.add(bs)

bs_h = video.Half()
bs_h.source(0,bs,2)
man.add(bs_h)

lc.source(1,bs_h)


mr = video.RenderMask(bgColour=(0.0,0.0,1.0))
mr.source(0,bs)
mr.source(1,vid)
man.add(mr)

ref = video.Reflect()
ref.source(0,mr)
man.add(ref)

out = video.ViewPyGame(320,240)
out.source(0,ref)
man.add(out)


man.run(profile=True)
