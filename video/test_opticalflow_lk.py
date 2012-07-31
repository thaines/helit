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



man = video.Manager()

vid = video.ReadCV('test.avi')
man.add(vid)

lc = video.LightCorrectMS()
lc.source(0,vid)
man.add(lc)

of = video.OpticalFlowLK()
of.source(0,vid)
of.source(1,lc,2)
man.add(of)

rf = video.RenderFlow(4.0)
rf.source(0,of)
man.add(rf)


winIn = video.ViewCV('Input')
winIn.move(0,0)
winIn.source(0,vid)
man.add(winIn)

winOF = video.ViewCV('Optical Flow')
winOF.move(vid.width()+5,0)
winOF.source(0,rf)
man.add(winOF)


if len(sys.argv)>1: # Any parameter and we save the output.
  cg = video.CombineGrid(2,1)
  cg.source(0,vid)
  cg.source(1,rf)
  man.add(cg)

  out = video.WriteCV('test_opticalflow_lk.avi')
  out.source(0,cg)
  man.add(out)


man.run()
