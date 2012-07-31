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

bs = video.BackSubDP()
bs.source(0,vid)
bs.source(1,lc,0)
man.add(bs)

lc.source(1,bs,2) # Calculate lighting change relative to current background estimate

cm = video.ClipMask(top = 40) # For the mile end data set chops off the problamatic sky.
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


mr = video.RenderMask(bgColour=(0.0,0.0,1.0))
mr.source(0,cm)
mr.source(1,vid)
man.add(mr)

rf = video.RenderFlow(1.0)
rf.source(0,mf)
man.add(rf)

rw = video.RenderWord()
rw.source(0,fw)
man.add(rw)


winIn = video.ViewCV('Input')
winIn.move(0,0)
winIn.source(0,vid)
man.add(winIn)

winFore = video.ViewCV('Foreground')
winFore.move(0,vid.height()+50)
winFore.source(0,mr)
man.add(winFore)

winOF = video.ViewCV('Optical Flow')
winOF.move(vid.width()+5,0)
winOF.source(0,rf)
man.add(winOF)

winWord = video.ViewCV('Words')
winWord.move(vid.width()+5,vid.height()+50)
winWord.source(0,rw)
man.add(winWord)


if len(sys.argv)>1: # Any parameter and we save the output.
  cg = video.CombineGrid(2,2)
  cg.source(0,vid)
  cg.source(1,rf)
  cg.source(2,mr)
  cg.source(3,rw)
  man.add(cg)

  out = video.WriteCV('test_five_word.avi')
  out.source(0,cg)
  man.add(out)


man.run()
