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



import video

man = video.Manager()

vid = video.ReadCV('test.avi')
man.add(vid)

lc = video.LightCorrectMS()
lc.source(0,vid)
man.add(lc)

diff = video.RenderDiff(100.0)
diff.source(0,vid)
diff.source(1,lc,3)
man.add(diff)

out1 = video.ViewCV('Current')
out1.source(0,vid)
out1.move(0,0)
man.add(out1)

out2 = video.ViewCV('Current corrected to previous')
out2.source(0,lc,3)
out2.move(vid.width()+5,0)
man.add(out2)

out3 = video.ViewCV('Difference')
out3.source(0,diff)
out3.move(vid.width()*2+10,0)
man.add(out3)


def func():
  mat = lc.fetch(0)
  print 'correction r=%f, g=%f, b=%f'%(mat[0,0],mat[1,1],mat[2,2])
man.run(callback = func)
