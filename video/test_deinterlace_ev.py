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

dein = video.DeinterlaceEV()
dein.source(0,vid)
man.add(dein)

out1 = video.ViewCV('Deinterlaced')
out1.source(0,dein)
out1.move(0,0)
man.add(out1)

man.run()
