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

half = video.Half()
half.source(0,vid)
man.add(half)

out1 = video.ViewCV('Input')
out1.source(0,vid)
man.add(out1)

out2 = video.ViewCV('Halfed')
out2.source(0,half)
man.add(out2)

man.run()
