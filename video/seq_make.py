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



import os
import os.path

from seq import Seq



def num_to_seq(fn, loader):
  """Given a filename of the form 'directory/start#end' finds all files that match the given form, where # is an arbitrary number. The files are sorted into numerical order, and each is loaded using the provided loader (ReadCV for instance - constructor should take a single filename.), a Seq object is then created. This in effect turns a directory of numbered video files into a single video file."""
  # Break the given filename form into its basic parts...
  path, fn = os.path.split(fn)
  start, end = map(lambda s: s.replace('#',''),fn.split('#',1))

  # Get all files from the directory, filter it down to only those that match the form...
  files = os.listdir(path)
  def valid(fn):
    if fn[:len(start)]!=start: return False
    if fn[-len(end):]!=end: return False
    if not fn[len(start):-len(end)].isdigit(): return False
    return True
  files = filter(valid,files)

  # Get the relevant numbers, sort the files by them...
  files.sort(key=lambda fn: int(fn[len(start):-len(end)]))

  # Put the paths back...
  files = map(lambda f: os.path.join(path, f), files)

  # Loop and load the files...
  seq = map(lambda fn: loader(fn), files)

  # Create and return the sequence object...
  return Seq(seq)
