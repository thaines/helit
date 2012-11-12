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

from utils import doc_gen



# Setup...
doc = doc_gen.DocGen('video', 'Video Node System', 'Video processing, a node based system')
doc.addFile('readme.txt', 'Overview')


# Variables...
doc.addVariable('MODE_RGB', 'Indicates a connection between nodes that uses a rgb colour stream, for normal video.')
doc.addVariable('MODE_MASK', 'Indicates a connection between nodes that uses a binary stream, for comunicating masks.')
doc.addVariable('MODE_FLOW', 'Indicates a connection between nodes that uses a pair of floating point numbers, for communicating optical flow.')
doc.addVariable('MODE_WORD', 'Indicates a connection between nodes that uses a discrete assignment - often used to indicate some kind of labeling.')
doc.addVariable('MODE_FLOAT', 'Indicates a connection between nodes that uses a float for each pixel - many uses.')
doc.addVariable('MODE_MATRIX', 'Indicates a connection between nodes that sends matrices.')
doc.addVariable('MODE_OTHER', 'Indicates a connection between nodes of an unknown type.')
doc.addVariable('mode_to_string', 'A dictionary indexed by MODE_ variables that provides human readable descriptions.')
doc.addVariable('five_word_colours', 'Default colours to use with the FiveWord and RenderWord classes.')

# Functions...
doc.addFunction(video.num_to_seq)


# Classes...
doc.addClass(video.Manager)
doc.addClass(video.VideoNode)
doc.addClass(video.Black)
doc.addClass(video.ReadCV)
doc.addClass(video.ReadCV_IS)
doc.addClass(video.Seq)
doc.addClass(video.FrameCrop)
doc.addClass(video.Half)
doc.addClass(video.StepScale)
doc.addClass(video.Reflect)
doc.addClass(video.WriteCV)
doc.addClass(video.WriteFramesCV)
doc.addClass(video.WriteFrameCV)
doc.addClass(video.ViewCV)
doc.addClass(video.ViewPyGame)
doc.addClass(video.Record)
doc.addClass(video.Play)
doc.addClass(video.Remap)
doc.addClass(video.DeinterlaceEV)
doc.addClass(video.ColourBias)
doc.addClass(video.ColourUnBias)
doc.addClass(video.LightCorrectMS)
doc.addClass(video.BackSubDP)
doc.addClass(video.OpticalFlowLK)
doc.addClass(video.FiveWord)
doc.addClass(video.ClipMask)
doc.addClass(video.MaskFlow)
doc.addClass(video.MaskFromColour)
doc.addClass(video.Mask_SABS)
doc.addClass(video.MaskStats)
doc.addClass(video.StatsCD)
doc.addClass(video.CombineGrid)
doc.addClass(video.RenderDiff)
doc.addClass(video.RenderFlow)
doc.addClass(video.RenderMask)
doc.addClass(video.RenderWord)
