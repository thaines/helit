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
import pydoc

doc = pydoc.HTMLDoc()


# Open the document...
out = open('video.html','w')
out.write('<html>\n')
out.write('<head>\n')
out.write('<title>Video Node System</title>\n')
out.write('</head>\n')
out.write('<body>\n')


# Openning blob...
readme = open('readme.txt','r').read()
readme = readme.replace('\n','<br/>')
out.write(doc.bigsection('Overview','#ffffff','#7799ee',readme))


# Functions...
funcs = doc.docroutine(video.num_to_seq)
funcs = funcs.replace('&nbsp;',' ')
out.write(doc.bigsection('Functions','#ffffff','#eeaa77',funcs))


# Classes...
classes = ''
classes += doc.docclass(video.Manager)
classes += doc.docclass(video.VideoNode)
classes += doc.docclass(video.Black)
classes += doc.docclass(video.ReadCV)
classes += doc.docclass(video.ReadCV_IS)
classes += doc.docclass(video.Seq)
classes += doc.docclass(video.FrameCrop)
classes += doc.docclass(video.Half)
classes += doc.docclass(video.WriteCV)
classes += doc.docclass(video.WriteFrameCV)
classes += doc.docclass(video.ViewCV)
classes += doc.docclass(video.Record)
classes += doc.docclass(video.Play)
classes += doc.docclass(video.Remap)
classes += doc.docclass(video.DeinterlaceEV)
classes += doc.docclass(video.ColourBias)
classes += doc.docclass(video.ColourUnBias)
classes += doc.docclass(video.LightCorrectMS)
classes += doc.docclass(video.BackSubDP)
classes += doc.docclass(video.OpticalFlowLK)
classes += doc.docclass(video.FiveWord)
classes += doc.docclass(video.ClipMask)
classes += doc.docclass(video.MaskFlow)
classes += doc.docclass(video.MaskFromColour)
classes += doc.docclass(video.Mask_SABS)
classes += doc.docclass(video.MaskStats)
classes += doc.docclass(video.CombineGrid)
classes += doc.docclass(video.RenderDiff)
classes += doc.docclass(video.RenderFlow)
classes += doc.docclass(video.RenderMask)
classes += doc.docclass(video.RenderWord)
classes = classes.replace('&nbsp;',' ')
out.write(doc.bigsection('Classes','#ffffff','#ee77aa',classes))


# Close the document...
out.write('</body>\n')
out.write('</html>\n')
out.close()
