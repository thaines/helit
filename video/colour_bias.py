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



import os.path
import math
import numpy
import numpy.linalg

try:
  import scipy.weave as weave
except:
  weave = None

from utils.start_cpp import start_cpp
from utils.make import make_mod

from video_node import *



# The OpenCL version - try to compile it, and use it if at all possible...
try:
  make_mod('colour_bias_cl', os.path.dirname(__file__), ['manager_cl.h', 'open_cl_help.h', 'colour_bias_cl.c'], openCL=True)
  import colour_bias_cl
except:
  colour_bias_cl = None



class ColourBias(VideoNode):
  """This converts rgb colour to a luminance/chromaticity colour space - nothing special, specific or well defined. Its trick is that you can choose the scale of the luminance channel, to adjust distance in the space to emphasis differences in colour or lightness. Luminance is put into the red channel with chromaticity in green and blue. Done such that the volume of the colour space always remains as 1."""
  def __init__(self, lumScale = 1.0, noiseFloor = 0.2, cl = None):
    """lumScale is how much to bias the luminance - high makes luminance matter, low makes chromaticity matter; 1 is no bias at all. noiseFloor stabalises the variance when it gets darker than the provided value, to prevent noise resulting in unstable chromaticity information - it must not exceed sqrt(3)/3. cl is the return value from getCL() on the manager object - if not provided OpenCL will not be used."""
    self.chromaScale = 0.7176 * math.pow(1.0/lumScale,1.0/3.0)
    self.lumScale = lumScale * self.chromaScale

    self.noiseFloor = noiseFloor

    self.video = None
    self.channel = 0

    self.output = None
    self.inter = None

    # Generate the rotation matrix...
    sr2 = math.sqrt(1.0/2.0)
    sr3 = math.sqrt(1.0/3.0)
    sr6 = math.sqrt(1.0/6.0)
    self.matrix = numpy.array([[sr3,sr3,sr3],[0,sr2,-sr2],[-2.0*sr6,sr6,sr6]])

    self.cl = cl if colour_bias_cl!=None else None
    self.cb_cl = None


  def width(self):
    return self.video.width()

  def height(self):
    return self.video.height()

  def fps(self):
    return self.video.fps()

  def frameCount(self):
    return self.video.frameCount()


  def inputCount(self):
    return 1

  def inputMode(self, channel=0):
    return MODE_RGB

  def inputName(self, channel=0):
    return 'Input rgb video stream.'

  def source(self, toChannel, video, videoChannel=0):
    self.video = video
    self.channel = videoChannel


  def dependencies(self):
    return [self.video]

  def nextFrame(self):
    # Fetch the frame...
    img = self.video.fetch(self.channel)
    if img==None:
      self.output = None
      return False

    # Handle the OpenCL object creation if needed...
    if self.cb_cl==None and self.cl!=None:
      try:
        self.cb_cl = colour_bias_cl.ColourBiasCL(self.cl, self.width(), self.height(), os.path.abspath(os.path.dirname(__file__)))

        self.cb_cl.chromaScale = self.chromaScale
        self.cb_cl.lumScale = self.lumScale
        self.cb_cl.noiseFloor = self.noiseFloor
        self.cb_cl.post_change()
      except:
        self.cl = None

    # Check we have a copy to mess with...
    if self.output==None:
      self.output = img.copy()
      if weave==None: self.inter = img.copy()

    # Run using OpenCl if possible...
    if self.cb_cl!=None:
      self.cb_cl.from_rgb(img)
      self.cb_cl.fetch(self.output)
    else:
      # Do the conversion - per pixel matrix multiplication, a division to seperate chromacity and a scalling to give it a volume of 1, plus the luminance bias term...
      if weave:
        code = start_cpp() + """
        for (int y=0;y<Noutput[0];y++)
        {
         for (int x=0;x<Noutput[1];x++)
         {
          // Rotate luminance into channel 0 (A matrix multiplication.)...
           for (int r=0;r<3;r++)
           {
            OUTPUT3(y,x,r) = 0.0;
            for (int c=0;c<3;c++)
            {
             OUTPUT3(y,x,r) += MATRIX2(r,c) * IMG3(y,x,c);
            }
           }

          // Seperate chromacity, taking into account the noise floor and apply the luminance bias, which includes scaling to a volume of 1...
           float cDiv = OUTPUT3(y,x,0);
           if (cDiv<noiseFloor) cDiv = noiseFloor;

           OUTPUT3(y,x,0) *= lumScale;
           OUTPUT3(y,x,1) *= chromaScale / cDiv;
           OUTPUT3(y,x,2) *= chromaScale / cDiv;
         }
        }
        """

        output = self.output
        matrix = self.matrix
        lumScale = self.lumScale
        chromaScale = self.chromaScale
        noiseFloor = self.noiseFloor

        weave.inline(code, ['img', 'output', 'matrix', 'lumScale', 'chromaScale', 'noiseFloor'])
      else:
        ## Apply the rotation matrix...
        for r in xrange(3):
          self.inter[:,:,r] = (img * self.matrix[r,:].reshape((1,1,-1))).sum(axis=2)

        ## Divide by luminance to seperate chromacity...
        self.inter[:,:,1] /= numpy.clip(self.inter[:,:,0], self.noiseFloor, 1e100)
        self.inter[:,:,2] /= numpy.clip(self.inter[:,:,0], self.noiseFloor, 1e100)

        # Rescale to get the distance biases...
        self.output[:,:,0]  = self.inter[:,:,0]  * self.lumScale
        self.output[:,:,1:] = self.inter[:,:,1:] * self.chromaScale

    return True


  def outputCount(self):
    return 2

  def outputMode(self, channel=0):
    return MODE_RGB

  def outputName(self, channel=0):
    if channel==0: return 'Video stream with biased colour space.'
    else: return 'Same as cahnnel 0, but with [0..1] normalisation - for visualisation.'

  def fetch(self, channel=0):
    if channel==0: return self.output
    else: return self.inter



class ColourUnBias(VideoNode):
  """Does the exact opposite of ColourBias, assuming you provide the exact same parameters."""
  def __init__(self, lumScale = 1.0, noiseFloor = 0.2, cl = None):
    """lumScale is how much to bias the luminance - high makes luminance matter, low makes chromaticity matter. cl is the return value from getCL() on the manager object - if not provided OpenCL will not be used."""
    self.chromaScale = 0.7176 * math.pow(1.0/lumScale,1.0/3.0)
    self.lumScale = lumScale * self.chromaScale

    self.noiseFloor = noiseFloor

    self.video = None
    self.channel = 0

    self.output = None
    self.inter = None

    # We are going to need the inverse of the matrix...
    sr2 = math.sqrt(1.0/2.0)
    sr3 = math.sqrt(1.0/3.0)
    sr6 = math.sqrt(1.0/6.0)
    matrix = numpy.array([[sr3,sr3,sr3],[0,sr2,-sr2],[-2.0*sr6,sr6,sr6]])

    self.invMatrix = numpy.linalg.inv(matrix)

    self.cl = cl if colour_bias_cl!=None else None
    self.cb_cl = None


  def width(self):
    return self.video.width()

  def height(self):
    return self.video.height()

  def fps(self):
    return self.video.fps()

  def frameCount(self):
    return self.video.frameCount()


  def inputCount(self):
    return 1

  def inputMode(self, channel=0):
    return MODE_RGB

  def inputName(self, channel=0):
    return 'Input video stream, with biased colour.'

  def source(self, toChannel, video, videoChannel=0):
    self.video = video
    self.channel = videoChannel


  def dependencies(self):
    return [self.video]

  def nextFrame(self):
    # Fetch the frame...
    img = self.video.fetch(self.channel)
    if img==None:
      self.output = None
      return False

    # Handle the OpenCL object creation if needed...
    if self.cb_cl==None and self.cl!=None:
      try:
        self.cb_cl = colour_bias_cl.ColourBiasCL(self.cl, self.width(), self.height(), os.path.abspath(os.path.dirname(__file__)))

        self.cb_cl.chromaScale = self.chromaScale
        self.cb_cl.lumScale = self.lumScale
        self.cb_cl.noiseFloor = self.noiseFloor
        self.cb_cl.post_change()
      except:
        self.cl = None

    # Check we have a copy to mess with...
    if self.output==None:
      self.output = img.copy()
      if weave==None: self.inter = img.copy()

    # Run using OpenCl if possible...
    if False: #self.cb_cl!=None:
      self.cb_cl.to_rgb(img)
      self.cb_cl.fetch(self.output)
    else:
      # Convert back to rgb...
      if weave:
        code = start_cpp() + """
        for (int y=0;y<Noutput[0];y++)
        {
         for (int x=0;x<Noutput[1];x++)
         {
          // Merge luminance and chromacity, taking into account the noise floor and remove the luminance bias, which includes scaling to a volume of 1...
           float inter[3];
           inter[0] = IMG3(y,x,0) / lumScale;

           float cDiv = inter[0];
           if (cDiv<noiseFloor) cDiv = noiseFloor;
           cDiv /= chromaScale;

           inter[1] = IMG3(y,x,1) * cDiv;
           inter[2] = IMG3(y,x,2) * cDiv;

          // Rotate back to rgb (A matrix multiplication.)...
           for (int r=0;r<3;r++)
           {
            OUTPUT3(y,x,r) = 0.0;
            for (int c=0;c<3;c++)
            {
             OUTPUT3(y,x,r) += MATRIX2(r,c) * inter[c];
            }
           }
         }
        }
        """

        output = self.output
        matrix = self.invMatrix
        lumScale = self.lumScale
        chromaScale = self.chromaScale
        noiseFloor = self.noiseFloor

        weave.inline(code, ['img', 'output', 'matrix', 'lumScale', 'chromaScale', 'noiseFloor'])
      else:
        ## Apply the inverse of the bias scalling...
        self.inter[:,:,0]  = img[:,:,0]  / self.lumScale
        self.inter[:,:,1:] = img[:,:,1:] / self.chromaScale

        ## Multiply by luminance to get back the dependency...
        self.inter[:,:,1] *= numpy.clip(self.inter[:,:,0], self.noiseFloor, 1e100)
        self.inter[:,:,2] *= numpy.clip(self.inter[:,:,0], self.noiseFloor, 1e100)

        ## Apply the inverse matrix operation...
        for r in xrange(3):
          self.output[:,:,r] = (self.inter * self.invMatrix[r,:].reshape((1,1,-1))).sum(axis=2)

    return True


  def outputCount(self):
    return 1

  def outputMode(self, channel=0):
    return MODE_RGB

  def outputName(self, channel=0):
    return 'Video stream returned to rgb colour space.'

  def fetch(self, channel=0):
    return self.output
