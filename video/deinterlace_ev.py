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



import numpy
import scipy.weave as weave

from utils.start_cpp import start_cpp

from video_node import *



class DeinterlaceEV(VideoNode):
  """Does exactly as specified - it deinterlaces. Not real time however, in fact its bloody slow - uses a system of making multiple estimates based on different assumptions and then taking a 'vote' as to which estimate has the most support. By default fast mode is on, which uses a multi-deminsional median rather than a fancy falloff function - this makes it about 3 times faster for only a slight loss in quality (Just about real time on low-resolution video, if your not doing anything else.)."""
  def __init__(self, oddFirst = True, fast = True):
    """Video is the video object, from which the frames are pulled, oddFirst dictates if the odd field or the even field comes first, double FPS indicates if it should double the frame rate or throw away the additional frames from the deinterlacing process."""
    self.video = None
    self.channel = 0
    self.oddFirst = oddFirst
    self.fast = fast

    self.frames = None # Becomes [previous,current] after first call to nextFrame.
    self.output = None # Output frame, which is updated each time nextframe is called..

    self.bucket = numpy.empty((6,4),dtype=numpy.float32)

  def width(self):
    return self.video.width()

  def height(self):
    return self.video.height()

  def fps(self):
    return self.video.fps()

  def frameCount(self):
    """Returns the number of frames for the video."""
    return self.video.frameCount()


  def inputCount(self):
    return 1

  def inputMode(self, channel=0):
    return MODE_RGB

  def inputName(self, channel=0):
    return 'Video to be deinterlaced'

  def source(self, toChannel, video, videoChannel=0):
    self.video = video
    self.channel = videoChannel


  def dependencies(self):
    return [self.video]

  def nextFrame(self):
    # If its first call do the initalisation...
    if self.frames==None:
      first = self.video.fetch(self.channel).copy()
      self.frames = [None,first]

      self.output = first.copy()

      self.nextOdd = self.oddFirst
      self.nextFirst = False

      return True # Can't deinterlace the first as no previous.

    # Get next frame...
    self.frames[0] = self.frames[1]
    self.frames[1] = self.video.fetch(self.channel)
    if self.frames[1]!=None: self.frames[1] = self.frames[1].copy()

    # Handle if out of content...
    if self.frames[0]==None:
      self.output = None
      return False

    # Handle if it is the last frame - can't deinterlace...
    if self.frames[1]==None:
      self.output[:] = self.frames[0]
      self.frames[0] = None
      return True


    # Deinterlace the current data and output it...
    code = start_cpp() + """
    const float falloff = 1.5;
    const float falloffLog = log(falloff);

    int keepMod = 0;
    if (PyObject_IsTrue(keepOdd)) keepMod = 1;

    for (int y=0;y<Nfirst[0];y++)
    {
     if ((y%2)==keepMod)
     {
      // Keep this field...
      for (int x=0;x<Nfirst[1];x++)
      {
       for (int c=0;c<3;c++) OUT3(y,x,c) = FIRST3(y,x,c);
      }
     }
     else
     {
      // Interpolate this field...
      for (int x=0;x<Nfirst[1];x++)
      {
       // Go through and collect various guesses as to what the correct colour is...
       int guessCount = 0;

       for (int c=0;c<3;c++) BUCKET2(guessCount,c) = FIRST3(y,x,c);
       guessCount += 1;

       for (int c=0;c<3;c++) BUCKET2(guessCount,c) = SECOND3(y,x,c);
       guessCount += 1;

       for (int c=0;c<3;c++) BUCKET2(guessCount,c) = 0.5*(float(FIRST3(y,x,c)) + float(SECOND3(y,x,c)));
       guessCount += 1;

       bool safeNY = y!=0;
       bool safePY = y+1<Nfirst[0];

       if (safeNY)
       {
        for (int c=0;c<3;c++) BUCKET2(guessCount,c) = FIRST3(y-1,x,c);
        guessCount += 1;
       }

       if (safePY)
       {
        for (int c=0;c<3;c++) BUCKET2(guessCount,c) = FIRST3(y+1,x,c);
        guessCount += 1;
       }

       if (safeNY&&safePY)
       {
        for (int c=0;c<3;c++) BUCKET2(guessCount,c) = 0.5*(float(FIRST3(y-1,x,c)) + float(FIRST3(y+1,x,c)));
        guessCount += 1;
       }



       // Weight all the guesses by how similar they are to the other guesses...
       for (int g=0;g<guessCount;g++) BUCKET2(g,3) = 0.0;

       for (int g1=0;g1<guessCount;g1++)
       {
        for (int g2=g1+1;g2<guessCount;g2++)
        {
         float dist = 0.0;
         for (int c=0;c<3;c++) dist += (BUCKET2(g1,c) - BUCKET2(g2,c)) * (BUCKET2(g1,c) - BUCKET2(g2,c));
         dist = sqrt(dist);

         float weight = (Py_True==fast)?dist:(falloffLog/log(falloff + dist));

         BUCKET2(g1,3) += weight;
         BUCKET2(g2,3) += weight;
        }
       }


       // Find the guess with the strongest support, assign it...
       int bestInd = 0;
       for (int g=1;g<guessCount;g++)
       {
        if (Py_True==fast)
        {
         if (BUCKET2(g,3)<BUCKET2(bestInd,3)) bestInd = g;
        }
        else
        {
         if (BUCKET2(g,3)>BUCKET2(bestInd,3)) bestInd = g;
        }
       }
       for (int c=0;c<3;c++) OUT3(y,x,c) = BUCKET2(bestInd,c);
      }
     }
    }
    """

    if self.nextFirst:
      first = self.frames[0]
      second = self.frames[1]
    else:
      first = self.frames[1]
      second = self.frames[0]
    out = self.output
    keepOdd = self.nextOdd
    bucket = self.bucket
    fast = self.fast
    weave.inline(code,['first','second','out','keepOdd','bucket','fast'])

    # Return the output frame...
    return True


  def outputCount(self):
    return 1

  def outputMode(self, channel=0):
    return MODE_RGB

  def outputName(self, channel=0):
    return 'Deinterlaced video frames'

  def fetch(self, channel=0):
    return self.output
