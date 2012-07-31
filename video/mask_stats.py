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

from video_node import *



class MaskStats(VideoNode):
  """Calculates various statistics based on having two input masks, one being an estimate, the other ground truth. Can also take a validity mask, that indicates the areas where scores are to be calculated. Statistics are stored per frame, and can be queried at any time whilst running or after running. They are basically a per-frame confusion matrix, but interfaces are provided to get the recall, the precision and the f-measure, as defined by the paper 'Evaluation of Background Subtraction Techneques for Video Surveillance' by S. Brutzer, B. Hoferlin and G. Heidemann, to give one example. Averages can also be obtained over ranges of frames. Be warned that the frame indices are zero based."""
  def __init__(self):
    self.guess = None
    self.guessChannel = None

    self.truth = None
    self.truthChannel = None

    self.valid = None
    self.validChannel = None

    self.confusion = [] # Each 2x2 array is indexed [truth,guess], where 0=False and 1=True.


  def width(self):
    return self.truth.width()

  def height(self):
    return self.truth.height()

  def fps(self):
    return self.truth.fps()

  def frameCount(self):
    return self.truth.frameCount()


  def inputCount(self):
    return 3

  def inputMode(self, channel=0):
    return MODE_MASK

  def inputName(self, channel=0):
    if channel==0: return 'Estimated mask.'
    elif channel==1: return 'Ground truth mask.'
    else: return 'Optional scoring mask, indicating the areas to factor in.'

  def source(self, toChannel, video, videoChannel=0):
    if toChannel==0:
      self.guess = video
      self.guessChannel = videoChannel
    elif toChannel==1:
      self.truth = video
      self.truthChannel = videoChannel
    else:
      self.valid = video
      self.validChannel = videoChannel


  def dependencies(self):
    if self.valid==None: return [self.guess, self.truth]
    else: return [self.guess, self.truth, self.valid]

  def nextFrame(self):
    # Fetch the frames...
    guess = self.guess.fetch(self.guessChannel)
    truth = self.truth.fetch(self.truthChannel)
    if self.valid!=None: valid = self.valid.fetch(self.validChannel)

    # Calculate the scores...
    confusion = numpy.zeros((2,2), dtype=numpy.int32)

    s = numpy.logical_and(truth==0, guess==0)
    if self.valid!=None: s = numpy.logical_and(s, valid!=0)
    confusion[0,0] = s.sum()

    s = numpy.logical_and(truth==0, guess!=0)
    if self.valid!=None: s = numpy.logical_and(s, valid!=0)
    confusion[0,1] = s.sum()

    s = numpy.logical_and(truth!=0, guess==0)
    if self.valid!=None: s = numpy.logical_and(s, valid!=0)
    confusion[1,0] = s.sum()

    s = numpy.logical_and(truth!=0, guess!=0)
    if self.valid!=None: s = numpy.logical_and(s, valid!=0)
    confusion[1,1] = s.sum()

    # Store the confusion...
    self.confusion.append(confusion)


  def outputCount(self):
    return 0


  def framesAvaliable(self):
    """Returns the number of frames avaliable."""
    return len(self.confusion)

  def getConfusion(self, frame):
    """Returns the confusion matrix of a specific frame."""
    return self.confusion[frame]

  def getConfusionTotal(self, start, end):
    """Given an inclusive frame range returns the sum of the confusion matrices over that range."""
    ret = numpy.zeros((2,2), dtype=numpy.int64)
    for i in xrange(start,end+1): ret += self.confusion[i]
    return ret


  def getRecall(self, frame):
    """Given a frame returns the recall for that frame."""
    con = self.confusion[frame]
    if (con[1,0] + con[1,1])==0: return 1.0
    return float(con[1,1]) / float(con[1,0] + con[1,1])

  def getPrecision(self, frame):
    """Given a frame number returns that framess precision."""
    con = self.confusion[frame]
    if (con[0,1] + con[1,1])==0: return 1.0
    return float(con[1,1]) / float(con[0,1] + con[1,1])

  def getFMeasure(self, frame):
    """Returns the f-measure, which is the harmonic mean of the recall and precision, i.e. 2*recall*precision / (recall+precision)."""
    con = self.confusion[frame]

    if (con[1,0] + con[1,1])==0: recall = 1.0
    else: recall = float(con[1,1]) / float(con[1,0] + con[1,1])

    if (con[0,1] + con[1,1])==0: prec = 1.0
    else: prec = float(con[1,1]) / float(con[0,1] + con[1,1])

    return (2.0 * recall * prec) / (recall + prec)

  def getFMeasureAvg(self, start, end):
    """Given an inclusive frame range returns the average of the f-measure for that range."""
    ret = 0.0
    for i in xrange(start,end+1):
      val = self.getFMeasure(i)
      ret += (val - ret) / float(i+1-start)
    return ret

  def getFMeasureTotal(self, start, end):
    """Returns the f-measure by summing the confusion matrix over the entire range and then calculating."""
    con = self.getConfusionTotal(start, end)

    recall = float(con[1,1]) / float(con[1,0] + con[1,1])
    prec = float(con[1,1]) / float(con[0,1] + con[1,1])

    return (2.0 * recall * prec) / (recall + prec)
