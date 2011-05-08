# -*- coding: utf-8 -*-

# Copyright 2010 Tom SF Haines

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



class Params:
  """Parameters for running the fitter that are universal to all fitters."""
  def __init__(self):
    self.runs = 8
    self.samples = 10
    self.burnIn = 1000
    self.lag = 100

    self.iterT = 1
    self.iterR = 1


  def setRuns(self, runs):
    """Sets the number of runs, i.e. how many seperate chains are run."""
    self.runs = runs

  def setSamples(self, samples):
    """Number of samples to extract from each chain - total number of samples going into the final estimate will then be sampels*runs."""
    self.samples = samples

  def setBurnIn(self, burnIn):
    """Number of Gibbs iterations to do for burn in before sampling starts."""
    self.burnIn = burnIn

  def setLag(self, lag):
    """Number of Gibbs iterations to do between samples."""
    self.lag = lag

  def setIterT(self, iterT):
    """Number of iterations of updating t to do for each inner loop."""
    self.iterT = iterT
    
  def setIterR(self, iterR):
    """Number of iterations of updating r to do for each inner loop."""
    self.iterR = iterR


  def getRuns(self):
    """Returns the number of runs."""
    return self.runs

  def getSamples(self):
    """Returns the number of samples."""
    return self.samples

  def getBurnIn(self):
    """Returns the burn in length."""
    return self.burnIn

  def getLag(self):
    """Returns the lag length."""
    return self.lag

  def getIterT(self):
    """Return the number of t iterations."""
    return self.iterT

  def getIterR(self):
    """Return the number of r iterations."""
    return self.iterR


  def fromArgs(self, args, prefix = ''):
    """Extracts from an arg string, typically sys.argv[1:], the parameters, leaving them untouched if not given. Uses --runs, --samples, --burnIn, --lag, --iterT and --iterR. Can optionally provide a prefix which is inserted after the '--'"""

    def getParam(args, prefix, name, default):
      try:
        ind = args[:-1].index('--'+prefix+name)
        return int(args[ind+1])
      except:
        return default

    self.runs    = getParam(args,prefix,'runs'   ,self.runs)
    self.samples = getParam(args,prefix,'samples',self.samples)
    self.burnIn  = getParam(args,prefix,'burnIn' ,self.burnIn)
    self.lag     = getParam(args,prefix,'lag'    ,self.lag)
    self.iterT   = getParam(args,prefix,'iterT'  ,self.iterT)
    self.iterR   = getParam(args,prefix,'iterr'  ,self.iterR)
