# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



class Params:
  """Parameters for running the fitter that are universal to all fitters - basically the parameters you would typically associate with Gibbs sampling."""
  def __init__(self, toClone = None):
    """Sets the parameters to reasonable defaults. Will act as a copy constructor if given an instance of this object."""
    if toClone!=None:
      self.__runs = toClone.runs
      self.__samples = toClone.samples
      self.__burnIn = toClone.burnIn
      self.__lag = toClone.lag
    else:
      self.__runs = 8
      self.__samples = 10
      self.__burnIn = 1000
      self.__lag = 100


  def setRuns(self, runs):
    """Sets the number of runs, i.e. how many seperate chains are run."""
    assert(isinstance(runs, int))
    assert(runs>0)
    self.__runs = runs

  def setSamples(self, samples):
    """Number of samples to extract from each chain - total number of samples extracted will hence be samples*runs."""
    assert(isinstance(samples, int))
    assert(samples>0)
    self.__samples = samples

  def setBurnIn(self, burnIn):
    """Number of Gibbs iterations to do for burn in before sampling starts."""
    assert(isinstance(burnIn, int))
    assert(burnIn>=0)
    self.__burnIn = burnIn

  def setLag(self, lag):
    """Number of Gibbs iterations to do between samples."""
    assert(isinstance(lag, int))
    assert(lag>0)
    self.__lag = lag


  def getRuns(self):
    """Returns the number of runs."""
    return self.__runs

  def getSamples(self):
    """Returns the number of samples."""
    return self.__samples

  def getBurnIn(self):
    """Returns the burn in length."""
    return self.__burnIn

  def getLag(self):
    """Returns the lag length."""
    return self.__lag


  runs = property(getRuns, setRuns, None, "Number of seperate chains to run.")
  samples = property(getSamples, setSamples, None, "Number of samples to extract from each chain")
  burnIn = property(getBurnIn, setBurnIn, None, "Number of iterations to do before taking the first sample of a chain.")
  lag = property(getLag, setLag, None, "Number of iterations to do between samples.")


  def fromArgs(self, args, prefix = ''):
    """Extracts from an arg string, typically sys.argv[1:], the parameters, leaving them untouched if not given. Uses --runs, --samples, --burnIn and --lag. Can optionally provide a prefix which is inserted after the '--'"""
    try:
      ind = args[:-1].index('--'+prefix+'runs')
      self.runs = int(args[ind+1])
    except ValueError:
      pass

    try:
      ind = args[:-1].index('--'+prefix+'samples')
      self.samples = int(args[ind+1])
    except ValueError:
      pass

    try:
      ind = args[:-1].index('--'+prefix+'burnIn')
      self.burnIn = int(args[ind+1])
    except ValueError:
      pass

    try:
      ind = args[:-1].index('--'+prefix+'lag')
      self.lag = int(args[ind+1])
    except ValueError:
      pass
