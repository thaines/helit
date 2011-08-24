# Copyright 2011 Tom SF Haines

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



import math
import random



class ConcentrationDP:
  """Represents the concentration parameter of a Dirichlet process - contains the parameters of its prior and updates its estimate as things change. The estimate is actually the mean of many Gibbs samples of the parameter."""
  def __init__(self):
    """Initialises with both parameters of the prior set to 1 - i.e. both alpha and beta of the gamma distribution."""
    self.conc = 10.0
    
    self.alpha = 1.0
    self.beta = 1.0

    self.burnIn = 128
    self.samples = 128

  def setPrior(self, alpha, beta):
    """Sets the alpha and beta parameters of the concentrations gamma prior. They both default to 1."""
    self.alpha = alpha
    self.beta = beta

  def setParms(self, burnIn, samples):
    """Sets the Gibbs sampling parameters for updating the estimate. They both default to 128."""
    self.burnIn = burnIn
    self.samples = samples


  def __resample(self, conc, k, n):
    """Internal method - does a single resampling."""
    nn = random.betavariate(conc + 1.0, n)

    fAlpha = self.alpha + k
    fBeta = self.beta - math.log(nn)

    pi_n_mod = (fAlpha - 1.0) / (n * fBeta)
    r = random.random()
    r_mod = r / (1.0 - r)
    if r_mod>=pi_n_mod: fAlpha -= 1.0

    return random.gammavariate(fAlpha,fBeta)
    
  def update(self, k, n):
    """Given k, the number of dp instances, and n, the number of samples drawn from the dp, updates and returns the concentration parameter."""
    assert(n>=k)
    if k>1:
      conc = self.conc
      for _ in xrange(self.burnIn): conc = self.__resample(conc, k, n)

      self.conc = 0.0
      for count in xrange(self.samples):
        conc = self.__resample(conc, k, n)
        self.conc += (conc-self.conc) / float(count+1)

    return self.conc

  def getConcentration(self):
    """Returns the most recent estimate of the concentration."""
    return self.conc
