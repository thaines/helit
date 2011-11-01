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



import numpy
from scipy import weave

from utils.start_cpp import start_cpp

from flag_index_array import FlagIndexArray
from smp_cpp import smp_code



class SMP:
  """Impliments a Python wrapper around the C++ code for the Sparse Multinomial Posterior. Estimates the multinomial distribution from which various samples have been drawn, where those samples are sparse, i.e. not all counts are provided."""
  def __init__(self, fia):
    """Initialises with a FlagIndexArray object (Which must of had the addSingles method correctly called.) - this specifies the various combinations of counts being provided that are allowed."""
    self.flagMat = fia.getFlagMatrix()
    self.power = numpy.zeros(self.flagMat.shape[0], dtype=numpy.int32)

    self.sampleCount = 1024

    self.priorMN = numpy.ones(self.flagMat.shape[1], dtype=numpy.float32)
    self.priorMN /= self.flagMat.shape[1]
    self.priorConc = 0.0

  def setSampleCount(self, count):
    """Sets the number of samples to use when approximating the integral."""
    self.sampleCount = count

  def setPrior(self, conc, mn = None):
    """Sets the prior, as a Dirichlet distribution represented by a concentration and a multinomial distribution. Can leave out the multinomial to just update the concentration."""
    if mn!=None: self.priorMN[:] = mn
    self.priorConc = conc


  def reset(self):
    """Causes a reset, so you may add a new set of samples."""
    self.power[:] = 0

  def add(self, fi, counts):
    """Given the flag index returned from the relevant fia and an array of counts this adds it to the smp."""
    c = counts * self.flagMat[fi,:]
    self.power[:self.flagMat.shape[1]] += c
    self.power[fi] -= c.sum() + 1


  def mean(self):
    """Returns an estimate of the mean for each value of the multinomial, as an array, given the evidence provided. (Will itself sum to one - a necesary consequence of being an average of points constrained to the simplex."""
    code = start_cpp(smp_code) + """
    srand48(time(0));

    SMP smp(NflagMat[1],NflagMat[0]);
    smp.SetFIA(flagMat);
    smp.SetSampleCount(sampleCount);
    smp.SetPrior(priorMN, priorConc);
    smp.Add(power_array);

    smp.Mean(out);
    """

    flagMat = self.flagMat
    sampleCount = self.sampleCount
    priorMN = self.priorMN
    priorConc = self.priorConc
    power = self.power

    out = numpy.empty(flagMat.shape[1] ,dtype=numpy.float32)
    weave.inline(code, ['flagMat', 'sampleCount', 'priorMN', 'priorConc', 'power', 'out'], support_code=smp_code)
    return out
