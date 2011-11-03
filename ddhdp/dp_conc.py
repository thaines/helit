# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



class PriorConcDP:
  """Contains the parameters required for the concentration parameter of a DP - specifically its Gamma prior and initial concentration value."""
  def __init__(self, other = None):
    if other!=None:
      self.__alpha = other.alpha
      self.__beta = other.beta
      self.__conc = other.conc
    else:
      self.__alpha = 1.0
      self.__beta = 1.0
      self.__conc = 16.0

  def getAlpha(self):
    """Getter for alpha."""
    return self.__alpha

  def getBeta(self):
    """Getter for beta."""
    return self.__beta

  def getConc(self):
    """Getter for the initial concentration."""
    return self.__conc

  def setAlpha(self, alpha):
    """Setter for alpha."""
    assert(alpha>0.0)
    self.__alpha = alpha

  def setBeta(self, beta):
    """Setter for beta."""
    assert(beta>0.0)
    self.__beta = beta

  def setConc(self, conc):
    """Setter for the initial concentration."""
    assert(conc>=0.0)
    self.__conc = conc

  alpha = property(getAlpha, setAlpha, None, "The alpha parameter of the Gamma prior over the concentration parameter.")
  beta = property(getBeta, setBeta, None, "The beta parameter of the Gamma prior over the concentration parameter.")
  conc = property(getConc, setConc, None, "The starting value of the concentration parameter, to be updated.")
