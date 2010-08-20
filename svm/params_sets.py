# -*- coding: utf-8 -*-

# Copyright 2010 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



from params import Kernel
from params import Params

from copy import copy



class ParamsRange:
  """A parameters object where each variable takes a list rather than a single value - it then pretends to be a list of Params objects, which consists of every combination implied by the ranges."""
  def __init__(self):
    """Initialises to contain just the default."""
    self.c = [10.0]
    self.rebalance = [True]

    # Which kernel to use and two parameters whose meanings depend on the kernel...
    self.kernel = [Kernel.linear]
    self.p1 = [1.0]
    self.p2 = [1.0]

  def getCList(self):
    """Returns the list of c parameters."""
    return self.c

  def setCList(self,c):
    """Sets the list of c values."""
    self.c = c

  def getRebalanceList(self):
    """Returns the list of rebalance options - can only ever be two"""
    return self.rebalance

  def setRebalanceList(self,rebalance):
    """Sets if c is rebalanced or not."""
    self.rebalance = rebalance


  def setKernelList(self,kernel):
    """Sets the list of kernels."""
    self.kernel = kernel

  def setP1List(self,p1):
    """Sets the list of P1 values."""
    self.p1 = p1

  def setP2List(self,p2):
    """Sets the list of P2 values."""
    self.p2 = p2

  def getKernelList(self):
    """Returns the list of kernels."""
    return self.kernel

  def getP1List(self):
    """returns the list of kernel parameters 1, not always used."""
    return self.p1

  def getP2List(self):
    """returns the list of kernel parameters 2, not always used."""
    return self.p2


  def permutations(self):
    p = Params()
    for c in self.c:
      p.setC(c)
      for rebalance in self.rebalance:
        p.setRebalance(c)
        for kernel in self.kernel:
          p.setKernel(kernel)
          for p1 in self.p1:
            p.setP1(p1)
            for p2 in self.p2:
              p.setP2(p2)
              yield copy(p)

  def __iter__(self):
    return self.permutations()



class ParamsSet:
  """Pretends to be a list of parameters, when instead it is a list of parameter ranges, where each set of ranges defines a search grid - used for model selection, typically by being passed as the params input to the MultiModel class."""
  def __init__(self, incDefault = False, incExtra = False):
    """Initialises the parameter set - with the default constructor this is empty. However, initalising it with paramsSet(True) gets you a good default set to model select with (That is the addLinear and addPoly methods are called with default parameters.), whilst paramsSet(True,True) gets you an insanely large default set for if your feeling particularly patient (It being all the add methods with default parameters.)."""
    self.ranges = []

    if incDefault:
      self.addLinear()
      self.addPoly()
    
    if incExtra:
      self.addHomoPoly()
      self.addBasisFuncs()
      self.addSigmoid()


  def addRange(self, ran):
    """Adds a new ParamsRange to the set."""
    self.ranges.append(ran)


  def addLinear(self, cExpLow = -3, cExpHigh = 3, cExp = 10.0, rebalance = True):
    """Adds a standard linear model to the set, with a range of c values. These values will range from cExp^cExpLow to cExp^cExpHigh, and by default are the set {0.001,0.01,0.1,1,10,100,1000}, which is typically good enough."""
    ran = ParamsRange()
    ran.setCList(map(lambda x:cExp**x,xrange(cExpLow,cExpHigh+1)))
    ran.setRebalanceList([rebalance])

    self.addRange(ran)

  def addHomoPoly(self, maxDegree = 6, cExpLow = -3, cExpHigh = 3, cExp = 10.0, rebalance = True):
    """Adds the homogenous polynomial to the set, from an exponent of 2 to the given value inclusive, which defaults to 8. Same c controls as for addLinear."""
    ran = ParamsRange()
    ran.setCList(map(lambda x:cExp**x,xrange(cExpLow,cExpHigh+1)))
    ran.setRebalanceList([rebalance])

    ran.setKernelList([Kernel.homo_polynomial])
    ran.setP1List(range(2,maxDegree+1))

    self.addRange(ran)

  def addPoly(self, maxDegree = 6, cExpLow = -3, cExpHigh = 3, cExp = 10.0, rebalance = True):
    """Adds the polynomial to the set, from an exponent of 2 to the given value inclusive, which defaults to 8. Same c controls as for addLinear."""
    ran = ParamsRange()
    ran.setCList(map(lambda x:cExp**x,xrange(cExpLow,cExpHigh+1)))
    ran.setRebalanceList([rebalance])

    ran.setKernelList([Kernel.polynomial])
    ran.setP1List(range(2,maxDegree+1))

    self.addRange(ran)

  def addBasisFuncs(self, rExpHigh = 6, rExp = 2.0, sdExpHigh = 6, sdExp = 2.0, cExpLow = -3, cExpHigh = 3, cExp = 10.0, rebalance = True):
    """Adds the basis functions to the set, both Radial and Gaussian. The parameter for the radial basis functions go from rExp^0 to rExp^rExpHigh, whilst the parameter for the Gaussian does the same thing, but with the sd parameters. Same c controls as for addLinear."""
    ran = ParamsRange()
    ran.setCList(map(lambda x:cExp**x,xrange(cExpLow,cExpHigh+1)))
    ran.setRebalanceList([rebalance])

    ran.setKernelList([Kernel.rbf])
    ran.setP1List(map(lambda x:rExp**x,xrange(rExpHigh+1)))

    self.addRange(ran)

    ran = ParamsRange()
    ran.setCList(map(lambda x:cExp**x,xrange(cExpLow,cExpHigh+1)))
    ran.setRebalanceList([rebalance])

    ran.setKernelList([Kernel.gbf])
    ran.setP1List(map(lambda x:sdExp**x,xrange(sdExpHigh+1)))

    self.addRange(ran)

  def addSigmoid(self, sExpLow = -3, sExpHigh = 3, sExp = 10.0, oExpLow = -3, oExpHigh = 3, oExp = 10.0, cExpLow = -3, cExpHigh = 3, cExp = 10.0, rebalance = True):
    """Add sigmoids to the set - the parameters use s for the scale component and o for the offset component; these parameters use the same exponential scheme as for c and others. Same c controls as for addLinear."""
    ran = ParamsRange()
    ran.setCList(map(lambda x:cExp**x,xrange(cExpLow,cExpHigh+1)))
    ran.setRebalanceList([rebalance])

    ran.setKernelList([Kernel.sigmoid])
    ran.setP1List(map(lambda x:oExp**x,xrange(oExpLow,oExpHigh+1)))
    ran.setP2List(map(lambda x:sExp**x,xrange(sExpLow,sExpHigh+1)))

    self.addRange(ran)


  def permutations(self):
    for ran in self.ranges:
      for p in ran:
        yield p

  def __iter__(self):
    return self.permutations()
