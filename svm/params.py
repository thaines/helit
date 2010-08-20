# -*- coding: utf-8 -*-

# Copyright 2010 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



class Kernel:
  """Enum of supported kernel types, with some helpful static methods."""
  linear = 0          # dot(x1,x2)
  homo_polynomial = 1 # dot(x1,x2)^p1
  polynomial = 2      # (dot(x1,x2)+1)^p1
  rbf = 3             # exp(-p1||x1-x2||^2)
  gbf = 3             # exp(-||x1-x2||^2 / 2p1^2)
  sigmoid = 4         # tanh(p2*dot(x1,x2) + p1)

  def getList():
    """Returns a list of all the kernels."""
    return [Kernel.linear, Kernel.homo_polynomial, Kernel.polynomial, Kernel.rbf, Kernel.gbf, Kernel.sigmoid]
    
  def toName(kernel):
    """Returns the full name of the kernel."""
    data = {Kernel.linear:'Linear', Kernel.homo_polynomial:'Homogeneous Polynomial', Kernel.polynomial:'Polynomial', Kernel.rbf:'Radial Basis Function', Kernel.gbf:'Gaussian Basis Function', Kernel.sigmoid:'Sigmoid'}
    return data[kernel]

  def toShortName(kernel):
    """Returns the short name of the kernel."""
    data = {Kernel.linear:'lin', Kernel.homo_polynomial:'homo-poly', Kernel.polynomial:'poly', Kernel.rbf:'rbf', Kernel.gbf:'gbf', Kernel.sigmoid:'sig'}
    return data[kernel]

  def toEquation(kernel):
    """Return a textural representation of the equation implimented by the kernel."""
    data = {Kernel.linear:'dot(x1,x2)', Kernel.homo_polynomial:'dot(x1,x2)^p1', Kernel.polynomial:'(dot(x1,x2)+1)^p1', Kernel.rbf:'exp(-p1||x1-x2||^2)', Kernel.gbf:'exp(-||x1-x2||^2 / 2p1^2)', Kernel.sigmoid:'tanh(p2*dot(x1,x2) + p1)'}
    return data[kernel]

  def toCode(kernel,p1,p2):
    """Given the two parameters this returns the C code for a kernel that calculates the function given the two vectors."""
    # Head...
    ret = 'double kernel(int length, double * x1, double * x2)\n'
    ret += '{\n'

    # If kernel requires a dot product generate it...
    if kernel in [Kernel.linear, Kernel.homo_polynomial, Kernel.polynomial, Kernel.sigmoid]:
      ret += ' double dot = 0.0;\n'
      ret += ' for (int i=0;i<length;i++)\n'
      ret += ' {\n'
      ret += '  dot += x1[i]*x2[i];\n'
      ret += ' }\n\n'

    # If kernel requires distance generate it, as distance squared...
    if kernel in [Kernel.rbf, Kernel.gbf]:
      ret += ' double dist2 = 0.0;\n'
      ret += ' for (int i=0;i<length;i++)\n'
      ret += ' {\n'
      ret += '  double diff = x1[i] - x2[i];\n'
      ret += '  dist2 += diff*diff;\n'
      ret += ' }\n\n'

    # Add in the return statement, which is unique to each kernel. Also remove the polynomial 'pow' commands if its to the power of 2...
    data = {Kernel.linear:'dot', Kernel.homo_polynomial:'pow(dot,{p1})', Kernel.polynomial:'pow(dot+1.0,{p1})', Kernel.rbf:'exp(-{p1}*dist2)', Kernel.gbf:'exp(-dist2 / (2.0*{p1}*{p1}))', Kernel.sigmoid:'tanh({p2}*dot + {p1})'}

    if (abs(p1-2.0)<1e-6) and (kernel in [Kernel.homo_polynomial, Kernel.polynomial]):
      if kernel==Kernel.homo_polynomial: exp = 'dot*dot'
      else: exp = '(dot+1.0)*(dot+1.0)'
    else:
      exp = data[kernel]
      exp = exp.replace('{p1}',str(p1))
      exp = exp.replace('{p2}',str(p2))

    ret += ' return ' + exp + ';\n'

    # Tail and return...
    ret += '}\n'
    return ret

  getList = staticmethod(getList)
  toName = staticmethod(toName)
  toShortName = staticmethod(toShortName)
  toEquation = staticmethod(toEquation)
  toCode = staticmethod(toCode)



class Params:
  """Parameters for the svm algorithm - softness and kernel. Defaults to a C value of 10 and a linear kernel."""
  def __init__(self):
    # The 'softness' parameter, and if it is rebalanced in the case of an unbalanced dataset...
    self.c = 10.0
    self.rebalance = True

    # Which kernel to use and two parameters whose meanings depend on the kernel...
    self.kernel = Kernel.linear
    self.p1 = 1.0
    self.p2 = 1.0

  def __str__(self):
    return '<C=' + str(self.c) + '(' + str(self.rebalance) + '); ' + Kernel.toShortName(self.kernel) + '(' + str(self.p1) + ',' + str(self.p2) + ')>'


  def getC(self):
    """Returns c, the softness parameter."""
    return self.c

  def setC(self,c):
    """Sets the c value, whcih indicates how soft the answer can be. (0 don't care, infinity means perfect seperation.) Default is 10.0"""
    self.c = c

  def getRebalance(self):
    """Returns whether the c value is rebalanced or not - defaults to true."""
    return self.rebalance

  def setRebalance(self,rebalance):
    """Sets if c is rebalanced or not."""
    self.rebalance = rebalance


  def getKernel(self):
    """Returns which kernel is being used; see the Kernel enum for transilations of the value."""
    return self.kernel

  def getP1(self):
    """returns kernel parameter 1, not always used."""
    return self.p1

  def getP2(self):
    """returns kernel parameter 2, not always used."""
    return self.p2

  def setKernel(self,kernel, p1 = None, p2 = None):
    """Sets the kernel to use, and the parameters if need be."""
    self.kernel = kernel
    if p1!=None: self.p1 = p1
    if p2!=None: self.p2 = p2

  def setP1(self, p1):
    """Sets parameter p1."""
    self.p1 = p1

  def setP2(self, p2):
    """Sets parameter p2."""
    self.p2 = p2
    


  def setLinear(self):
    """Sets it to use the linear kernel."""
    self.kernel = Kernel.Linear

  def setHomoPoly(self,degree):
    """Sets it to use a homogenous polynomial, with the given degree."""
    self.kernel = Kernel.homo_polynomial
    self.p1 = degree

  def setPoly(self,degree):
    """Sets it to use a polynomial, with the given degree."""
    self.kernel = Kernel.polynomial
    self.p1 = degree

  def setRBF(self,scale):
    """Sets it to use a radial basis function, with the given distance scale."""
    self.kernel = Kernel.rbf
    self.p1 = scale

  def setGBF(self,sd):
    """Sets it to use a gaussian basis function, with the given standard deviation. (This is equivalent to a RBF with the scale set to 1/(2*sd^2))"""
    self.kernel = Kernel.gbf
    self.p1 = sd

  def setSigmoid(self,scale,offset):
    """Sets it to be a sigmoid, with the given parameters."""
    self.kernel = Kernel.sigmoid
    self.p1 = offset
    self.p2 = scale


  def getCode(self):
    """Returns the code for a function that impliments the specified kernel, with the parameters hard coded in."""
    return Kernel.toCode(self.kernel,self.p1,self.p2)

  def kernelKey(self):
    """Returns a string unique to the kernel/kernel parameters combo."""
    ret = Kernel.toShortName(self.kernel)
    if self.kernel!=Kernel.linear:
      ret += ':' + str(self.p1)
    if self.kernel==Kernel.sigmoid:
      ret += ':' + str(self.p2)
    return ret
