# Copyright 2013 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import numpy
import scipy.misc
import scipy.special
import numpy.linalg as la



class TPS:
  """A n-dimensional thin plate spline implimentation. Nothing fancy - just a basic implimentation, so don't go throwing large data sets at it. Implimentation based on a snipet of a pdf found on the internet written by David Eberly. Includes smoothing, just by the addition of an identity matrix to the Green kernal matrix."""
  def __init__(self, n, smooth = 0.0):
    """Initialises the model for n dimensions. Can also include a smoothing parameter, which allows it to select a smoother model by not interpolating the given points perfectly."""
    
    # For evaluating the green function...
    self.green_pow = 4 - n
    if n==2 or n==4:
      green_mult = numpy.power(-1.0, 0.5*n + 1.0) / (8.0*numpy.sqrt(numpy.pi)*scipy.misc.factorial(2 - n//2))
      self.green_inc_ln = True
    else:
      green_mult = scipy.special.gamma(0.5*n - 2.0) / (16.0*numpy.power(numpy.pi, 0.5*n))
      self.green_inc_ln = False
    
    # Record the smoothing, with the green multiplication factored in...
    self.smooth = smooth / green_mult
    
    # Set the needed variables to None, ready for initialisation...
    self.n = n
    self.x = None
    self.a = None
    self.b = None
  
  
  def learn(self, x, y, a = None, b = None):
    """Allows you to learn a model, partially if you so choose. x is the data matrix of points to fit the spline through - a numpy array of shape [d,n], where d is the number of points (d>=n) and n the number of dimensions. y is the answer for each point, a vector aligned with x. If a and b are provided it can be set to None instead. b is the parameters of the plane, a (n+1) vector with the first n entries aligned with the dimensions and the final one the constant to offset it. a is the kernel weights, a size d matrix that indicates the weight associated with each kernel evalatuon of a point with the green function kernel."""
    # Safety checks...
    assert(len(x.shape)==2)
    assert(x.shape[0]>=self.n)
    assert(x.shape[1]==self.n)
    
    assert(y!=None or (a!=None and b!=None))
    assert(y==None or (len(y.shape)==1 and y.shape[0]==x.shape[0]))
    
    assert(a==None or (len(a.shape)==1 and a.shape[0]==x.shape[0]))
    assert(b==None or (len(b.shape)==1 and b.shape[0]==(self.n+1)))
    
    # Copy over the provided stuff...
    self.x = x.copy()
    if a!=None: self.a = a.copy()
    if b!=None: self.b = b.copy()
    
    # Generate the matrices...
    n = numpy.empty((x.shape[0], x.shape[1]+1), dtype=x.dtype)
    n[:,:-1] = x
    n[:,-1] = 1.0
      
    dist = numpy.zeros((x.shape[0], x.shape[0]), dtype=numpy.float32)
    for i in xrange(self.n):
      dist += numpy.square(x[:,i].reshape((x.shape[0],1)) - x[:,i].reshape((1,x.shape[0])))
    dist = numpy.sqrt(dist)
      
    m = numpy.power(dist, self.green_pow)
    if self.green_inc_ln:
      m *= numpy.log(dist)
    m += self.smooth * numpy.eye(m.shape[0])
      
    self.mult = 1.0 / m.max()
    m *= self.mult
    
    # If needed learn b...
    if b==None:
      ntmin = numpy.dot(n.T, la.pinv(m))
      bib = numpy.dot(ntmin, n)
      ymod = numpy.dot(ntmin, y)
      self.b = la.lstsq(bib, ymod)[0]
    
    # If needed learn a...
    if a==None:
      self.a = la.lstsq(m, y - numpy.dot(n, self.b))[0]
  
  
  def get_n(self):
    """Returns the number of dimensions of the thin plate spline."""
    return self.n
    
  def get_x(self):
    """Returns the set of points that locate the basis functions."""
    return self.x
    
  def get_a(self):
    """Returns a, the vector of kernel weights for each point in the spline."""
    return self.a
  
  def get_b(self):
    """Returns the vector b, the parameters of the base plane. The last entry is the constant."""
    return self.b
  
  
  def __call__(self, data):
    """Evaluates the spline at one or more points - if data is a vector at that point alone, and it returns a single number; if it is a data matrix at every point in the data matrix, and it returns a vector."""
    # Conversion
    assert(len(data.shape)<=2)
    assert(data.shape[-1]==self.n)
    
    if len(data.shape)==1:
      # Single vector has bene passed in...
      dist = numpy.sqrt(numpy.square(self.x - data.reshape((1,-1))).sum(axis=1))
      
      green = numpy.power(dist, self.green_pow)
      if self.green_inc_ln: # Implies dimensions = 2 or 4.
        green *= numpy.log(dist)
      green *= self.mult
      
      ret = (self.a * green).sum()
      ret += (data * self.b[:-1]).sum()
      ret += self.b[-1]
      
      return ret
    else:
      # Data matrix has been passed in...
      dist = numpy.zeros((data.shape[0], self.x.shape[0]), dtype=numpy.float32)
      for i in xrange(self.n):
        dist += numpy.square(data[:,i].reshape((-1,1)) - self.x[:,i].reshape((1,-1)))
      dist = numpy.sqrt(dist)
      
      green = numpy.power(dist, self.green_pow)
      if self.green_inc_ln:
        green *= numpy.log(dist)
      green *= self.mult
      
      ret = (self.a.reshape((1,-1)) * green).sum(axis=1)
      ret += (data * self.b[:-1].reshape((1,-1))).sum(axis=1)
      ret += self.b[-1]
      
      return ret
