# Copyright (c) 2011, Tom SF Haines
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#  * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



import os.path
import pickle
import numpy



class Iris1D:
  """1D interface to the iris data set - uses pca to make it a 1D problem that is good for extremelly simple tests/demonstrations."""
  def __init__(self):
    # First load the data from the file...
    directory = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(directory, 'iris.data')
    
    lines = open(filename, 'r').readlines()
    lines = lines[:-1] # Last line is empty
    data_mat = numpy.empty((len(lines),4), dtype=numpy.float32)
    str_to_int = dict()
    self.ans = numpy.empty(len(lines), dtype=numpy.uint8)

    for i, line in enumerate(lines):
      parts = line.split(',')
      data_mat[i,:] = map(float, parts[:4])
      cat = parts[-1]
      if cat not in str_to_int: str_to_int[cat] = len(str_to_int)
      self.ans[i] = str_to_int[cat]

    # Normalise it...
    data_mat -= data_mat.mean(axis=0)
    data_mat /= numpy.abs(data_mat).mean(axis=0)

    # Calculate the pca matrix...
    u, _, _ = numpy.linalg.svd(data_mat, full_matrices=False)
    pca_mat = u[0,:]

    # Apply the 'matrix' to all the instances, to create the final data structure...
    self.vec = numpy.empty(len(lines), dtype=numpy.float32)

    for i in xrange(len(lines)):
      self.vec[i] = numpy.dot(pca_mat, data_mat[i,:])


  def getVectors(self):
    """Returns a numpy vector of float32 that contains the single parameter extracted from PCA for each entry."""
    return self.vec

  def getClasses(self):
    """Returns the class represented by each parameter - a numpy array of uint8, containing {0,1,2}."""
    return self.ans
