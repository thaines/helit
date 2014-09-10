#! /usr/bin/env python

# Copyright 2014 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import os.path

import cv
import numpy
from scipy.weave import inline

import h5py

from utils.cvarray import *
from utils.start_cpp import start_cpp



class HMS:
  """Utility library for loading the files written out by the seg_hierarchy.py script and using them for basic inferences tasks."""
  def __init__(self, fn):
    """Provide the filename of the image, as it needs that to make sense of the hms file, which it will also load."""
    # Load the image...
    image = cv.LoadImage(fn)
    image = cv2array(image)
    self.shape = image.shape
    
    # Load the hms file and extract the basic data structure...
    hms_fn = os.path.splitext(fn)[0] + '.hms'
    f = h5py.File(hms_fn, 'r')
    
    self.levels = []
    
    while True:
      # Calculate the layer number and verify it exists...
      l = len(self.levels)
      if ('%i.sizes'%l) not in f:
        break
      
      # Load the parts...
      clusters = numpy.array(f['%i.clusters'%l]) if ('%i.clusters'%l) in f else None
      parents = numpy.array(f['%i.parents'%l]) if ('%i.parents'%l) in f else None
      sizes = numpy.array(f['%i.sizes'%l])
      
      # Record...
      self.levels.append([clusters, parents, sizes])
    
    # Finish the bottom layer of the data structure using the image...
    gridX, gridY = numpy.meshgrid(numpy.arange(image.shape[1]), numpy.arange(image.shape[0]))
    clusters = numpy.concatenate((gridY[:,:,numpy.newaxis], gridX[:,:,numpy.newaxis], image), axis=2).reshape((-1, 5))
    self.levels[0][0] = clusters
    
    # If the top level does not go to a single mode create a new one that does, so everything is connected...
    if self.levels[-1][0].shape[0]>1:
      self.levels[-1][1] = numpy.zeros(self.levels[-1][0].shape[0], dtype=numpy.int32)
      
      clusters = numpy.average(self.levels[-1][0], axis=0, weights=self.levels[-1][2])[numpy.newaxis,:]
      parents = None
      sizes = numpy.array([[self.levels[-1][2].sum()]], dtype=self.levels[-1][2].dtype)
      
      self.levels.append([clusters, parents, sizes])
    
    # Clean up...
    f.close()


  def __ensure_tree(self):
    """Verifies that the data structure contains the tree version, which is setup for a certain kind of inference, and if not creates it."""
    if not hasattr(self, 'parent'):
      
      # Calculate required size...
      total = 0
      for lev in self.levels:
        total += lev[0].shape[0]
      
      # Create data structure...
      self.parent = numpy.empty(total, dtype=numpy.int32)
      self.parent[:] = -1
      
      self.s_dist = numpy.zeros(total, dtype=numpy.float32) # Spatial distance
      self.c_dist = numpy.zeros(total, dtype=numpy.float32) # Colourmetric distance
      
      # Loop and fill the structure in...
      base = 0
      for l, lev in enumerate(self.levels[:-1]):
        # Calculate the base for the next level, to offset parents by...
        next_base = base + lev[0].shape[0]
        
        # Handle the parent locations...
        self.parent[base:next_base] = lev[1] + next_base
          
        # Handle the squared differences...
        delta_sqr = numpy.square(lev[0] - self.levels[l+1][0][lev[1],:])
        
        self.s_dist[base:next_base] = delta_sqr[:,:2].sum(axis=1)
        self.s_dist[base:next_base] = delta_sqr[:,2:].sum(axis=1)
        
        # Set the base ready for the next level...
        base = next_base
        
      # Create a basic cost function - just the two squares added and sqrt, so even balance...
      self.distance = numpy.sqrt(self.s_dist + self.c_dist)
      
      # Cache used on occasion - made here as it aligns with the distance array...
      self.temp_label = numpy.empty(total, dtype=numpy.int32)
      self.temp_travel = numpy.empty(total, dtype=numpy.float32)
  
  
  def set_ratio(self, s, c):
    """Sets the weighting of spatial vs colourmetric for the distances - by default its (1,1)."""
    self.__ensure_tree()
    self.distance = numpy.sqrt(self.s_dist * s + self.c_dist * c)
  
  
  def dist(self, a, b):
    """Returns the distance between a and b given the spatial pyramid. Expe4cts coordinates [y,x], but vectorised so you can provide entire arrays, which is recomended for speed."""
    
    # Need a tree...
    self.__ensure_tree()
    
    # Make shape consistant...
    if len(a.shape)==1:
      a = a[numpy.newaxis,:]
    if len(b.shape)==1:
      b = b[numpy.newaxis,:]
    
    # Convert to offsets into the tree arrays...
    a = a[:,0] * self.shape[1] + a[:,1]
    b = b[:,0] * self.shape[1] + b[:,1]
    
    # Create the return value...
    ret = numpy.zeros(a.shape[0], dtype=numpy.float32)
    
    # Keep updating the two parent arrays until they point to the same value, selecting the lowest each time, until neither need changing because they have converged to the same value...
    while True:
      stop = True
      
      # Update a...
      ind = numpy.where(a<b)[0]
      if ind.shape[0]>0:
        stop = False
        ret[ind] += self.distance[a[ind]]
        a[ind] = self.parent[a[ind]]
      
      # Update b...
      ind = numpy.where(a>b)[0]
      if ind.shape[0]>0:
        stop = False
        ret[ind] += self.distance[b[ind]]
        b[ind] = self.parent[b[ind]]
      
      # Break if done...
      if stop:
        break
    
    # Return the distance...
    return ret
  
  
  def edge(self, offset):
    """Given an offset (typically [0,1] or [1,0], note its [y,x]) this returns an array that gives the difference between a pixel and the pixel at the given offset from the pixel. The returned array is reduced in size by the absolute values of the offset in each direction, so it fits. The coordinates map in the obvious way, given what was just stated... which isn't that obvious really:-P"""
    
    # Calculate the coordinates...
    out_shape = (self.shape[0] - abs(offset[0]), self.shape[1] - abs(offset[1]))
    
    gridX, gridY = numpy.meshgrid(numpy.arange(out_shape[1]), numpy.arange(out_shape[0]))
    
    a = numpy.concatenate((gridY[:,:,numpy.newaxis], gridX[:,:,numpy.newaxis]), axis=2).reshape((-1, 2))
    b = a.copy()
    
    if offset[0]<0:
      a[:,0] -= offset[0]
    if offset[0]>0:
      b[:,0] += offset[0]
      
    if offset[1]<0:
      a[:,1] -= offset[1]
    if offset[1]>0:
      b[:,1] += offset[1]
    
    # Do the distance calculation...
    ret = self.dist(a, b)
    
    # Return with the correct shape...
    return ret.reshape(out_shape)
  
  
  def closest(self, label, out):
    """Given a labeling of an image (same size, integers) in the label parameter this copies the labling into the provided out variable, replacing all negative values with the closest non-negative value in label. Closest is as provided by the dist method; due to the tree structure this can be calculated at real time speeds. (Dependent on scipy.weave as it uses inline C)"""
    
    # Need a tree...
    self.__ensure_tree()
    
    # Code that does the actual calculation...
    code = start_cpp() + """
    // Reset the temp_label array to be negative...
     for (int i=0; i<Ntemp_label[0]; i++)
     {
      TEMP_LABEL1(i) = -1;
     }
     
    // Fill in TEMP_* with the starting conditions from label...
     for (int y=0; y<Nlabel[0]; y++)
     {
      for (int x=0; x<Nlabel[1]; x++)
      {
       int l = LABEL2(y, x);
       if (l>=0)
       {
        int i = Nlabel[1] * y + x;
        TEMP_LABEL1(i) = l;
        TEMP_TRAVEL1(i) = 0.0;
       }
      }
     }
    
    // Upwards pass, to transmit closest labels to the root in temp_label...
     for (int i=0; i<Nparent[0]-1; i++)
     {
      if (TEMP_LABEL1(i)>=0)
      {
       int p = PARENT1(i);
       float d = TEMP_TRAVEL1(i) + DISTANCE1(i);
       
       if ((TEMP_LABEL1(p)<0)||(TEMP_TRAVEL1(p)>d))
       {
        TEMP_LABEL1(p)  = TEMP_LABEL1(i);
        TEMP_TRAVEL1(p) = d;
       }
      }
     }
    
    // Downwards pass, to complete labels...
     for (int i=Nparent[0]-2; i>=0; i--)
     {
      int p = PARENT1(i);
      float d = TEMP_TRAVEL1(i) + DISTANCE1(p);
      
      if ((TEMP_LABEL1(i)<0)||(TEMP_TRAVEL1(i)>d))
      {
       TEMP_LABEL1(i)  = TEMP_LABEL1(p);
       TEMP_TRAVEL1(i) = d;
      }
     }
    
    // Transfer labels to out...
     for (int y=0; y<Nout[0]; y++)
     {
      for (int x=0; x<Nout[1]; x++)
      {
       OUT2(y, x) = TEMP_LABEL1(Nout[1] * y + x);
      }
     }
    """
    
    # Make variables local as required...
    parent = self.parent
    distance = self.distance
    temp_label = self.temp_label
    temp_travel = self.temp_travel
    
    # Run...
    inline(code, ['out','label','parent','distance','temp_label','temp_travel'])



if __name__=='__main__':
  import sys
  if len(sys.argv)<2:
    print 'Expects a filename of an image file for which seg_hierarchy.py has been run'
    sys.exit(1)
    
  f = HMS(sys.argv[1])
  
  edge_img = numpy.zeros((f.shape[0], f.shape[1], 3), dtype=numpy.float32)
  edge_img[:-1,:,0] = f.edge([1,0])
  edge_img[:,:-1,2] = f.edge([0,1])
  
  edge_img *= 255.0 / edge_img.max()
  edge_img = array2cv(edge_img)
  cv.SaveImage(os.path.splitext(sys.argv[1])[0] + '_edges.png', edge_img)
