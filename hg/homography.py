# Copyright 2016 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import numpy
import numpy.linalg



# Note that homographies are to be applied to vectors [x, y, w], to be consistant with everyone else, but then the arrays are indexed [y, x] - makes things a touch confusing at points.



def translate(offset):
  """Returns the homography for offsetting an input by a given amount - offset must be interpretable as a 2-element vector. Return matrix multiplied by a vector (x,y,1) will return a new vector (x+offset[0], y+offset[1], 1)."""
  hg = numpy.eye(3, dtype=numpy.float32)
  hg[0,2] = offset[0]
  hg[1,2] = offset[1]
  return hg



def rotate(angle):
  """Applies the given rotation, anticlockwise around the origin, in radians."""
  hg = numpy.eye(3, dtype=numpy.float32)
  hg[0,0] = numpy.cos(angle)
  hg[0,1] = -numpy.sin(angle)
  hg[1,0] = -hg[0,1]
  hg[1,1] = hg[0,0]
  return hg



def scale(amount):
  """Scales everythinng to be the given amount times bigger (or smaller if <1)."""
  hg = numpy.eye(3, dtype=numpy.float32)
  hg[0,0] = amount
  hg[1,1] = amount
  return hg



def match(source, dest):
  """Calculates a 2D homography that converts from the source coordinates to the dest coordinates. both are (4,2) data matrices of 4 x,y coordinates. Returns a 3x3 matrix that when multiplied by the homogenous versions of source gets you to dest."""
  bm = numpy.zeros((8,9), dtype=numpy.float32)
  for i in xrange(4):
    bm[i*2,3] = -source[i,0]
    bm[i*2,4] = -source[i,1]
    bm[i*2,5] = -1.0
    bm[i*2,6] = dest[i,1] * source[i,0]
    bm[i*2,7] = dest[i,1] * source[i,1]
    bm[i*2,8] = dest[i,1]
        
    bm[i*2+1,0] = source[i,0]
    bm[i*2+1,1] = source[i,1]
    bm[i*2+1,2] = 1.0
    bm[i*2+1,6] = -dest[i,0]*source[i,0]
    bm[i*2+1,7] = -dest[i,0]*source[i,1]
    bm[i*2+1,8] = -dest[i,0]
        
  hg = numpy.linalg.svd(bm)[2][-1,:].reshape((3,3))
  return hg



def bounds(hg, lower, upper):
  """Given a homography and a rectangle, as a lower/upper pair of coordinate (x, y order), this returns the axis-aligned rectangle (as the tuple (lower, upper)) that contains the rectangle after the provided homography has been applied."""
  
  # Corner points...
  c0 = numpy.array((lower[0], lower[1], 1.0), dtype=numpy.float32)
  c1 = numpy.array((lower[0], upper[1], 1.0), dtype=numpy.float32)
  c2 = numpy.array((upper[0], lower[1], 1.0), dtype=numpy.float32)
  c3 = numpy.array((upper[0], upper[1], 1.0), dtype=numpy.float32)
  
  # Transform...
  c0 = hg.dot(c0)
  c1 = hg.dot(c1)
  c2 = hg.dot(c2)
  c3 = hg.dot(c3)
  
  c0[:2] /= c0[2]
  c1[:2] /= c1[2]
  c2[:2] /= c2[2]
  c3[:2] /= c3[2]
  
  # Find range and return...
  low = numpy.array((min(c0[0],c1[0],c2[0],c3[0]), min(c0[1],c1[1],c2[1],c3[1])), dtype=numpy.float32)
  high = numpy.array((max(c0[0],c1[0],c2[0],c3[0]), max(c0[1],c1[1],c2[1],c3[1])), dtype=numpy.float32)
  
  return low, high



def fit(hg, shape):
  """Given a homography and the shape of the image (in height, width order - consistant with image arrays, not the homography!) it is to be applied to this returns the tuple (hg, shape), which is a replacement for the homography and the output shape, such that the image after transformation is not clipped. Effectively offsets the homography so there are no negative values and increases the shape as required."""
  lower, upper = bounds(hg, (0,0), (shape[1], shape[0]))
  
  new_hg = translate(-lower).dot(hg)
  new_shape = (int(numpy.ceil(upper[1] - lower[1])), int(numpy.ceil(upper[0] - lower[0])))
  
  return (new_hg, new_shape)



def scaling(hg, lower, upper, divisions = 100):
  """Given a rectangle this returns the tuple (minimum, maximum) giving the range of scaling encountered by the homography. Works by effectivly dividing the given rectangle (lower and upper) into the given number of divisions (defaults to 100) and finding their lengths before and after the transform, so the scaling factor for each can be found and the minimum/maximum determined. It assumes the homography doesn't have singularities, and hence only evaluates the 8 corner edges for efficiency."""
  
  # Corner points...
  corner = [numpy.array((lower[0], lower[1], 1.0), dtype=numpy.float32), numpy.array((lower[0], upper[1], 1.0), dtype=numpy.float32), numpy.array((upper[0], lower[1], 1.0), dtype=numpy.float32), numpy.array((upper[0], upper[1], 1.0), dtype=numpy.float32)]
  
  # Neighbours...
  near = (divisions - 1.0) / divisions
  far = 1.0 / divisions
  
  x_neighbour = []
  y_neighbour = []
  for ci in xrange(4):
    x_neighbour.append(near*corner[ci] + far*corner[ci^2])
    y_neighbour.append(near*corner[ci] + far*corner[ci^1])
  
  # Function to calculate the distance between two homogenius points...
  def dist(a, b):
    an = a[:2] / a[2]
    bn = b[:2] / b[2]
    
    return numpy.sqrt(numpy.square(an - bn).sum())
  
  # Loop and calculate all of the scales...
  scales = []
  for ci in xrange(4):
    ct = hg.dot(corner[ci])
    
    start = dist(corner[ci], x_neighbour[ci])
    end = dist(ct, hg.dot(x_neighbour[ci]))
    scales.append(end / start)
    
    start = dist(corner[ci], y_neighbour[ci])
    end = dist(ct, hg.dot(y_neighbour[ci]))
    scales.append(end / start)
  
  # Return the minimum and maximum...
  return min(scales), max(scales)
