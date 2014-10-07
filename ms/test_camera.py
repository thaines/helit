#! /usr/bin/env python

# Copyright 2014 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import numpy
import numpy.random

import cv
from utils.cvarray import *
from utils.prog_bar import ProgBar

from ms import MeanShift



# Does a full 3D camera multiplication simulation...



# Function that, given a position, returns the angle axis to make a camera at that position point at the origin (ignores roll!)...
def to_origin(pos):
  start = numpy.array([1,0,0], dtype=numpy.float32) # Starting direction of camera.
  end = numpy.array(pos, dtype=numpy.float32)
  end /= numpy.sqrt(numpy.square(end).sum())
  
  # Direction is cross product - calclate and normalise, taking care of the whole zero length vector issue...
  aa = numpy.cross(start, end)
  
  # Angle from dot product...
  ang = acos(numpy.start.dot(end))
  
  # Make sure the length of aa is right...
  if ang>1e-6:
    aa /= numpy.sqrt(numpy.square(aa).sum())
    aa *= ang
  
  return aa



# Create two camera distributions - a great circle at radius 4 pointing at the center in both cases...



# Add a bit of noise...



# Create two distributions...



# A function for converting a distribution into a ply file...



# Save the two distributions...



# Multiply them together...



# Save out the multiplication...
