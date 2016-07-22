# Copyright 2016 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import os.path
import unittest

import math
import numpy

from utils.make import make_mod



# Compile the code if need be...
make_mod('maxflow_c', os.path.dirname(__file__), ['maxflow_c.h', 'maxflow_c.c'])



# Import the compiled module into this space, so we can pretend they are one and the same, just with automatic compilation...
from maxflow_c import *



# Some unit testing...
class TestMaxFlow(unittest.TestCase):
  def test_degenerate(self):
    mf = MaxFlow(2,1)
    mf.set_source(0)
    mf.set_sink(1)
    mf.set_edges(numpy.array([0]), numpy.array([1]))
    mf.set_flow_cap(numpy.array([8.0], dtype=numpy.float32), numpy.array([3.0], dtype=numpy.float32))
    
    mf.solve()
    
    self.assertTrue(math.fabs(mf.max_flow-3.0)<1e-12)
  
  
  def test_chain(self):
    mf = MaxFlow(5,4)
    mf.set_source(0)
    mf.set_sink(4)
    
    e_from = [0,1,2,3]
    e_to   = [1,2,3,4]
    mf.set_edges(numpy.array(e_from), numpy.array(e_to))
    
    cost = numpy.array([3.0,7.0,5.0,8.0])
    mf.set_flow_cap(cost, cost)
    
    mf.solve()
    
    self.assertTrue(math.fabs(mf.max_flow-3.0)<1e-12)
  
  
  def test_dual(self):
    mf = MaxFlow(4,5)
    mf.set_source(0)
    mf.set_sink(3)
    
    e_from = [0,0,1,1,2]
    e_to   = [1,2,2,3,3]
    mf.set_edges(numpy.array(e_from), numpy.array(e_to))
    
    cost = numpy.array([8.0,2.0,5.0,3.0,9.0])
    mf.set_flow_cap(cost, cost)
    
    mf.solve()
    
    self.assertTrue(math.fabs(mf.max_flow-10.0)<1e-12)
    
    
  def test_line(self):
    mf = MaxFlow(7, 14)
    mf.set_source(0)
    mf.set_sink(6)
    
    e_from = [  0,  0,  0,  0,  0,  1,  2,  3,  4,  1,  2,  3,  4,  5]
    e_to   = [  1,  2,  3,  4,  5,  2,  3,  4,  5,  6,  6,  6,  6,  6]
    cost   = [9.0,7.0,5.0,3.0,1.0,3.0,3.0,3.0,3.0,1.0,3.0,5.0,7.0,9.0]
    
    mf.set_edges(numpy.array(e_from), numpy.array(e_to))
    cost = numpy.array(cost)
    mf.set_flow_cap(cost, cost)
    
    mf.solve()
    
    out = numpy.empty(7, dtype=numpy.int32)
    mf.store_side(out, -1, 1)
    
    self.assertTrue(out[0]==-1)
    self.assertTrue(out[1]==-1)
    self.assertTrue(out[2]==-1)
    self.assertTrue(out[4]==1)
    self.assertTrue(out[5]==1)
    self.assertTrue(out[6]==1)

    
  def test_mult_layer(self):
    mf = MaxFlow(8, 15)
    mf.set_source(0)
    mf.set_sink(7)
    
    begin = [  0,  0,  0,  1,  2,  1,  1,  2,  2,  3,  4,  5,  4,  5,  6]
    end   = [  1,  2,  3,  2,  3,  4,  5,  5,  6,  6,  5,  6,  7,  7,  7]
    neg   = [0.0,0.0,0.0,2.0,5.0,0.0,0.0,0.0,1.0,0.0,0.0,3.0,0.0,0.0,0.0]
    pos   = [5.0,4.0,9.0,3.0,2.0,2.0,1.0,6.0,5.0,3.0,2.0,0.0,6.0,8.0,5.0]
    
    mf.set_edges(numpy.array(begin), numpy.array(end))
    mf.set_flow_cap(numpy.array(neg), numpy.array(pos))
    
    mf.solve()
    
    self.assertTrue(math.fabs(mf.max_flow-15.0)<1e-12)
    
    neg_rem = numpy.empty(15, dtype=numpy.float32)
    pos_rem = numpy.empty(15, dtype=numpy.float32)
    mf.store_unused(neg_rem, pos_rem)
    
    for i in xrange(15):
      self.assertTrue(math.fabs(neg_rem[i]+pos_rem[i]-neg[i]-pos[i])<1e-12)



# If run from the command line do the unit tests...
if __name__ == '__main__':
    unittest.main()
