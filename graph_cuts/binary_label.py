# Copyright 2016 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import numpy

from maxflow import MaxFlow



class BinaryLabel:
  """Uses maximum flow to solve a MRF on a n-dimensional grid, under the usual condition that the costs be metric/sub-modular. (i.e. the costs can not encourage dissimilarity.) The output labels will be False and True, hence the naming scheme."""
  def __init__(self, shape):
    """You initialise with the shape of the grid - a tuple of sizes, the length of which is the number of dimensions."""
    
    # Constant cost - whilst not of much interest to most users we keep it anyway...
    self.constant = 0.0
    
    # Cost of assigning each random variable to one of the two classes...
    self.costFalse = numpy.zeros(shape, dtype=numpy.float32)
    self.costTrue = numpy.zeros(shape, dtype=numpy.float32)
    
    # Cost of adjacent labels being different, as a list indexed by the dimension involved. Each entry is a cost matrix with the dimension of the costs reduced by one. Submodularity is not enforced until use - negatives can occur...
    # (Interface accepts the 2x2 grid of costs and converts them on the fly.)
    self.costDifferent = map(lambda d: numpy.zeros(map(lambda e: shape[e] if e!=d else shape[e]-1, xrange(len(shape))), dtype=numpy.float32), xrange(len(shape)))
    
    # Create the MaxFlow object...
    nodes = reduce(lambda a,b: a*b, shape)
    vertices = 2 + nodes
    edges = 2 * nodes + sum(map(lambda cd: reduce(lambda a,b: a*b, cd.shape), self.costDifferent))
    self.mf = MaxFlow(vertices, edges)
    
    # Create the source/sink and link up all the edges...
    self.mf.set_source(nodes)
    self.mf.set_sink(nodes+1)
    
    eb = 0
    
    ## Source to nodes...
    const = numpy.empty(nodes, dtype=numpy.int32)
    const[:] = nodes
    node_indices = numpy.arange(nodes, dtype=numpy.int32)
    self.mf.set_edges_range(eb, const, node_indices)
    eb += nodes
    
    ## Nodes to sink...
    const[:] = nodes+1
    self.mf.set_edges_range(eb, node_indices, const)
    eb += nodes
    
    ## Interconnects for each dimension...
    node_indices = node_indices.reshape(self.costFalse.shape)
    
    for dim in xrange(len(self.costFalse.shape)):
      index = [slice(None)] * len(self.costFalse.shape)
      
      index[dim] = slice(-1)
      start = node_indices[index].flatten()
      
      index[dim] = slice(1,None)
      end = node_indices[index].flatten()
      
      self.mf.set_edges_range(eb, start, end)
      eb += start.shape[0]
  
  
  def reset(self):
    """Resets all the costs to zero, so they can be rebuilt if you want."""
    self.constant = 0.0
    self.costFalse[:] = 0.0
    self.costTrue[:]  = 0.0
    for cd in self.costDifferent: cd[:] = 0.0
  
  
  def shape(self):
    """Returns the shape of the structure"""
    return self.costFalse.shape
  
  
  def addCostConstant(self, costConstant):
    """A constant cost is stored, so you can get the right cost out at the end - this allows you to offset it. Can't actually think of any scenario where you would use it, but here it is just incase"""
    self.constant += costConstant
  
  
  def addCostFalse(self, costFalse):
    """Incriments the stored costs of assigning the label False with those in the array. Input array must broadcast to the shape provided on construction."""
    self.costFalse += costFalse
    
  def addCostTrue(self, costTrue):
    """Incriments the stored costs of assigning the label False with those in the array. Input array must broadcast to the shape provided on construction."""
    self.costTrue += costTrue
  
  
  def addCostDifferent(self, dim, cd):
    """Adds to the cost of being different - you might want to instead provide the costs for the various neighbour combinations, in which case use the addCostAdjacent method. First you provide the index of the dimension for which the costs are to be added, the a numpy array of costs that must broadcast to the provided shape, with one subtracted from the column of the dimension, to account for the links being between random variables."""
    self.costDifferent[dim] += cd
  
  def addCostAdjacent(self, dim, cdFalseFalse, cdFalseTrue, cdTrueFalse, cdTrueTrue):
    """Does the same as addCostDifferent, but allows you to provide the full set of costs for all 4 possible adjacencies, and it updates the costs accordingly, including offsetting. In the event you provide something that is not sub-modular this can result in negative costs in the system, which will be clamped to zero at runtime. The variables provided are named so the element with the lowest index comes first."""
    self.constant += cdFalseFalse.sum()
    
    index = [slice(None)] * len(self.costFalse.shape)
    index[dim] = slice(-1)
    self.costTrue[index] += 0.5*(cdTrueFalse + cdTrueTrue - cdFalseTrue - cdFalseFalse)
    index[dim] = slice(1,None)
    self.costTrue[index] += 0.5*(cdFalseTrue + cdTrueTrue - cdTrueFalse - cdFalseFalse)
    
    self.costDifferent[dim] += 0.5*(cdFalseTrue + cdTrueFalse - cdFalseFalse - cdTrueTrue)


  def setLonelyCost(self, target_cost):
    """Goes through all costs and increases them for each random variable such that every pixel has at least the given cost of having the same label as one of its neighbours - a cute noise reduction hack."""
    # Calculate the cost for each node of being different from all its neighbours...
    current = numpy.zeros(self.costFalse.shape, dtype=numpy.float32)
    
    for dim, cost in enumerate(self.costDifferent):
      index = [slice(None)] * len(current.shape)
      
      index[dim] = slice(-1)
      numpy.maximum(current[index], cost, current[index])
      
      index[dim] = slice(1, None)
      numpy.maximum(current[index], cost, current[index])
    
    # Use that cost to calculate the change required...
    delta = numpy.clip(target_cost - current, 0.0, 1e32)
    
    # Increase the costs, uniformly, to achieve the affect...
    for dim, cost in enumerate(self.costDifferent):
      index = [slice(None)] * len(current.shape)
      
      index[dim] = slice(-1)
      cost += delta[index]
      
      index[dim] = slice(1, None)
      cost += delta[index]
  
  
  def fix(self, fix, almost_inf = 1e32):
    """Given an array of integers that matches the shape this fixes the values of some labels - where the array is zero the label is left free to take either value, where it is negative it is forced to be False, when it is positive it is forced to be True. This updates the cost function, so you should call this last before the costs are used."""
    self.costTrue[fix<0]  = almost_inf
    self.costFalse[fix>0] = almost_inf


  def solve(self):
    """Solves for the contained costs, returning a boolean numpy array giving the highest probability labeling, in a tuple with its cost - (array, cost)."""
    
    # Input the costs...
    eb = 0

    flatTrue = numpy.clip(self.costTrue.flatten(), 0.0, 1e32)
    self.mf.set_flow_cap_range(eb, numpy.zeros(flatTrue.shape, dtype=numpy.float32), flatTrue)
    eb += flatTrue.shape[0]
    
    flatFalse = numpy.clip(self.costFalse.flatten(), 0.0, 1e32)
    self.mf.set_flow_cap_range(eb, numpy.zeros(flatFalse.shape, dtype=numpy.float32), flatFalse)
    eb += flatFalse.shape[0]
    
    for dim, costDiff in enumerate(self.costDifferent):
      flat = numpy.clip(costDiff.flatten(), 0.0, 1e32)
      self.mf.set_flow_cap_range(eb, flat, flat)
      eb += flat.shape[0]
    
    # Solve...
    self.mf.solve()
    
    # Extract the result into a numpy array...
    result = numpy.empty(self.costFalse.shape, dtype=numpy.int8)
    result = result.flatten()
    self.mf.store_side_range(0, result, 0, 1)
    result = result.reshape(self.costFalse.shape).astype(numpy.bool)
  
    # Return the tuple of assignment/cost...
    return (result, self.constant + self.mf.max_flow)
