# Copyright 2012 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



class Pruner:
  """This abstracts the decision of when to stop growing a tree. It takes various statistics and stops growing when some condition is met."""
  
  def clone(self):
    """Returns a copy of this object."""
    raise NotImplementedError
    
  def keep(self, depth, trueCount, falseCount, infoGain, node):
    """Each time a node is split this method is called to decide if the split should be kept or not - it returns True to keep (And hence the children will be recursivly split, and have keep called on them, etc..) and False to discard the nodes children and stop. depth is how deep the node in question is, where the root node is 0, its children 1, and do on. trueCount and falseCount indicate how many data points in the training set go each way, whilst infoGain is the information gained by the split. Finally, node is the actual node incase some more complicated analysis is desired - at the time of passing in its test and stats will exist, but everything else will not."""
    raise NotImplementedError



class PruneCap(Pruner):
  """A simple but effective Pruner implimentation - simply provides a set of thresholds on depth, number of training samples required to split and information gained - when any one of the thresholds is tripped it stops further branching."""
  def __init__(self, maxDepth = 8, minTrain = 8, minGain = 1e-3, minDepth = 2):
    """maxDepth is the maximum depth of a node in the tree, after which it stops - remember that the maximum node count based on this threshold increases dramatically as this number goes up, so don't go too crazy. minTrain is the smallest size node it will consider for further splitting. minGain is a lower limit on how much information gain a split must provide to be accepted. minDepth overrides the minimum node size - as long as the node count does not reach zero in either branch it will always split to the given depth - used to force it to at least learn something."""
    self.maxDepth = maxDepth
    self.minDepth = minDepth
    self.minTrain = minTrain
    self.minGain = minGain
  
  def clone(self):
    return PruneCap(self.maxDepth, self.minTrain, self.minGain)
    
  def keep(self, depth, trueCount, falseCount, infoGain, node):
    if depth>=self.maxDepth: return False
    if depth>=self.minDepth and (trueCount+falseCount)<self.minTrain: return False
    if infoGain<self.minGain: return False
    return True
  
  
  def setMinDepth(self, minDepth):
    """Sets the minimum tree growing depth - trees will be grown at least this deep, baring insurmountable issues."""
    self.minDepth = minDepth
    
  def setMaxDepth(self, maxDepth):
    """Sets the depth cap on the trees."""
    self.maxDepth = maxDepth
  
  def setMinTrain(self, minTrain):
    """Sets the minimum number of nodes allowed to be split."""
    self.minTrain = minTrain
  
  def setMinGain(self, mingain):
    """Sets the minimum gain that is allowed for a split to be accepted."""
    self.minGain = mingain
