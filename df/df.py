# Copyright 2012 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import numpy
import numpy.random

try:
  from multiprocessing import Pool, Manager, cpu_count
except:
  Pool = None


from exemplars import *

from goals import *
from pruners import *
from nodes import *

from tests import *
from generators import *
from gen_median import *
from gen_random import *
from gen_classify import *



class DF:
  """Master object for the decision forest system - provides the entire interface. Typical use consists of setting up the system - its goal, pruner and generator(s), providing data to train a model and then using the model to analyse new exemplars. Incrimental learning is also supported however, albeit a not very sophisticated implimentation. Note that this class is compatable with pythons serialisation routines, for if you need to save/load a trained model."""
  def __init__(self, other = None):
    """Initialises as a blank model, ready to be setup and run. Can also act as a copy constructor if you provide an instance of DF as a single parameter."""
    if isinstance(other, DF):
      self.goal = other.goal.clone() if other.goal!=None else None
      self.pruner = other.pruner.clone() if other.pruner!=None else None
      self.gen = other.gen.clone() if other.gen!=None else None
      self.trees = map(lambda t: (t[0].clone(), t[1]), other.trees)
      self.inc = other.inc
      self.trainCount = other.trainCount
    else:
      self.goal = None
      self.pruner = PruneCap()
      self.gen = None
      self.trees = [] # A list of pairs: (root node, oob score)
      self.inc = False # True to support incrimental learning, False to not.
      self.trainCount = 0 # Count of how many trainning examples were used to train with - this is so it knows how to split up the data when doing incrimental learning (between new and old exmeplars.). Also used to detect if trainning has occured.
  
  
  def setGoal(self, goal):
    """Allows you to set a goal object, of type Goal - must be called before doing anything, and must not be changed after anything is done."""
    assert(self.trainCount==0)
    self.goal = goal
    
  def getGoal(self):
    """Returns the curent Goal object."""
    return self.goal
  
  def setPruner(self, pruner):
    """Sets the pruner, which controls when to stop growing each tree. By default this is set to the PruneCap object with default parameters, though you might want to use getPruner to get it so you can adjust its parameters to match the problem at hand, as the pruner is important for avoiding overfitting."""
    assert(self.trainCount==0)
    self.pruner = pruner
  
  def getPruner(self):
    """Returns the current Pruner object."""
    return self.pruner
  
  def setGen(self, gen):
    """Allows you to set the Generator object from which node tests are obtained - must be set before anything happens. You must not change this once trainning starts."""
    assert(self.trainCount==0)
    self.gen = gen
  
  def getGen(self):
    """Returns the Generator object for the system."""
    return self.gen
  
  def setInc(self, inc):
    """Set this to True to support incrimental learning, False ot not. Having incrimental learning on costs extra memory, but has little if any computational affect."""
    assert(self.trainCount==0)
    self.inc = inc
  
  def getInc(self):
    """Returns the status of incrimental learning - True if its enabled, False if it is not."""
    return self.inc
  
  
  def addTree(self, es, weightChannel = None, ret = False):
    """Adds an entirely new tree to the system given all of the new data. Uses all exemplars in the ExemplarSet, which can optionally include a channel with a single feature in it to weight the vectors; indicated via weightChannel. Typically this is used indirectly via the learn method, rather than by the user of an instance of this class."""
    
    # First select which samples are to be used for trainning, and which for testing, calculating the relevant weights...
    draw = numpy.random.poisson(size=es.exemplars()) # Equivalent to a bootstrap sample, assuming an infinite number of exemplars are avaliable. Correct thing to do given that incrimental learning is an option.
    
    train = numpy.where(draw!=0)[0]
    test = numpy.where(draw==0)[0]
    
    if weightChannel==None:
      trainWeight = numpy.asarray(draw, dtype=numpy.float32)
      testWeight = None
    else:
      weights = es[weightChannel,:,0]
      trainWeight = numpy.asarray(draw * weights, dtype=numpy.float32)
      testWeight = numpy.asarray(weights, dtype=numpy.float32)
    
    if train.shape[0]==0: return # Safety for if it selects to use none of the items - do nothing...
    
    # Grow a tree...
    tree = Node(self.goal, self.gen, self.pruner, es, train, trainWeight)
    
    # Calculate the oob error for the tree...
    if test.shape[0]!=0:
      error = tree.error(self.goal, self.gen, es, test, testWeight, self.inc)
    else:
      error = 1e100 # Can't calculate an error - record a high value so we lose the tree at the first avaliable opportunity, which is sensible behaviour given that we don't know how good it is.
    
    # Store it...
    if ret: return (tree, error)
    else: self.trees.append((tree, error))

  
  def lumberjack(self, count):
    """Once a bunch of trees have been learnt this culls them, reducing them such that there are no more than count. It terminates those with the highest error rate first, and does nothing if there are not enough trees to excede count. Typically this is used by the learn method, rather than by the object user."""
    if len(self.trees)>count:
      self.trees.sort(key = lambda t: t[1])
      self.trees = self.trees[:count]
  

  def learn(self, trees, es, weightChannel = None, clamp = None, mp = True, callback = None):
    """This learns a model given data, and, when it is switched on, will also do incrimental learning. trees is how many new trees to create - for normal learning this is just how many to make, for incrimental learning it is how many to add to those that have already been made - more is always better, but it is these that cost you computation and memory. es is the ExemplarSet containing the data to train on. For incrimental learning you always provide the previous data, at the same indices, with the new exemplars appended to the end. weightChannel allows you to give a channel containing a single feature if you want to weight the importance of the exemplars. clamp is only relevent to incrimental learning - it is effectivly a maximum number of trees to allow, where it throws away the weakest trees first. This is how incrimental learning works, and so must be set for that - by constantly adding new trees as new data arrives and updating the error metrics of the older trees (The error will typically increase with new data.) the less-well trainned (and typically older) trees will be culled. mp indicates if multiprocessing should be used or not - True to do so, False to not. Will automatically switch itself off if not supported."""
    
    # Prepare for multiprocessing...
    if Pool==None: mp = False
    elif cpu_count()<2: mp = False
    if mp:
      pool = Pool()
      manager = Manager()
      treesDone = manager.Value('i',0)
      result = None
    
    totalTrees = len(self.trees) + trees
    
    # If this is an incrimental pass then first update all the pre-existing trees...
    if self.trainCount!=0:
      assert(self.inc)
      newCount = es.exemplars() - self.trainCount
      
      if mp:
        result = pool.map_async(updateTree, map(lambda tree_error: (self.goal, self.gen, tree_error[0], tree_error[1], self.trainCount, newCount, es, weightChannel, treesDone, numpy.random.randint(1000000000)), self.trees))
      else:
        newTrees = []
        for ti, (tree, error) in enumerate(self.trees):
          if callback: callback(ti, totalTrees)
          data = (self.goal, self.gen, tree, error, self.trainCount, newCount, es, weightChannel)
          newTrees.append(updateTree(data))
        self.trees = newTrees
    
    # Record how many exemplars were trained with most recently - needed for incrimental learning...
    self.trainCount = es.exemplars()
  
    # Create new trees...
    if mp:
      newTreesResult = pool.map_async(mpGrowTree, map(lambda _: (self, es, weightChannel, treesDone, numpy.random.randint(1000000000)), xrange(trees)))
      
      while (not newTreesResult.ready()) and ((result==None) or (not result.ready())):
        newTreesResult.wait(0.1)
        if result: result.wait(0.1)
        if callback: callback(treesDone.value, totalTrees)
        
      if result: self.trees = result.get()
      self.trees += filter(lambda tree: tree!=None, newTreesResult.get())
    else:
      for ti in xrange(trees):
        if callback: callback(len(self.trees)+ti, totalTrees)
        self.addTree(es, weightChannel)
    
    # Prune trees down to the right number of trees if needed...
    if clamp!=None: self.lumberjack(clamp)
    
    # Clean up if we have been multiprocessing...
    if mp:
      pool.close()
      pool.join()
  
  
  def evaluate(self, es, index = slice(None), best = False, mp = False, callback = None):
    """Given some exemplars returns a list containing the output of the model for each exemplar. The returned list will align with the index, which defaults to everything and hence if not provided is aligned with es, the ExemplarSet. The meaning of the entity will depend on the Goal of the model. If best is set to True then instead of the output of Goal.merge you get the output of Goal.best. If you set best to None then you get both - a 2-tuple of (output of Goal.merge, output of Goal.best). Can be run in multiprocessing mode if you set the mp variable to True - only worth it if you have a lot of data (Also note that it splits by tree, so each process does all data items but for just one of the trees.). Should not be called if size()==0."""
    if isinstance(index, slice): index = numpy.arange(*index.indices(es.exemplars()))
    
    # If multiprocessing has been requested set it up...
    if Pool==None: mp = False
    elif cpu_count()<2: mp = False
    if mp:
      pool = Pool()
      manager = Manager()
      treesDone = manager.Value('i',0)

    # Collate the relevent stats objects...
    store = dict()
    if mp:
      result = pool.map_async(treeEval, map(lambda tree_error: (tree_error[0], self.gen, es, index, treesDone), self.trees))
      
      while not result.ready():
        result.wait(0.1)
        if callback: callback(treesDone.value, len(self.trees))

      result = result.get()
      for res in result:
        for key, value in res.iteritems():
          if key in store: store[key] += value
          else: store[key] = value
    else:
      for ti, (tree, _) in enumerate(self.trees):
        if callback: callback(ti, len(self.trees))
        tree.evaluate(store, self.gen, es, index)
    
    # Merge and obtain answers for the output...
    ret = []
    for i in index:
      m = self.goal.merge(store[i])
      if best==None: ret.append((m, self.goal.best(m)))
      elif best==False: ret.append(m)
      elif best==True: ret.append(self.goal.best(m))
      else: raise Exception('Bad value for best') 
    
    # Clean up if we have been multiprocessing...
    if mp:
      pool.close()
      pool.join()

    # Return the answer...
    return ret
  
  
  def size(self):
    """Returns the number of trees within the forest."""
    return len(self.trees)
    
  def nodes(self):
    """Returns the total number of nodes in all the trees."""
    return sum(map(lambda t: t[0].size(), self.trees))
    
  def error(self):
    """Returns the average error of all the trees - meaning depends on the Goal at hand, but should provide an idea of how well the model is working."""
    return numpy.mean(map(lambda t: t[1], self.trees))



def mpGrowTree(data):
  """Part of the multiprocessing system - grows and returns a tree."""
  self, es, weightChannel, treesDone, seed = data
  numpy.random.seed(seed)
  ret = self.addTree(es, weightChannel, True)
  treesDone.value += 1
  return ret



def updateTree(data):
  """Updates a tree - kept external like this for the purpose of multiprocessing."""
  goal, gen, tree, error, prevCount, newCount, es, weightChannel = data[:8]
  if len(data)>9: numpy.random.seed(data[9])
  
  # Choose which of the new samples are train and which are test, prepare the relevent inputs...
  draw = numpy.random.poisson(size=newCount)
        
  train = numpy.where(draw!=0)[0] + prevCount
  test = numpy.where(draw==0)[0] + prevCount
        
  if weightChannel==None:
    trainWeight = numpy.asarray(draw, dtype=numpy.float32)
    testWeight = None
  else:
    weights = es[weightChannel,:,prevCount:]
    trainWeight = numpy.asarray(draw * weights, dtype=numpy.float32)
    testWeight = numpy.asarray(weights, dtype=numpy.float32)
        
  pad = numpy.zeros(prevCount, dtype=numpy.float32)
  trainWeight = numpy.append(pad, trainWeight)
  if testWeight!=None: testWeight = numpy.append(pad, testWeight)
      
  # Update both test and train for the tree...
  if train.shape[0]!=0:
    tree.addTrain(goal, gen, es, train, trainWeight)
  if test.shape[0]!=0:
    error = tree.error(goal, gen, es, test, testWeight, True)
  
  # If provided update the trees updated count...
  if len(data)>8: data[8].value += 1
    
  # Return the modified tree in a pair with the updated error...
  return (tree, error)



def treeEval(data):
  """Used by the evaluate method when doing multiprocessing."""
  tree, gen, es, index, treesDone = data
  
  ret = dict()
  tree.evaluate(ret, gen, es, index)
  treesDone.value += 1
  return ret
