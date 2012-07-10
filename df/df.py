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
      self.trees = map(lambda t: (t[0].clone(), t[1], t[2].copy() if t[2]!=None else None), other.trees)
      self.inc = other.inc
      self.grow = other.grow
      self.trainCount = other.trainCount
      
      self.evaluateCodeC = dict(other.evaluateCodeC)
      self.addCodeC = dict(other.addCodeC)
      self.errorCodeC = dict(other.errorCodeC)
      self.addTrainCodeC = dict(other.addTrainCodeC)
      self.useC = other.useC
    else:
      self.goal = None
      self.pruner = PruneCap()
      self.gen = None
      self.trees = [] # A list of tuples: (root node, oob score, draw) Last entry is None if self.grow is False, otherwise a numpy array of repeat counts for trainning for the exemplars.
      self.inc = False # True to support incrimental learning, False to not.
      self.grow = False # If true then during incrimental learning it checks pre-existing trees to see if they can grow some more each time.
      self.trainCount = 0 # Count of how many trainning examples were used to train with - this is so it knows how to split up the data when doing incrimental learning (between new and old exmeplars.). Also used to detect if trainning has occured.
      
      # Assorted code caches...
      self.evaluateCodeC = dict()
      self.addCodeC = dict()
      self.errorCodeC = dict()
      self.addTrainCodeC = dict()
      
      self.useC = True
  
  
  def setGoal(self, goal):
    """Allows you to set a goal object, of type Goal - must be called before doing anything, and must not be changed after anything is done."""
    assert(self.trainCount==0)
    self.addCodeC = dict()
    self.errorCodeC = dict()
    self.addTrainCodeC = dict()
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
    self.evaluateCodeC = dict()
    self.addCodeC = dict()
    self.errorCodeC = dict()
    self.addTrainCodeC = dict()
    self.gen = gen
  
  def getGen(self):
    """Returns the Generator object for the system."""
    return self.gen
  
  def setInc(self, inc, grow = False):
    """Set this to True to support incrimental learning, False to not. Having incrimental learning on costs extra memory, but has little if any computational affect. If incrimental learning is on you can also switch grow on, in which case as more data arrives it tries to split the leaf nodes of trees that have already been grown. Requires a bit more memory be used, as it needs to keep the indices of the training set for future growth. Note that the default pruner is entirly inappropriate for this mode - the pruner has to be set such that as more data arrives it will allow future growth."""
    assert(self.trainCount==0)
    self.inc = inc
    self.grow = grow
  
  def getInc(self):
    """Returns the status of incrimental learning - True if its enabled, False if it is not."""
    return self.inc
  
  def getGrow(self):
    """Returns True if the trees will be subject to further growth during incrimental learning, when they have gained enough data to subdivide further."""
    return self.grow
  
  def allowC(self, allow):
    """By default the system will attempt to compile and use C code instead of running the (much slower) python code - this allows you to force it to not use C code, or switch C back on if you had previously switched it off. Typically only used for speed comparisons and debugging, but also useful if the use of C code doesn't work on your system. Just be aware that the speed difference is galactic."""
    self.useC = allow
  
  
  def addTree(self, es, weightChannel = None, ret = False, dummy = False):
    """Adds an entirely new tree to the system given all of the new data. Uses all exemplars in the ExemplarSet, which can optionally include a channel with a single feature in it to weight the vectors; indicated via weightChannel. Typically this is used indirectly via the learn method, rather than by the user of an instance of this class."""

    # Arrange for code...
    if self.useC:
      key = es.key()
      
      if key not in self.addCodeC:
        self.addCodeC[key] = Node.initC(self.goal, self.gen, es)
      code = self.addCodeC[key]
      
      if key not in self.errorCodeC:
        self.errorCodeC[key] = Node.errorC(self.goal, self.gen, es)
      errCode = self.errorCodeC[key]
    else:
      code = None
      errCode = None
      
    # Special case code for a dummy run...
    if dummy:
      i = numpy.zeros(0, dtype=numpy.int32)
      w = numpy.ones(0, dtype=numpy.float32)
      
      if code!=None:  
        Node(self.goal, self.gen, self.pruner, es, i, w, code=code)
      if errCode!=None:
        Node.error.im_func(None, self.goal, self.gen, es, i, w, self.inc, code = errCode)
      return
    
    # First select which samples are to be used for trainning, and which for testing, calculating the relevant weights...
    draw = numpy.random.poisson(size=es.exemplars()) # Equivalent to a bootstrap sample, assuming an infinite number of exemplars are avaliable. Correct thing to do given that incrimental learning is an option.
    
    train = numpy.asarray(numpy.where(draw!=0)[0], dtype=numpy.int32)
    test = numpy.asarray(numpy.where(draw==0)[0], dtype=numpy.int32)
    
    if weightChannel==None:
      trainWeight = numpy.asarray(draw, dtype=numpy.float32)
      testWeight = None
    else:
      weights = es[weightChannel,:,0]
      trainWeight = numpy.asarray(draw * weights, dtype=numpy.float32)
      testWeight = numpy.asarray(weights, dtype=numpy.float32)
    
    if train.shape[0]==0: return # Safety for if it selects to use none of the items - do nothing...
    
    # Grow a tree...
    tree = Node(self.goal, self.gen, self.pruner, es, train, trainWeight, code=code)
    
    # Apply the goal-specific post processor to the tree...
    self.goal.postTreeGrow(tree, self.gen)
    
    # Calculate the oob error for the tree...
    if test.shape[0]!=0:
      error = tree.error(self.goal, self.gen, es, test, testWeight, self.inc, code=errCode)
    else:
      error = 1e100 # Can't calculate an error - record a high value so we lose the tree at the first avaliable opportunity, which is sensible behaviour given that we don't know how good it is.
    
    # Store it...
    if self.grow==False: draw = None
    if ret: return (tree, error, draw)
    else: self.trees.append((tree, error, draw))

  
  def lumberjack(self, count):
    """Once a bunch of trees have been learnt this culls them, reducing them such that there are no more than count. It terminates those with the highest error rate first, and does nothing if there are not enough trees to excede count. Typically this is used by the learn method, rather than by the object user."""
    if len(self.trees)>count:
      self.trees.sort(key = lambda t: t[1])
      self.trees = self.trees[:count]
  

  def learn(self, trees, es, weightChannel = None, clamp = None, mp = True, callback = None):
    """This learns a model given data, and, when it is switched on, will also do incrimental learning. trees is how many new trees to create - for normal learning this is just how many to make, for incrimental learning it is how many to add to those that have already been made - more is always better, within reason, but it is these that cost you computation and memory. es is the ExemplarSet containing the data to train on. For incrimental learning you always provide the previous data, at the same indices, with the new exemplars appended to the end. weightChannel allows you to give a channel containing a single feature if you want to weight the importance of the exemplars. clamp is only relevent to incrimental learning - it is effectivly a maximum number of trees to allow, where it throws away the weakest trees first. This is how incrimental learning works, and so must be set for that - by constantly adding new trees as new data arrives and updating the error metrics of the older trees (The error will typically increase with new data.) the less-well trainned (and typically older) trees will be culled. mp indicates if multiprocessing should be used or not - True to do so, False to not. Will automatically switch itself off if not supported."""
    
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
      
      key = es.key()
      code = self.addCodeC[key] if key in self.addCodeC else None
      errCode = self.errorCodeC[key] if key in self.errorCodeC else None
      
      if key not in self.addTrainCodeC:
        c = Node.addTrainC(self.goal, self.gen, es)
        self.addTrainCodeC[key] = c
        if c!=None:
          # Do a dummy run, to avoid multiproccess race conditions...
          i = numpy.zeros(0, dtype=numpy.int32)
          w = numpy.ones(0, dtype=numpy.float32)
          Node.addTrain.im_func(None, self.goal, self.gen, es, i, w, c)
      addCode = self.addTrainCodeC[key]
      
      if mp:
        result = pool.map_async(updateTree, map(lambda tree_tup: (self.goal, self.gen, self.pruner if self.grow else None, tree_tup, self.trainCount, newCount, es, weightChannel, (code, errCode, addCode), treesDone, numpy.random.randint(1000000000)), self.trees))
      else:
        newTrees = []
        for ti, tree_tup in enumerate(self.trees):
          if callback: callback(ti, totalTrees)
          data = (self.goal, self.gen, self.pruner if self.grow else None, tree_tup, self.trainCount, newCount, es, weightChannel, (code, errCode, addCode))
          newTrees.append(updateTree(data))
        self.trees = newTrees
    
    # Record how many exemplars were trained with most recently - needed for incrimental learning...
    self.trainCount = es.exemplars()
  
    # Create new trees...
    if trees!=0:
      if mp and trees>1:
        # There is a risk of a race condition caused by compilation - do a dummy run to make sure we compile in advance...
        self.addTree(es, weightChannel, dummy=True)
      
        # Set the runs going...
        newTreesResult = pool.map_async(mpGrowTree, map(lambda _: (self, es, weightChannel, treesDone, numpy.random.randint(1000000000)), xrange(trees)))
      
        # Wait for the runs to complete...
        while (not newTreesResult.ready()) and ((result==None) or (not result.ready())):
          newTreesResult.wait(0.1)
          if result: result.wait(0.1)
          if callback: callback(treesDone.value, totalTrees)
      
        # Put the result into the dta structure...
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
  
  
  def answer_types(self):
    """Returns a dictionary giving all the answer types that can be requested using the which parameter of the evaluate method. The keys give the string to be provided to which, whilst the values give human readable descriptions of what will be returned. 'best' is always provided, as a point estimate of the best answer; most models also provide 'prob', which is a probability distribution over 'best', such that 'best' is the argmax of 'prob'."""
    return self.goal.answer_types()

  def evaluate(self, es, index = slice(None), which = 'best', mp = False, callback = None):
    """Given some exemplars returns a list containing the output of the model for each exemplar. The returned list will align with the index, which defaults to everything and hence if not provided is aligned with es, the ExemplarSet. The meaning of the entrys in the list will depend on the Goal of the model and which: which can either be a single answer type from the goal object or a list of answer types, to get a tuple of answers for each list entry - the result is what the Goal-s answer method returns. The answer_types method passes through to provide relevent information. Can be run in multiprocessing mode if you set the mp variable to True - only worth it if you have a lot of data (Also note that it splits by tree, so each process does all data items but for just one of the trees.). Should not be called if size()==0."""
    if isinstance(index, slice): index = numpy.arange(*index.indices(es.exemplars()))
    
    # Handle the generation of C code, with caching...
    if self.useC:
      es_type = es.key()
      if es_type not in self.evaluateCodeC:
        self.evaluateCodeC[es_type] = Node.evaluateC(self.gen, es)
      code = self.evaluateCodeC[es_type]
    else:
      code = None
    
    # If multiprocessing has been requested set it up...
    if Pool==None: mp = False
    elif cpu_count()<2: mp = False
    if mp:
      pool = Pool()
      pool_size = cpu_count()
      manager = Manager()
      treesDone = manager.Value('i',0)

    # Collate the relevent stats objects...
    store = []
    if mp:
      # Dummy run, to avoid a race condition during compilation...
      if code!=None:
        ei = numpy.zeros(0, dtype=index.dtype)
        self.trees[0][0].evaluate([], self.gen, es, ei, code)
      
      # Do the actual work...
      result = pool.map_async(treeEval, map(lambda tree_error: (tree_error[0], self.gen, es, index, treesDone, code), self.trees))
      
      while not result.ready():
        result.wait(0.1)
        if callback: callback(treesDone.value, len(self.trees))

      store += result.get()
    else:
      for ti, (tree, _, _) in enumerate(self.trees):
        if callback: callback(ti, len(self.trees))
        res = [None] * es.exemplars()
        tree.evaluate(res, self.gen, es, index, code)
        store.append(res)
    
    # Merge and obtain answers for the output...
    if mp and index.shape[0]>1:
      step = index.shape[0]//pool_size
      excess = index.shape[0] - step*pool_size
      starts = map(lambda i: i*(step+1), xrange(excess))
      starts += map(lambda i: ranges[-1] + i*step, xrange(pool_size-excess))
      starts += [index.shape[0]]
      ranges = map(lambda a, b: slice(a, b), starts[:-1], starts[1:])
      
      ret = pool.map(getAnswer, map(lambda ran: (self.goal, map(lambda i: map(lambda s: s[i], store), index[ran]), which, es, index[ran], map(lambda t: t[0], self.trees)), ranges))
      ret = reduce(lambda a,b: a+b, ret)
    else:
      ret = self.goal.answer_batch(map(lambda i: map(lambda s: s[i], store), index), which, es, index, map(lambda t: t[0], self.trees))
    
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
  goal, gen, pruner, (tree, error, old_draw), prevCount, newCount, es, weightChannel, (code, errCode, addCode) = data[:9]
  if len(data)>10: numpy.random.seed(data[10])
  
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
    tree.addTrain(goal, gen, es, train, trainWeight, addCode)
  error = tree.error(goal, gen, es, test, testWeight, True, code=errCode)
  
  # If we are growing its time to grow the tree...
  draw = None if old_draw==None else numpy.append(old_draw, draw)
  
  if pruner!=None:
    index = numpy.where(draw!=0)[0]
    if weightChannel==None: weights = numpy.asarray(draw, dtype=numpy.float32)
    else: weights = es[weightChannel,:,prevCount:] * numpy.asarray(draw, dtype=numpy.float32)
    tree.grow(goal, gen, pruner, es, index, weights, 0, code)

  # If provided update the trees updated count...
  if len(data)>9: data[9].value += 1
    
  # Return the modified tree in a tuple with the updated error updated draw array...
  return (tree, error, draw)



def treeEval(data):
  """Used by the evaluate method when doing multiprocessing."""
  tree, gen, es, index, treesDone, code = data
  
  ret = [None] * es.exemplars()
  tree.evaluate(ret, gen, es, index, code)
  treesDone.value += 1
  return ret



def getAnswer(data):
  """Used for multiprocessing the calls to the answer method."""
  goal, stores, which, es, indices, trees = data
  
  return goal.answer_batch(stores, which, es, indices, trees)
