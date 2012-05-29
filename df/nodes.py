# Copyright 2012 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import numpy

try: from scipy import weave
except: weave = None

from utils.start_cpp import start_cpp



class Node:
  """Defines a node - these are the bread and butter of the system. Each decision tree is made out of nodes, each of which contains a binary test - if a feature vector passes the test then it travels to the true child node; if it fails it travels to the false child node (Note lowercase to avoid reserved word clash.). Eventually a leaf node is reached, where test==None, at which point the stats object is obtained, merged with the equivalent for all decision trees, and then provided as the answer to the user. Note that this python object uses the __slots__ techneque to keep it small - there will often be many thousands of these in a trained model."""
  __slots__ = ['test', 'true', 'false', 'stats', 'summary']
  
  def __init__(self, goal, gen, pruner, es, index = slice(None), weights = None, depth = 0, stats = None, entropy = None):
    """This recursivly grows the tree until the pruner says to stop. goal is a Goal object, so it knows what to optimise, gen a Generator object that provides tests for it to choose between and pruner is a Pruner object that decides when to stop growing. The exemplar set to train on is then provided, optionally with the indices of which members to use and weights to assign to them (weights align with the exemplar set, not with the relative exemplar indices defined by index. depth is the depth of this node - part of the recursive construction and used by the pruner as a possible reason to stop growing. stats is optionally provided to save on duplicate calculation, as it will be calculated as part of working out the split. entropy matches up with stats - if stats is provided so must it be."""
    
    if goal==None: return # For the clone method.
    
    # Calculate the stats if not provided, and get the entropy...
    if stats==None:
      self.stats = goal.stats(es, index, weights)
      entropy = goal.entropy(self.stats)
    else:
      self.stats = stats
    self.summary = None
    
    # Select the best test...
    ## Details of best test found so far...
    bestInfoGain = -1.0
    bestTest = None
    trueStats = None
    trueEntropy = None
    trueIndex = None
    falseStats = None
    falseEntropy = None
    falseIndex = None
    
    ## Get a bunch of tests and evaluate them against the goal...
    for test in gen.itertests(es, index, weights):
      # Apply the test, work out which items pass and which fail..
      res = gen.do(test, es, index)
      tIndex = index[res==True]
      fIndex = index[res==False]
      
      # Check its safe to continue...
      if tIndex.shape[0]==0 or fIndex.shape[0]==0: continue
      
      # Calculate the statistics...
      tStats = goal.stats(es, tIndex, weights)
      fStats = goal.stats(es, fIndex, weights)
      
      # Calculate the information gain...
      tEntropy = goal.entropy(tStats)
      fEntropy = goal.entropy(fStats)
      tWeight = float(tIndex.shape[0]) / float(tIndex.shape[0]+fIndex.shape[0])
      fWeight = float(fIndex.shape[0]) / float(tIndex.shape[0]+fIndex.shape[0])
      infoGain = entropy - tWeight*tEntropy - fWeight*fEntropy
      
      # Store if the best so far...
      if infoGain>bestInfoGain:
        bestInfoGain = infoGain
        bestTest = test
        trueStats = tStats
        trueEntropy = tEntropy
        trueIndex = tIndex
        falseStats = fStats
        falseEntropy = fEntropy
        falseIndex = fIndex
    
    # Use the pruner to decide if we should split or not, and if so do it...
    self.test = bestTest
    if bestTest!=None and pruner.keep(depth, trueIndex.shape[0], falseIndex.shape[0], infoGain, self)==True:
      # We are splitting - time to recurse...
      self.true = Node(goal, gen, pruner, es, trueIndex, weights, depth+1, trueStats, trueEntropy)
      self.false = Node(goal, gen, pruner, es, falseIndex, weights, depth+1, falseStats, falseEntropy)
    else:
      self.test = None
      self.true = None
      self.false = None
  
  def clone(self):
    """Returns a deep copy of this node. Note that it only copys the nodes - test, stats and summary are all assumed to contain invariant entities that are always replaced, never editted."""
    ret = Node(None, None, None, None)
    
    ret.test = self.test
    ret.true = self.true.clone() if self.true!=None else None
    ret.false = self.false.clone() if self.false!=None else None
    ret.stats = self.stats
    ret.summary = self.summary
    
    return ret


  def evaluate(self, out, gen, es, index = slice(None), code=None):
    """Given a set of exemplars, and possibly an index, this outputs the infered stats entities. Requires the generator so it can apply the tests. The output goes into out, a list indexed by exemplar position. If code is set to a string generated by evaluateC it uses that, for speed."""
    if isinstance(index, slice): index = numpy.arange(*index.indices(es.exemplars()))
    
    if isinstance(code, str) and weave!=None:
      init = start_cpp(code) + """
       if (Nindex[0]!=0)
       {
        // Create the Exemplar data structure...
         Exemplar * test_set = (Exemplar*)malloc(sizeof(Exemplar)*Nindex[0]);
         for (int i=0; i<Nindex[0]; i++)
         {
          test_set[i].index = index[i];
          test_set[i].next = &test_set[i+1];
         }
         test_set[Nindex[0]-1].next = 0;
       
        // Do the work...
         evaluate(self, data, test_set, out);
      
        // Clean up...
         free(test_set);
       }
      """
      
      data = es.tupleInputC()
      weave.inline(init, ['self', 'data', 'index', 'out'], support_code=code)
      return

    if self.test==None:
      # At a leaf - store this nodes stats object for the relevent nodes...
      for val in index: out[val] = self.stats
    else:
      # Need to split the index and send it down the two branches, as needed...
      res = gen.do(self.test, es, index)
      tIndex = index[res==True]
      fIndex = index[res==False]
      
      if tIndex.shape[0]!=0: self.true.evaluate(out, gen, es, tIndex)
      if fIndex.shape[0]!=0: self.false.evaluate(out, gen, es, fIndex)
  
  
  def evaluateC(self, gen, es):
    """For a given generator and exemplar set this returns the C code (Actually the support code.) that evaluate can use to accelerate its run time, or None if the various components involved do not support C code generation."""
    # First do accessors for the data set...
    try:
      escl = es.listCodeC('es')
    except NotImplementedError: return None
    
    code = ''
    for channel in escl:
      code += channel['get'] + '\n'
      code += channel['exemplars'] + '\n'
      code += channel['features'] + '\n'
    
    # Now throw in the test code...
    try:
      code += gen.testCodeC('do_test', escl) + '\n'
    except NotImplementedError: return None
    
    # Finally add in the code that recurses through and evaluates the nodes on the provided data...
    code += start_cpp() + """
    // So we can use an inplace modified linkied list to avoid malloc's during the real work...
    struct Exemplar
    {
     int index;
     Exemplar * next;
    };
    
    // Recursivly does the work...
    // node - node of the tree; for an external user this will always be the root.
    // data - python tuple containing the inputs needed at each stage.
    // test_set - Linked list of entities to analyse.
    // out - python list in which the output is to be stored.
    void evaluate(PyObject * node, PyObject * data, Exemplar * test_set, PyObject * out)
    {
     PyObject * test = PyObject_GetAttrString(node, "test");
     
     if (test==Py_None)
     {
      // Leaf node - assign the relevent stats to the members of the test-set...
       PyObject * stats = PyObject_GetAttrString(node, "stats");
      
       while (test_set)
       {
        Py_INCREF(stats);
        PyList_SetItem(out, test_set->index, stats);
        test_set = test_set->next;
       }
       
       Py_DECREF(stats);
     }
     else
     {
      // Branch node - use the test to split the test_set and recurse...
       // Tests...
        Exemplar * pass = 0;
        Exemplar * fail = 0;
        
        void * test_ptr = PyString_AsString(test);
        size_t test_len = PyString_Size(test);
        
        while (test_set)
        {
         Exemplar * next = test_set->next;
         
         if (do_test(data, test_ptr, test_len, test_set->index))
         {
          test_set->next = pass;
          pass = test_set;
         }
         else
         {
          test_set->next = fail;
          fail = test_set;
         }
         
         test_set = next;
        }
       
       // Recurse...
        if (pass)
        {
         PyObject * child = PyObject_GetAttrString(node, "true");
         evaluate(child, data, pass, out);
         Py_DECREF(child);
        }

        if (fail)
        {
         PyObject * child = PyObject_GetAttrString(node, "false");
         evaluate(child, data, fail, out);
         Py_DECREF(child);
        }
     }
     
     Py_DECREF(test);
    }
    """
    
    return code
  
  
  def size(self):
    """Returns how many nodes this (sub-)tree consists of."""
    if self.test==None: return 1
    else: return 1 + self.true.size() + self.false.size()
  
  
  def error(self, goal, gen, es, index = slice(None), weights = None, inc = False, store = None):
    """Once a tree is trained this method allows you to determine how good it is, using a test set, which would typically be its out-of-bag (oob) test set. Given a test set, possibly weighted, it will return its error rate, as defined by the goal. goal is the Goal object used for trainning, gen the Generator. Also supports incrimental testing, where the information gleened from the test set is stored such that new test exemplars can be added. This is the inc variable - True to store this (potentially large) quantity of information, and update it if it already exists, False to not store it and therefore disallow incrimental learning whilst saving memory. Note that the error rate will change by adding more training data as well as more testing data - you can call it with es==None to get an error score without adding more testing exemplars, assuming it has previously been called with inc==True. store is for internal use only."""
    
    # Bookkepping - work out if we need to return a score; make sure there is a store list...
    ret = store==None
    if ret:
      store = []
      if isinstance(index, slice): index = numpy.arange(*index.indices(es.exemplars()))
    
    # Update the summary at this node if needed...
    summary = None
    if es!=None:
      if self.summary==None: summary = goal.summary(es, index, weights)
      else: summary = goal.updateSummary(self.summary, es, index, weights)
      if inc: self.summary = summary
    
    # Either recurse to the leafs or include this leaf...
    if self.test==None:
      # A leaf...
      if summary!=None: store.append(goal.error(self.stats, summary))
    else:
      # Not a leaf...
      if es!=None:
        res = gen.do(self.test, es, index)
        tIndex = index[res==True]
        fIndex = index[res==False]
      
        if tIndex.shape[0]!=0: self.true.error(goal, gen, es, tIndex, weights, inc, store)
        elif inc==True: self.true.error(goal, gen, None, tIndex, weights, inc, store)
        if fIndex.shape[0]!=0: self.false.error(goal, gen, es, fIndex, weights, inc, store)
        elif inc==True: self.false.error(goal, gen, None, fIndex, weights, inc, store)
      else:
        self.true.error(goal, gen, es, index, weights, inc, store)
        self.false.error(goal, gen, es, index, weights, inc, store)
    
    # Calculate the weighted average of all the leafs all at once, to avoid an inefficient incrimental calculation, or just sum them up if a weight of None has been provided at any point...
    if ret and len(store)!=0:
      if None in map(lambda t: t[1], store):
        return sum(map(lambda t: t[0], store))
      else:
        store = numpy.asarray(store, dtype=numpy.float32)
        return numpy.average(store[:,0], weights=store[:,1])

  def removeIncError(self):
    """Culls the information for incrimental testing from the data structure, either to reset ready for new information or just to shrink the data structure after learning is finished."""
    self.summary = None
    if self.test!=None:
      self.false.removeSummary()
      self.true.removeSummary()
  
  def addTrain(self, goal, gen, es, index = slice(None), weights = None):
    """This allows you to update the nodes with more data, as though it was used for trainning. The actual tests are not affected, only the statistics at each node - part of incrimental learning."""
    if isinstance(index, slice): index = numpy.arange(*index.indices(es.exemplars()))
    
    # Update this nodes stats...
    self.stats = goal.updateStats(self.stats, es, index, weights)
    
    # Check if it has children that need updating...
    if self.test!=None:
      # Need to split the index and send it down the two branches, as needed...
      res = gen.do(self.test, es, index)
      tIndex = index[res==True]
      fIndex = index[res==False]
      
      if tIndex.shape[0]!=0: self.true.addTrain(goal, gen, es, tIndex, weights)
      if fIndex.shape[0]!=0: self.false.addTrain(goal, gen, es, fIndex, weights)
