# Copyright 2012 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import numpy
import numpy.random

try: from scipy import weave
except: weave = None

from utils.start_cpp import start_cpp



class Node:
  """Defines a node - these are the bread and butter of the system. Each decision tree is made out of nodes, each of which contains a binary test - if a feature vector passes the test then it travels to the true child node; if it fails it travels to the false child node (Note lowercase to avoid reserved word clash.). Eventually a leaf node is reached, where test==None, at which point the stats object is obtained, merged with the equivalent for all decision trees, and then provided as the answer to the user. Note that this python object uses the __slots__ techneque to keep it small - there will often be many thousands of these in a trained model."""
  __slots__ = ['test', 'true', 'false', 'stats', 'summary']
    
  def __init__(self, goal, gen, pruner, es, index = slice(None), weights = None, depth = 0, stats = None, entropy = None, code = None):
    """This recursivly grows the tree until the pruner says to stop. goal is a Goal object, so it knows what to optimise, gen a Generator object that provides tests for it to choose between and pruner is a Pruner object that decides when to stop growing. The exemplar set to train on is then provided, optionally with the indices of which members to use and weights to assign to them (weights align with the exemplar set, not with the relative exemplar indices defined by index. depth is the depth of this node - part of the recursive construction and used by the pruner as a possible reason to stop growing. stats is optionally provided to save on duplicate calculation, as it will be calculated as part of working out the split. entropy should match up with stats. The static method initC can be called to generate code that can be used by this constructor to accelerate test selection, but only if it is passed in."""
    
    if goal==None: return # For the clone method.
    
    # Calculate the stats if not provided, and get the entropy...
    if stats==None:
      self.stats = goal.stats(es, index, weights)
    else:
      self.stats = stats
    self.summary = None
    
    # Use the grow method to do teh actual growth...
    self.give_birth(goal, gen, pruner, es, index, weights, depth, entropy, code)
  
  def give_birth(self, goal, gen, pruner, es, index = slice(None), weights = None, depth = 0, entropy = None, code = None):
    """This recursivly grows the tree until the pruner says to stop. goal is a Goal object, so it knows what to optimise, gen a Generator object that provides tests for it to choose between and pruner is a Pruner object that decides when to stop growing. The exemplar set to train on is then provided, optionally with the indices of which members to use and weights to assign to them (weights align with the exemplar set, not with the relative exemplar indices defined by index. depth is the depth of this node - part of the recursive construction and used by the pruner as a possible reason to stop growing. entropy should match up with self.stats. The static method initC can be called to generate code that can be used to accelerate test selection, but only if it is passed in."""
    if entropy==None: entropy = goal.entropy(self.stats)

    # Select the best test...
    if isinstance(code, str) and weave!=None:
      # Do things in C...
      init = start_cpp(code) + """
      if (Nindex[0]!=0)
      {
       srand48(rand);
       
       // Create the Exemplar data structure, in  triplicate!..
        Exemplar * items = (Exemplar*)malloc(sizeof(Exemplar)*Nindex[0]);
        Exemplar * splitItems = (Exemplar*)malloc(sizeof(Exemplar)*Nindex[0]);
        Exemplar * temp = (Exemplar*)malloc(sizeof(Exemplar)*Nindex[0]);
        for (int i=0; i<Nindex[0]; i++)
        {
         int ind = index[i];
         float we = weights[ind];
        
         items[i].index = ind;
         items[i].weight = we;
         items[i].next = &items[i+1];
        
         splitItems[i].index = ind;
         splitItems[i].weight = we;
         splitItems[i].next = &splitItems[i+1];
        
         temp[i].next = &temp[i+1];
        }
        items[Nindex[0]-1].next = 0;
        splitItems[Nindex[0]-1].next = 0;
        temp[Nindex[0]-1].next = 0;
       
       // Do the work...
        selectTest(out, data, items, splitItems, temp, entropy);
      
       // Clean up...
        free(temp);
        free(splitItems);
        free(items);
      }
      """
      
      data = es.tupleInputC()
      out = dict()
      rand = numpy.random.randint(-1000000000,1000000000)
      
      if weights==None: weights = numpy.ones(es.exemplars(), dtype=numpy.float32)
      weave.inline(init, ['out', 'data', 'index', 'weights', 'entropy', 'rand'], support_code=code)
      if index.shape[0]==0: return
      
      bestTest = out['bestTest']
      if bestTest!=None:
        bestInfoGain = out['bestInfoGain']
        trueStats = out['trueStats']
        trueEntropy = out['trueEntropy']
        trueIndex = out['trueIndex']
        falseStats = out['falseStats']
        falseEntropy = out['falseEntropy']
        falseIndex = out['falseIndex']
      
        trueIndex.sort()  # Not needed to work - to improve cache coherance.
        falseIndex.sort() # "
    else:
      if index.shape[0]==0: return
      
      # Do things in python...
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
      
        if weights==None:
          tWeight = float(tIndex.shape[0])
          fWeight = float(fIndex.shape[0])
        else:
          tWeight = weights[tIndex].sum()
          fWeight = weights[fIndex].sum()
        div = tWeight + fWeight
        tWeight /= div
        fWeight /= div
      
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
    if bestTest!=None and pruner.keep(depth, trueIndex.shape[0], falseIndex.shape[0], bestInfoGain, self)==True:
      # We are splitting - time to recurse...
      self.true = Node(goal, gen, pruner, es, trueIndex, weights, depth+1, trueStats, trueEntropy, code)
      self.false = Node(goal, gen, pruner, es, falseIndex, weights, depth+1, falseStats, falseEntropy, code)
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


  @staticmethod
  def initC(goal, gen, es):
    # Get the evaluateC code, which this code is dependent on...
    code = Node.evaluateC(gen, es, 'es')
    if code==None: return None
    
    # Add in the generator code...
    escl = es.listCodeC('es')
    try:
      gCode, gState = gen.genCodeC('gen', escl)
    except NotImplementedError:
      return None
    code += gCode
    
    # Add in the goal code...
    try:
      gDic = goal.codeC('goal', escl)
    except NotImplementedError:
      return None
    
    try:
      code += gDic['stats']
      code += gDic['entropy']
    except KeyError:
      return None
    
    # And finally add in the code we need to specifically handle the selection of a test for a node...
    code += start_cpp() + """
    
    // out - A dictionary to output into; data - The list of entities that represent the exemplar set; items - The set of items to optimise the test for, splitItems - A copy of items, which will be screwed with; temp - Like items, same size, so we can keep a temporary copy; entropy - The entropy of the set of items.
    void selectTest(PyObject * out, PyObject * data, Exemplar * items, Exemplar * splitItems, Exemplar * temp, float entropy)
    {
     // Setup the generator...
      %(gState)s state;
      gen_init(state, data, items);
    
     // Loop the tests, scoring each one and keeping the best so far...
      void * bestTest = 0;
      size_t bestTestLen = 0;
      void * bestPassStats = 0;
      size_t bestPassStatsLen = 0;
      float bestPassEntropy = -1.0;
      Exemplar * bestPassItems = temp;
      int bestPassItemsLen = 0;
      void * bestFailStats = 0;
      size_t bestFailStatsLen = 0;
      float bestFailEntropy = -1.0;
      Exemplar * bestFailItems = 0;
      int bestFailItemsLen = 0;
      float bestGain = 0.0;
      
      Exemplar * pass = splitItems;
      void * passStats = 0;
      size_t passStatsLength = 0;
      
      Exemplar * fail = 0;
      void * failStats = 0;
      size_t failStatsLength = 0;
      
      while (gen_next(state, data, items))
      {
       // Apply the test...
        Exemplar * newPass = 0;
        float passWeight = 0.0;
        
        Exemplar * newFail = 0;
        float failWeight = 0.0;

        while (pass)
        {
         Exemplar * next = pass->next;
         
         if (do_test(data, state.test, state.length, pass->index))
         {
          pass->next = newPass;
          newPass = pass;
          passWeight += pass->weight;
         }
         else
         {
          pass->next = newFail;
          newFail = pass;
          failWeight += pass->weight;
         }
         
         pass = next;
        }
        
        while (fail)
        {
         Exemplar * next = fail->next;
         
         if (do_test(data, state.test, state.length, fail->index))
         {
          fail->next = newPass;
          newPass = fail;
          passWeight += fail->weight;
         }
         else
         {
          fail->next = newFail;
          newFail = fail;
          failWeight += fail->weight;
         }
         
         fail = next;
        }
        
        pass = newPass;
        fail = newFail;
        
        if ((pass==0)||(fail==0))
        {
         // All data has gone one way - this scernario can not provide an advantage so ignore it.
          continue;
        }

       // Generate the stats objects and entropy...
        goal_stats(data, pass, passStats, passStatsLength);
        goal_stats(data, fail, failStats, failStatsLength);
        
        float passEntropy = goal_entropy(passStats, passStatsLength);
        float failEntropy = goal_entropy(failStats, failStatsLength);
        
       // Calculate the information gain...
        float div = passWeight + failWeight;
        passWeight /= div;
        failWeight /= div;
        
        float gain = entropy - passWeight*passEntropy - failWeight*failEntropy;
       
       // If it is the largest store its output for future consumption...
        if (gain>bestGain)
        {
         bestTestLen = state.length;
         bestTest = realloc(bestTest, bestTestLen);
         memcpy(bestTest, state.test, bestTestLen);
         
         bestPassStatsLen = passStatsLength;
         bestPassStats = realloc(bestPassStats, bestPassStatsLen);
         memcpy(bestPassStats, passStats, bestPassStatsLen);
         
         bestFailStatsLen = failStatsLength;
         bestFailStats = realloc(bestFailStats, bestFailStatsLen);
         memcpy(bestFailStats, failStats, bestFailStatsLen);
         
         bestPassEntropy = passEntropy;
         bestFailEntropy = failEntropy;
         bestGain = gain;
         
         Exemplar * storeA = bestPassItems;
         Exemplar * storeB = bestFailItems;
         bestPassItems = 0;
         bestPassItemsLen = 0;
         bestFailItems = 0;
         bestFailItemsLen = 0;
         
         Exemplar * targPass = pass;
         while (targPass)
         {
          // Get an output node...
           Exemplar * out;
           if (storeA)
           {
            out = storeA;
            storeA = storeA->next;
           }
           else
           {
            out = storeB;
            storeB = storeB->next;
           }
           
          // Store it...
           out->next = bestPassItems;
           bestPassItems = out;
           bestPassItemsLen++;
           
           out->index = targPass->index;
          
          targPass = targPass->next;
         }
         
         Exemplar * targFail = fail;
         while (targFail)
         {
          // Get an output node...
           Exemplar * out;
           if (storeA)
           {
            out = storeA;
            storeA = storeA->next;
           }
           else
           {
            out = storeB;
            storeB = storeB->next;
           }
           
          // Store it...
           out->next = bestFailItems;
           bestFailItems = out;
           bestFailItemsLen++;
           
           out->index = targFail->index;
          
          targFail = targFail->next;
         }
        }
      }
    
     // Output the best into the provided dictionary - quite a lot of information...
      if (bestTest!=0)
      {
       PyObject * t = PyFloat_FromDouble(bestGain);
       PyDict_SetItemString(out, "bestInfoGain", t);
       Py_DECREF(t);
      
       t = PyString_FromStringAndSize((char*)bestTest, bestTestLen);
       PyDict_SetItemString(out, "bestTest", t);
       Py_DECREF(t);
      
       t = PyString_FromStringAndSize((char*)bestPassStats, bestPassStatsLen);
       PyDict_SetItemString(out, "trueStats", t);
       Py_DECREF(t);
   
       t = PyFloat_FromDouble(bestPassEntropy);
       PyDict_SetItemString(out, "trueEntropy", t);
       Py_DECREF(t);
      
       PyArrayObject * ta = (PyArrayObject*)PyArray_FromDims(1, &bestPassItemsLen, NPY_INT32);
       int i = 0;
       while (bestPassItems)
       {
        *(int*)(ta->data + ta->strides[0]*i) = bestPassItems->index;
        i++;
        bestPassItems = bestPassItems->next;
       }
       PyDict_SetItemString(out, "trueIndex", (PyObject*)ta);
       Py_DECREF(ta);

       t = PyString_FromStringAndSize((char*)bestFailStats, bestFailStatsLen);
       PyDict_SetItemString(out, "falseStats", t);
       Py_DECREF(t);
   
       t = PyFloat_FromDouble(bestFailEntropy);
       PyDict_SetItemString(out, "falseEntropy", t);
       Py_DECREF(t);
      
       ta = (PyArrayObject*)PyArray_FromDims(1, &bestFailItemsLen, NPY_INT32);
       i = 0;
       while (bestFailItems)
       {
        *(int*)(ta->data + ta->strides[0]*i) = bestFailItems->index;
        i++;
        bestFailItems = bestFailItems->next;
       }
       PyDict_SetItemString(out, "falseIndex", (PyObject*)ta);
       Py_DECREF(ta);
      }
      else
      {
       PyDict_SetItemString(out, "bestTest", Py_None);
       Py_INCREF(Py_None);
      }
      
     // Clean up...
      free(bestTest);
      free(bestPassStats);
      free(bestFailStats);
      free(passStats);
      free(failStats);
    }
    """%{'gState':gState}
    
    return code


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
  
  
  @staticmethod
  def evaluateC(gen, es, esclName = 'es'):
    """For a given generator and exemplar set this returns the C code (Actually the support code.) that evaluate can use to accelerate its run time, or None if the various components involved do not support C code generation."""
    # First do accessors for the data set...
    try:
      escl = es.listCodeC(esclName)
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
    // So we can use an inplace modified linkied list to avoid malloc's during the real work (Weight is included because this code is reused by the generator system, which needs it.)...
    struct Exemplar
    {
     int index;
     float weight;
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
  
  
  def error(self, goal, gen, es, index = slice(None), weights = None, inc = False, store = None, code = None):
    """Once a tree is trained this method allows you to determine how good it is, using a test set, which would typically be its out-of-bag (oob) test set. Given a test set, possibly weighted, it will return its error rate, as defined by the goal. goal is the Goal object used for trainning, gen the Generator. Also supports incrimental testing, where the information gleened from the test set is stored such that new test exemplars can be added. This is the inc variable - True to store this (potentially large) quantity of information, and update it if it already exists, False to not store it and therefore disallow incrimental learning whilst saving memory. Note that the error rate will change by adding more training data as well as more testing data - you can call it with es==None to get an error score without adding more testing exemplars, assuming it has previously been called with inc==True. store is for internal use only. code can be provided by the relevent parameter, as generated by the errorC method, allowing a dramatic speedup."""
    if code!=None and weave!=None:
       init = start_cpp(code) + """
       float err = 0.0;
       float weight = 0.0;
        
       if (dummy==0) // To allow for a dummy run.
       {
        if (Nindex[0]!=0)
        {
         Exemplar * test_set = (Exemplar*)malloc(sizeof(Exemplar)*Nindex[0]);
         for (int i=0; i<Nindex[0]; i++)
         {
          int ind = index[i];
          test_set[i].index = ind;
          test_set[i].weight = weights[ind];
          test_set[i].next = &test_set[i+1];
         }
         test_set[Nindex[0]-1].next = 0;
         
         error(self, data, test_set, err, weight, incNum==1);
       
         free(test_set);
        }
        else
        {
         error(self, data, 0, err, weight, incNum==1);
        }
       }
       
       return_val = err;
       """
       
       data = es.tupleInputC()
       dummy = 1 if self==None else 0
       incNum = 1 if inc else 0
       if weights==None: weights = numpy.ones(es.exemplars(), dtype=numpy.float32)
       return weave.inline(init, ['self', 'data', 'index', 'weights', 'incNum', 'dummy'], support_code=code)
    else:
      # Book-keeping - work out if we need to return a score; make sure there is a store list...
      ret = store==None
      if ret:
        store = []
        if isinstance(index, slice): index = numpy.arange(*index.indices(es.exemplars()))
    
      # Update the summary at this node if needed...
      summary = None
      if es!=None and index.shape[0]!=0:
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
  
  @staticmethod
  def errorC(goal, gen, es, esclName = 'es'):
    """Provides C code that can be used by the error method to go much faster. Makes use of a goal, a generator, and an exampler set, and the code will be unique for each keying combination of these. Will return None if C code generation is not supported for the particular combination."""
    # First do accessors for the data set...
    try:
      escl = es.listCodeC(esclName)
    except NotImplementedError: return None
    
    code = ''
    for channel in escl:
      code += channel['get'] + '\n'
      code += channel['exemplars'] + '\n'
      code += channel['features'] + '\n'
    
    # Throw in the test code...
    try:
      code += gen.testCodeC('do_test', escl) + '\n'
    except NotImplementedError: return None
    
    # Definition of Exemplar...
    code += start_cpp() + """
    // So we can use an inplace modified linkied list to avoid malloc's during the real work...
     struct Exemplar
     {
      int index;
      float weight;
      Exemplar * next;
     };
    """
    
    # Add the needed goal code...
    try:
      gDic = goal.codeC('goal', escl)
    except NotImplementedError:
      return None
    
    try:
      code += gDic['summary']
      code += gDic['updateSummary']
      code += gDic['error']
    except KeyError:
      return None
    
    # The actual code...
    code += start_cpp() + """     
    // Recursivly calculates the error whilst updating the summaries...
    // node - node of the tree; for an external user this will always be the root.
    // data - python tuple containing the inputs needed at each stage.
    // test_set - Linked list of entities to use to generate/update the error.
    // err - Variable into which the error will be output. Must be 0.0 on call.
    // weight - Weight that can be used in the error calculation - basically temporary storage. Must be 0.0 on call.
     void error(PyObject * node, PyObject * data, Exemplar * test_set, float & err, float & weight, bool inc)
     {
      // Calculate/update the summary at this node, but only store it if inc is true...
       void * sum = 0;
       size_t sumLen = 0;
       
       PyObject * summary = PyObject_GetAttrString(node, "summary");
       if (summary==Py_None)
       {
        goal_summary(data, test_set, sum, sumLen);
       }
       else
       {
        sumLen = PyString_Size(summary);
        sum = realloc(sum, sumLen);
        memcpy(sum, PyString_AsString(summary), sumLen);
       
        goal_updateSummary(data, test_set, sum, sumLen);
       }
       Py_DECREF(summary);
       
       if (inc)
       {
        PyObject * t = PyString_FromStringAndSize((char*)sum, sumLen);
        PyObject_SetAttrString(node, "summary", t);
        Py_DECREF(t);
       }
      
      // If there is a test then recurse, otherwise calculate and include the error...
       PyObject * test = PyObject_GetAttrString(node, "test");
       
       if (test==Py_None)
       {
        // Leaf node - calculate and store the error...
         PyObject * stats = PyObject_GetAttrString(node, "stats");
         
         void * s =  PyString_AsString(stats);
         size_t sLen = PyString_Size(stats);
         
         goal_error(s, sLen, sum, sumLen, err, weight);
         
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
         if ((pass!=0)||inc)
         {
          PyObject * child = PyObject_GetAttrString(node, "true");
          error(child, data, pass, err, weight, inc);
          Py_DECREF(child);
         }

        if ((fail!=0)||inc)
        {
         PyObject * child = PyObject_GetAttrString(node, "false");
         error(child, data, fail, err, weight, inc);
         Py_DECREF(child);
        }
       }
       
      // Clean up...
       Py_DECREF(test);
       free(sum);
     }
    """
    
    return code

  def removeIncError(self):
    """Culls the information for incrimental testing from the data structure, either to reset ready for new information or just to shrink the data structure after learning is finished."""
    self.summary = None
    if self.test!=None:
      self.false.removeIncError()
      self.true.removeIncError()
  
  
  def addTrain(self, goal, gen, es, index = slice(None), weights = None, code = None):
    """This allows you to update the nodes with more data, as though it was used for trainning. The actual tests are not affected, only the statistics at each node - part of incrimental learning. You can optionally proivde code generated by the addTrainC method to give it go faster stripes."""
    if isinstance(index, slice): index = numpy.arange(*index.indices(es.exemplars()))

    if code!=None:
      init = start_cpp(code) + """
      if (dummy==0) // To allow for a dummy run.
      {
       Exemplar * test_set = (Exemplar*)malloc(sizeof(Exemplar)*Nindex[0]);
       for (int i=0; i<Nindex[0]; i++)
       {
        int ind = index[i];
        test_set[i].index = ind;
        test_set[i].weight = weights[ind];
        test_set[i].next = &test_set[i+1];
       }
       test_set[Nindex[0]-1].next = 0;
         
       addTrain(self, data, test_set);
       
       free(test_set);
      }
      """
       
      data = es.tupleInputC()
      dummy = 1 if self==None else 0
      if weights==None: weights = numpy.ones(es.exemplars(), dtype=numpy.float32)
      return weave.inline(init, ['self', 'data', 'index', 'weights', 'dummy'], support_code=code)
    else:
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
  
  @staticmethod
  def addTrainC(goal, gen, es, esclName = 'es'):
    """Provides C code that the addTrain method can use to accelerate itself - standard rules about code being unique for each combination of input types applies."""
    # First do accessors for the data set...
    try:
      escl = es.listCodeC(esclName)
    except NotImplementedError: return None
    
    code = ''
    for channel in escl:
      code += channel['get'] + '\n'
      code += channel['exemplars'] + '\n'
      code += channel['features'] + '\n'
    
    # Throw in the test code...
    try:
      code += gen.testCodeC('do_test', escl) + '\n'
    except NotImplementedError: return None
    
    # Definition of Exemplar...
    code += start_cpp() + """
    // So we can use an inplace modified linkied list to avoid malloc's during the real work...
     struct Exemplar
     {
      int index;
      float weight;
      Exemplar * next;
     };
    """
    
    # Add the needed goal code...
    try:
      gDic = goal.codeC('goal', escl)
    except NotImplementedError:
      return None
    
    try:
      code += gDic['updateStats']
    except KeyError:
      return None
    
    code += start_cpp() + """
    void addTrain(PyObject * node, PyObject * data, Exemplar * test_set)
    {
     // Update the stats at this node...
      PyObject * stats = PyObject_GetAttrString(node, "stats");
      
      size_t stLen = PyString_Size(stats);
      void * st = malloc(stLen);
      memcpy(st, PyString_AsString(stats), stLen);
      
      goal_updateStats(data, test_set, st, stLen);
       
      PyObject * t = PyString_FromStringAndSize((char*)st, stLen);
      PyObject_SetAttrString(node, "stats", t);
      Py_DECREF(t);
      
      free(st);
      Py_DECREF(stats);
     
     // If its not a leaf recurse down and do its children also...
      PyObject * test = PyObject_GetAttrString(node, "test");
       
      if (test!=Py_None)
      {
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
        if (pass!=0)
        {
         PyObject * child = PyObject_GetAttrString(node, "true");
         addTrain(child, data, pass);
         Py_DECREF(child);
        }

        if (fail!=0)
        {
         PyObject * child = PyObject_GetAttrString(node, "false");
         addTrain(child, data, fail);
         Py_DECREF(child);
        }
      }
      
      Py_DECREF(test);
    }
    """
    
    return code
  
  
  def grow(self, goal, gen, pruner, es, index = slice(None), weights = None, depth = 0, code = None):
    """This is called on a tree that has already grown - it recurses to the children and continues as though growth never stopped. This can be to grow the tree further using a less stritc pruner or to grow the tree after further information has been added. code can be passed in as generated by the initC static method, and will be used to optimise test generation."""
    if self.test==None:
      self.give_birth(goal, gen, pruner, es, index, weights, depth, code = code)
    else:
      res = gen.do(self.test, es, index)
      tIndex = index[res==True]
      fIndex = index[res==False]
      
      self.true.grow(goal, gen, pruner, es, tIndex, weights, depth+1, code)
      self.false.grow(goal, gen, pruner, es, fIndex, weights, depth+1, code)
