# Copyright 2012 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import numpy
import numpy.random

from utils.start_cpp import start_cpp



class Generator:
  """A generator - provides lots of test entities designed to split an exemplar set via a (python) generator method (i.e. using yield). When a tree is constructed it is provided with a generator and each time it wants to split the generator is given the exemplar set and an index into the relevent exemplars to split on, plus an optional weighting. It then yields a set of test entities, which are applied and scored via the goal, such that the best can be selected. This is more inline with extremelly random decision forests, but there is nothing stopping the use of a goal-aware test generator that does do some kind of optimisation, potentially yielding just one test entity. The generator will contain the most important parameters of the decision forest, as it controls how the test entities are created and how many are tried - selecting the right generator and its associated parameters is essential for performance. An actual Generator is expected to also inherit from its associated Test object, such that it provides the do method. This is necesary as a test entity requires access to its associated Test object to work."""
  
  def clone(self):
    """Returns a (deep) copy of this object."""
    raise NotImplementedError
    
  def itertests(self, es, index, weights = None):
    """Generates test entities that split the provided features into two sets, yielding them one at a time such that the caller can select the best, according to the current goal. It really is allowed to do whatever it likes to create these test entities, which means it provides an insane amount of customisation potential, if possibly rather too much choice. es is the exemplar set, whilst index is the set of exemplars within the exemplar set it is creating a test for. weights optionally provides a weight for each exemplar, aligned with es."""
    raise NotImplementedError
    yield



class MergeGen(Generator):
  """As most generators only handle a specific kind of data (discrete, continuous, one channel at a time.) the need arises to merge multiple generators for a given problem, in the sense that when iterating the generators tests it provides the union of all tests by all of the contained generators. Alternativly, the possibility exists to get better results by using multiple generators with different properties, as the best test from all provided will ultimatly be selected. This class merges upto 256 generators as one. The 256 limit comes from the fact the test entities provided by it have to encode which generator made them, so that the do method can send the test entity to the right test object, and it only uses a byte - in the unlikelly event that more are needed a hierarchy can be used, though your almost certainly doing it wrong if you get that far."""
  def __init__(self, *args):
    """By default constructs the object without any generators in it, but you can provide generators to it as parameters to the constructor."""
    self.gens = args
    assert(len(self.gens)<=256)
  
  def clone(self):
    ret = MergeGen()
    ret.gens = map(lambda g: g.clone(), self.gens)
    return ret
  
  def add(self, gen):
    """Adds a generator to the provided set. Generators can be in multiple MergeGen/RandomGen objects, just as long as a loop is not formed."""
    self.gens.append(gen)
    assert(len(self.gens)<=256)
  
  
  def itertests(self, es, index, weights = None):
    for c, gen in enumerate(self.gens):
      code = chr(c)
      for test in gen.itertests(es, index, weights):
        yield code+test
  
  
  def do(self, test, es, index = slice(None)):
    code = ord(test[0])
    return self.gens[code].do(test[1:], es, index)
  
  def testCodeC(self, name, exemplar_list):
    # Add the children...
    ret = ''
    for i, gen in enumerate(self.gens):
      ret += gen.testCodeC(name + '_%i'%i, exemplar_list)
    
    # Put in the final test function...
    ret += start_cpp()
    ret += 'bool %s(PyObject * data, void * test, size_t test_length, int exemplar)\n'%name
    ret += '{\n'
    ret += 'void * sub_test = ((char*)test) + 1;\n'
    ret += 'size_t sub_test_length = test_length - 1;\n'
    ret += 'int which = *(unsigned char*)test;\n'
    ret += 'switch(which)\n'
    ret += '{\n'
    for i in xrange(len(self.gens)):
      ret += 'case %i: return %s_%i(data, sub_test, sub_test_length, exemplar);\n'%(i, name, i)
    ret += '}\n'
    ret += 'return 0;\n' # To stop the compiler issuing a warning.
    ret += '}\n'
    
    return ret



class RandomGen(Generator):
  """This generator contains several generators, and randomly selects one to provide the tests each time itertests is called - not entirly sure what this could be used for, but it can certainly add some more randomness, for good or for bad. Supports weighting and merging multiple draws from the set of generators contained within. Has the same limit of 256 that MergeGen has, for the same reasons."""
  def __init__(self, draws = 1, *args):
    """draws is the number of draws from the list of generators to merge to provide the final output. Note that it is drawing with replacement, and will call an underlying generator twice if it gets selected twice. After the draws parameter you can optionally provide generators, which will be put into the created object, noting that they will all have a selection weight of 1."""
    self.gens = map(lambda a: (a,1.0), args)
    self.draws = draws
    assert(len(self.gens)<=256)
  
  def clone(self):
    ret = MergeGen(self.draws)
    ret.gens = map(lambda g: g.clone(), self.gens)
    return ret
    
  def add(self, gen, weight = 1.0):
    """Adds a generator to the provided set. Generators can be in multiple MergeGen/RandomGen objects, just as long as a loop is not formed. You can also provide a weight, to bias how often particular generators are selected."""
    self.gens.append((gen, weight))
    assert(len(self.gens)<=256)
  
  
  def itertests(self, es, index, weights = None):
    # Select which generators get to play...
    w = numpy.asarray(map(lambda g: g[1], self.gens))
    w /= w.sum()
    toDo = numpy.random.multinomial(self.draws, w)
    
    # Go through and iterate the tests of each generator in turn, the number of times requested...
    for genInd in numpy.where(toDo!=0)[0]:
      code = chr(genInd)
      for _ in xrange(toDo[genInd]):
        for test in self.gens[genInd][0].itertests(es, index, weights):
          yield code+test


  def do(self, test, es, index = slice(None)):
    code = ord(test[0])
    return self.gens[code][0].do(test[1:], es, index)

  def testCodeC(self, name, exemplar_list):
    # Add the children...
    ret = ''
    for i, (gen, _) in enumerate(self.gens):
      ret += gen.testCodeC(name + '_%i'%i, exemplar_list)
    
    # Put in the final test function...
    ret += start_cpp()
    ret += 'bool %s(PyObject * data, void * test, size_t test_length, int exemplar)\n'%name
    ret += '{\n'
    ret += 'void * sub_test = ((char*)test) + 1;\n'
    ret += 'size_t sub_test_length = test_length - 1;\n'
    ret += 'int which = *(unsigned char*)test;\n'
    ret += 'switch(which)\n'
    ret += '{\n'
    for i in xrange(len(self.gens)):
      ret += 'case %i: return %s_%i(data, sub_test, sub_test_length, exemplar);\n'%(i, name, i)
    ret += '}\n'
    ret += 'return 0;\n' # To stop the compiler issuing a warning.
    ret += '}\n'
    
    return ret
