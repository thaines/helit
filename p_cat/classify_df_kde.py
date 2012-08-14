# Copyright 2012 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import numpy
from df.df import *
from kde_inc.kde_inc import KDE_INC

from prob_cat import ProbCat



class ClassifyDF_KDE(ProbCat):
  """A classifier that uses decision forests. Includes the use of a density estimate decision forest as a psuedo-prior. The incrimental method used is rather simple, but still works reasonably well. Provides default parameters for the decision forests, but allows access to them for if you want to mess around. Internally the decision forests have two channels - the first is the data, the second the class."""
  def __init__(self, prec, cap, treeCount, incAdd = 1, testDims = 3, dimCount = 4, rotCount = 32):
    """prec is the precision matrix for the density estimate done with kernel density estimation; cap is the component cap for said kernel density estimate. treeCount is how many trees to use for the classifying decision forest whilst incAdd is how many to train for each new sample. testDims is the number of dimensions to use for each test, dimCount the number of combinations of dimensions to try for generating each nodes decision and rotCount the number of orientations to try for each nodes test generation."""
    # Support structures...
    self.cats = dict() # Dictionary from cat to internal indexing number.
    self.treeCount = treeCount
    self.incAdd = incAdd
    
    # Setup the classification forest...
    self.classify = DF()
    self.classify.setInc(True)
    self.classify.setGoal(Classification(None, 1))
    self.classify.setGen(LinearClassifyGen(0, 1, testDims, dimCount, rotCount))
    
    self.classifyData = MatrixGrow()
    self.classifyTrain = self.treeCount
    
    # Setup the density estimation kde...
    self.density = KDE_INC(prec, cap)
  
  def getClassifier(self):
    """Returns the decision forest used for classification."""
    return self.classify
  
  def getDensityEstimate(self):
    """Returns the KDE_INC used for density estimation, as a psuedo-prior."""
    return self.density
  
  def priorAdd(self, sample):
    self.density.add(sample)

  def add(self, sample, cat):
    if cat in self.cats:
      c = self.cats[cat]
    else:
      c = len(self.cats)
      self.cats[cat] = c
    
    self.classifyData.append(numpy.asarray(sample, dtype=numpy.float32), numpy.asarray(c, dtype=numpy.int32).reshape((1,)))
    self.classifyTrain += self.incAdd


  def getSampleTotal(self):
    return self.classifyData.exemplars()


  def getCatTotal(self):
    return len(self.cats)

  def getCatList(self):
    return self.cats.keys()

  def getCatCounts(self):
    if len(self.cats)==0: return dict()
    
    counts = numpy.bincount(self.classifyData[1,:,0])
    
    ret = dict()
    for cat, c in self.cats.iteritems():
      ret[cat] = counts[c] if c<counts.shape[0] else 0
    return ret
  
  
  def listMode(self):
    return True


  def getDataProb(self, sample, state = None):
    # Update the model as needed - this will potentially take some time...
    if self.classifyTrain!=0 and self.classifyData.exemplars()!=0:
      self.classify.learn(min(self.classifyTrain, self.treeCount), self.classifyData, clamp = self.treeCount, mp=False)
      self.classifyTrain = 0
    
    # Generate the result and create and return the right output structure...
    ret = dict()
    
    if self.classify.size()!=0:
      eval_c = self.classify.evaluate(MatrixES(sample), which = 'gen')[0]
      for cat, c in self.cats.iteritems():
        ret[cat] = eval_c[c] if c<eval_c.shape[0] else 0.0

    ret[None] = self.density.prob(sample)

    return ret
  
  
  def getDataProbList(self, sample, state = None):
    # Update the models as needed - this will potentially take some time...
    if self.classifyTrain!=0 and self.classifyData.exemplars()!=0:
      self.classify.learn(min(self.classifyTrain, self.treeCount), self.classifyData, clamp = self.treeCount, mp=False)
      self.classifyTrain = 0
    
    # Fetch the required information...
    if self.classify.size()!=0:
      eval_c = self.classify.evaluate(MatrixES(sample), which = 'gen_list')[0]
    else:
      return [{None:1.0}]
    
    eval_d = self.density.prob(sample)
    
    # Construct and return the output...
    ret = []

    for ec in eval_c:
      r = {None:eval_d}

      for cat, c in self.cats.iteritems():
        r[cat] = ec[c] if c<ec.shape[0] else 0.0

      ret.append(r)

    return ret
