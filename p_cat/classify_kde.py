# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import numpy
from kde_inc.kde_inc import KDE_INC

from prob_cat import ProbCat



class ClassifyKDE(ProbCat):
  """A classifier that uses the incrimental kernel density estimate model for each category. It keeps a 'psuedo-prior', a KDE_INC with an (optionally) larger variance that contains all the samples. Uses entities that can index a dictionary for categories."""
  def __init__(self, prec, cap = 32, mult = 1.0):
    """You provide the precision that is to be used (As a 2D numpy array, so it implicitly provides the number of dimensions.), the cap on the number of components in the KDE_INC objects and the multiplier for the standard deviation of the components in the 'psuedo-prior'."""
    self.prec = numpy.array(prec, dtype=numpy.float32)
    self.cap = cap
    self.mult = mult

    self.prior = KDE_INC(self.prec / (self.mult*self.mult), self.cap)
    self.cats = dict() # Dictionary indexed by category going to the associated KDE_INC object.
    self.counts = None

  def setPrec(self, prec):
    """Changes the precision matrix - must be called before any samples are added, and must have the same dimensions as the current one."""
    self.prec = numpy.array(prec, dtype=numpy.float32)
    self.prior.setPrec(self.prec / (self.mult*self.mult))


  def priorAdd(self, sample):
    self.prior.add(sample)

  def add(self, sample, cat):
    if cat not in self.cats: self.cats[cat] = KDE_INC(self.prec, self.cap)
    
    self.cats[cat].add(sample)
    self.counts = None


  def getSampleTotal(self):
    return sum(map(lambda mm: mm.samples(), self.cats.itervalues()))


  def getCatTotal(self):
    return len(self.cats)

  def getCatList(self):
    return self.cats.keys()

  def getCatCounts(self):
    if self.counts==None:
      self.counts = dict()
      for cat, mm in self.cats.iteritems():
        self.counts[cat] = mm.samples()

    return self.counts
    

  def getDataProb(self, sample, state = None):
    ret = dict()
    ret[None] = self.prior.prob(sample)
    for cat, mm in self.cats.iteritems(): ret[cat] = mm.prob(sample)
    return ret

  def getDataNLL(self, sample, state = None):
    ret = dict()
    ret[None] = self.prior.nll(sample)
    for cat, mm in self.cats.iteritems(): ret[cat] = mm.nll(sample)
    return ret
