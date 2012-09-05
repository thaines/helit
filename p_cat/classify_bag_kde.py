# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



from collections import defaultdict
import numpy
import numpy.random

from kde_inc.kde_inc import KDE_INC
from prob_cat import ProbCat



class ClassifyBagKDE(ProbCat):
  """This is the same as ClassifyKDE, except it has multiple instances of KDE_INC for each class. It uses bagging with a boostrap sample to train each set of classifiers, including one density estimate per bagging instance. Because it is incrimental it uses the Poisson(1) approximation of a bootstrap sample."""
  def __init__(self, prec, cap = 32, bag_size = 16, useSingle = False):
    """You provide the precision that is to be used (As a 2D numpy array, so it implicitly provides the number of dimensions.), the cap on the number of components in the KDE_INC objects and the number of models to maintain. useSingle indicates if it should also maintain a non-bootstraped model, which it will use when non-list accesses are requested. This happens to mean that with the active learning module for qbc based algorithms you get list mode, but for testing/using the model it reverts to a single model which will generally do better than a set of bootstrapped models."""
    self.prec = numpy.array(prec, dtype=numpy.float32)
    self.cap = cap
    
    self.bags = map(lambda _: dict(), xrange(bag_size)) # Dictionary for each model, indexed by category. Additionally, the density estimate is indexed by 'None'.
    self.counts = defaultdict(int) # Number of instances of each category that have been provided - models might have different numbers due to the bootstrap sampling. Uses None for the density estimate count.
    
    self.single = dict() if useSingle else None

  def setPrec(self, prec):
    """Changes the precision matrix - must be called before any samples are added, and must have the same dimensions as the current one."""
    self.prec = numpy.array(prec, dtype=numpy.float32)
    self.prior.setPrec(self.prec / (self.mult*self.mult))
  
  
  def priorAdd(self, sample):
    self.counts[None] += 1
    for bag in self.bags:
      if None not in bag:
        bag[None] = KDE_INC(self.prec, self.cap)
      for _ in xrange(numpy.random.poisson(1.0)):
        bag[None].add(sample)
    if self.single!=None:
      if None not in self.single:
        self.single[None] = KDE_INC(self.prec, self.cap)
      self.single[None].add(sample)
        

  def add(self, sample, cat):
    self.counts[cat] += 1
    for bag in self.bags:
      if cat not in bag:
        bag[cat] = KDE_INC(self.prec, self.cap)
      for _ in xrange(numpy.random.poisson(1.0)):
        bag[cat].add(sample)
    if self.single!=None:
      if cat not in self.single:
        self.single[cat] = KDE_INC(self.prec, self.cap)
      self.single[cat].add(sample)


  def getSampleTotal(self):
    return sum(self.counts.itervalues()) - self.counts[None]


  def getCatTotal(self):
    return len(self.counts) - (1 if None in self.counts else 0)

  def getCatList(self):
    return filter(lambda cat: cat!=None, self.counts.keys())

  def getCatCounts(self):
    ret = dict(self.counts)
    if None in ret: del ret[None]
    return ret
  
  
  def listMode(self):
    return True


  def getDataProb(self, sample, state = None):
    if self.single!=None:
      ret = defaultdict(float)
      for cat, mod in self.single.iteritems():
        ret[cat] = mod.prob(sample)
      return ret
    else:
      ret = defaultdict(float)
      for bag in self.bags:
        for key, mod in bag.iteritems():
          ret[key] += mod.prob(sample)
      for key in ret.iterkeys():
        ret[key] /= len(bag)
      return ret

  def getDataNLL(self, sample, state = None):
    if self.single!=None:
      ret = defaultdict(lambda: -1e64)
      for cat, mod in self.single.iteritems():
        ret[cat] = mod.nll(sample)
      return ret
    else:
      ret = defaultdict(list)
      for bag in self.bags:
        for key, mod in bag.iteritems():
          ret[key].append(mod.nll(sample))
      log_len_bags = numpy.log(len(self.bags))
      for key in ret.iterkeys():
        high = max(ret[key])
        ret[key] = high + numpy.log(numpy.exp(numpy.array(ret[key])-high).sum())
        ret[key] -= log_len_bags
      return ret


  def getDataProbList(self, sample, state = None):
    ret = []
    for bag in self.bags:
      ans = defaultdict(float)
      for key, mod in bag.iteritems():
        ans[key] = mod.prob(sample)
      ret.append(ans)
    return ret

  def getDataNLLList(self, sample, state = None):
    ret = []
    for bag in self.bags:
      ans = defaultdict(lambda: -1e64)
      for key, mod in bag.iteritems():
        ans[key] = mod.nll(sample)
      ret.append(ans)
    return ret
