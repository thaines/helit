# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



from dpgmm.dpgmm import DPGMM

from prob_cat import ProbCat



class ClassifyDPGMM(ProbCat):
  """A classifier that uses a Dirichlet process Gaussian mixture model (DPGMM) for each category. Also includes a psuedo-prior in the form of an extra DPGMM that you can feed. Trains them incrimentally, increasing the mixture component cap when that results in an improvement in model performance. Be aware that whilst this is awesome its memory consumption can be fierce, and its a computational hog. Includes the ability to switch off incrimental learning, which can save some time if your not using the model between trainning samples."""
  def __init__(self, dims, runs = 1):
    """dims is the number of dimensions the input vectors have, whilst runs is how many starting points to converge from for each variational run. Increasing runs helps to avoid local minima at the expense of computation, but as it often converges well enough with the first attempt, so this is only for the paranoid."""
    self.dims = dims
    self.runs = runs

    self.inc = True

    self.prior = DPGMM(self.dims)
    self.cats = dict() # Dictionary indexed by category going to the associated DPGMM object.
    self.counts = None


  def priorAdd(self, sample):
    self.prior.add(sample)
    if self.inc and self.prior.setPrior():
      self.prior = self.prior.multiGrowSolve(self.runs)

  def add(self, sample, cat):
    if cat not in self.cats: self.cats[cat] = DPGMM(self.dims)

    self.cats[cat].add(sample)
    if self.inc and self.cats[cat].setPrior():
      self.cats[cat] = self.cats[cat].multiGrowSolve(self.runs)

    self.counts = None


  def setInc(self, state):
    """With a state of False it disables incrimental learning until further notice, with a state of True it reenables it, and makes sure that it is fully up to date by updating everything. Note that when reenabled it assumes that enough data is avaliable, and will crash if not, unlike the incrimental approach that just twiddles its thumbs - in a sense this is safer if you want to avoid bad results."""
    self.inc = state

    if self.inc:
      self.prior.setPrior()
      self.prior = self.prior.multiGrowSolve(self.runs)

      for cat in self.cats.iterkeys():
        self.cats[cat].setPrior()
        self.cats[cat] = self.cats[cat].multiGrowSolve(self.runs)


  def getSampleTotal(self):
    sum(map(lambda mm: mm.size(), self.cats.itervalues()))


  def getCatTotal(self):
    return len(self.cats)

  def getCatList(self):
    return self.cats.keys()

  def getCatCounts(self):
    if self.counts==None:
      self.counts = dict()
      for cat, mm in self.cats.iteritems():
        self.counts[cat] = mm.size()

    return self.counts


  def getDataProb(self, sample, state = None):
    ret = dict()
    for cat, mm in self.cats.iteritems(): ret[cat] = mm.prob(sample)
    return ret
