# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



from gcp import gcp

from prob_cat import ProbCat



class ClassifyGaussian(ProbCat):
  """A simplistic Gaussian classifier, that uses a single Gaussian to represent each category/the prior. It is of course fully Bayesian. It keeps a prior that is worth the number of dimensions with the mean and covariance of all the samples provided for its construction. Implimentation is not very efficient, though includes some caching to stop things being too slow."""
  def __init__(self, dims):
    """dims is the number of dimensions."""
    self.dims = dims
    self.prior = gcp.GaussianPrior(self.dims)
    
    self.cats = dict() # Dictionary indexed by categories with a value of the associated GaussianPrior object, without the current prior included - it is merged in as needed.
    self.counts = None # Dictionary going to sample counts for each category.
    self.cst = None # Dictionary indexed as above, but going to student-t distributions representing the current state. A caching layer that gets invalidated as needed.


  def priorAdd(self, sample):
    self.prior.addSample(sample)
    self.cst = None
  
  def add(self, sample, cat):
    if cat not in self.cats: self.cats[cat] = gcp.GaussianPrior(self.dims)

    self.cats[cat].addSample(sample)
    self.counts = None
    self.cst = None


  def getSampleTotal(self):
    return sum(map(lambda gp: int(gp.getN()), self.cats.itervalues()))


  def getCatTotal(self):
    return len(self.cats)

  def getCatList(self):
    return self.cats.keys()

  def getCatCounts(self):
    if self.counts==None:
      self.counts = dict()
      for cat, gp in self.cats.iteritems():
        self.counts[cat] = int(gp.getN())

    return self.counts


  def getStudentT(self):
    """Returns a dictionary with categories as keys and StudentT distributions as values, these being the probabilities of samples belonging to each class with the actual draw from the posterior integrated out. Also stores the prior, under a key of None."""
    if self.cst==None:
      self.cst = dict()

      # First prep the prior...
      prior = gcp.GaussianPrior(self.prior)

      prior.make_safe()
      prior.reweight()
      self.cst[None] = prior.intProb()

      # Then iterate the categories and extract their student-t's, after updating with the prior...
      for cat, gp in self.cats.iteritems():
        ngp = gcp.GaussianPrior(gp)
        ngp.addGP(prior)
        self.cst[cat] = ngp.intProb()

    return self.cst


  def getDataProb(self, sample, state = None):
    ret = dict()
    for cat, st in self.getStudentT().iteritems(): ret[cat] = st.prob(sample)
    return ret
