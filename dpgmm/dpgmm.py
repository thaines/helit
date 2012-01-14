# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import math
import random
import numpy
import numpy.random
import scipy.special

from gcp import gcp



class DPGMM:
  """A Dirichlet process Gaussian mixture model, implimented using the mean-field variational method, with the stick tying rather than capping method such that incrimental usage works. For those unfamiliar with Dirichlet processes the key thing to realise is that each stick corresponds to a mixture component for density estimation or a cluster for clustering, so the stick cap is the maximum number of these entities (Though it can choose to use less sticks than supplied.). As each stick has a computational cost standard practise is to start with a low number of sticks and then increase the count, updating the model each time, until no improvement to the model is obtained with further sticks. Note that because the model is fully Bayesian the output is in fact a probability distribution over the probability distribution from which the data is drawn, i.e. instead of the point estimate of a Gaussian mixture model you get back a probability distribution from which you can draw a Gaussian mixture model, though shortcut methods are provided to get the probability/component membership probability of features. Regarding the model it actually has an infinite number of sticks, but as computers don't have an infinite amount of memory or computation the sticks count is capped. The infinite number of sticks past the cap are still modeled in a limited manor, such that you can get the probability in clustering of a sample/feature belonging to an unknown cluster (i.e. one of the infinite number past the stick cap.). It also means that there is no explicit cluster count, as it is modelling a probability distribution over the number of clusters - if you need a cluster count the best choice is to threshold on the component weights, as returned by sampleMixture(...) or intMixture(...), i.e. keep only thise that are higher than a percentage of the highest weight."""
  def __init__(self, dims, stickCap = 1):
    """You initialise with the number of dimensions and the cap on the number of sticks to have. Note that the stick cap should be high enough for it to represent enough components, but not so high that you run out of memory. The better option is to set the stick cap to 1 (The default) and use the solve grow methods, which in effect find the right number of sticks. Alternativly if only given one parameter of the same type it acts as a copy constructor."""
    if isinstance(dims, DPGMM):
      self.dims = dims.dims
      self.stickCap = dims.stickCap

      self.data = map(lambda x: x.copy(), dims.data)

      self.prior = gcp.GaussianPrior(dims.prior)
      self.priorT = gcp.StudentT(dims.priorT) if dims.priorT!=None else None
      self.n = map(lambda x: gcp.GaussianPrior(x), dims.n)
      self.beta = dims.beta.copy()
      self.alpha = dims.alpha.copy()
      self.v = dims.v.copy()
      self.z = None if dims.z==None else dims.z.copy()

      self.skip = dims.skip
      self.epsilon = dims.epsilon

      self.nT = map(lambda x: None if x==None else gcp.StudentT(x), dims.nT)
      self.vExpLog = dims.vExpLog.copy()
      self.vExpNegLog = dims.vExpNegLog.copy()
    else:
      self.dims = dims
      self.stickCap = stickCap

      self.data = [] # A list containing data matrices - used to collate all the samples ready for processing. Before processing they are all appended into a single data matrix, such that this list is of length 1.

      self.prior = gcp.GaussianPrior(self.dims) # The prior over the mixture components.
      self.priorT = None
      self.n = map(lambda _: gcp.GaussianPrior(self.dims), xrange(self.stickCap)) # The mixture components, one for each stick.
      self.beta = numpy.ones(2, dtype=numpy.float32) # The two parameters (Typically named alpha & beta) for the Gamma distribution prior over alpha.
      self.alpha = numpy.ones(2, dtype=numpy.float32) # The parameters for the Gamma distribution that represents the current distribution over alpha - basically beta updated with the current stick configuration.
      self.v = numpy.ones((self.stickCap,2), dtype=numpy.float32) # Each [i,:] of this array represents the two parameters of the beta distribution over the strick breaking weight for the relevant mixture component.
      self.z = None # The matrix of multinomials over stick-assignment for each sample, aligned with the data matrix. In the case of incrimental use will not necessarily be complete.

      self.skip = 0 # Number of samples at the start of the data matrix to not bother updating - useful to speed things up with incrimental learning.
      self.epsilon = 1e-4 # Amount of change below which it stops iterating.

      # The cache of stuff kept around for speed...
      self.nT = [None]*self.stickCap # The student T distribution associated with each Gaussian.
      self.vExpLog = numpy.empty(self.stickCap, dtype=numpy.float32) # The expected value of the logorithm of each v.
      self.vExpNegLog = numpy.empty(self.stickCap, dtype=numpy.float32) # The expected value of the logarithm of 1 minus each v.

      self.vExpLog[:] = -1.0    # Need these to always be correct - this matches initialisation of v.
      self.vExpNegLog[:] = -1.0 # As above.

  def incStickCap(self, inc = 1):
    """Increases the stick cap by the given number of entrys. Can be used in collaboration with nllData to increase the number of sticks until insufficient improvement, indicating the right number has been found."""
    self.stickCap += inc
    self.n += map(lambda _: gcp.GaussianPrior(self.dims), xrange(inc))
    self.v = numpy.append(self.v,numpy.ones((inc,2), dtype=numpy.float32),0)
    
    if self.z!=None:
      self.z = numpy.append(self.z, numpy.random.mtrand.dirichlet(32.0*numpy.ones(inc), size=self.z.shape[0]), 1)
      weight = numpy.random.mtrand.dirichlet(numpy.ones(2), size=self.z.shape[0])
      self.z[:,:self.stickCap-inc] *= weight[:,0].reshape((self.z.shape[0],1))
      self.z[:,self.stickCap-inc:] *= weight[:,1].reshape((self.z.shape[0],1))
    
    self.nT += [None] * inc
    self.vExpLog = numpy.append(self.vExpLog,-1.0*numpy.ones(inc, dtype=numpy.float32))
    self.vExpNegLog = numpy.append(self.vExpNegLog,-1.0*numpy.ones(inc, dtype=numpy.float32))

  def getStickCap(self):
    """Returns the current stick cap."""
    return self.stickCap


  def setPrior(self, mean = None, covar = None, weight = None, scale = 1.0):
    """Sets a prior for the mixture components - basically a pass through for the addPrior method of the GaussianPrior class. If None (The default) is provided for the mean or the covar then it calculates these values for the currently contained sample set and uses them. Note that the prior defaults to nothing - this must be called before fitting the model, and if mean/covar are not provided then there must be enough data points to avoid problems. weight defaults to the number of dimensions if not specified. If covar is not given then scale is a multiplier for the covariance matrix - setting it high will soften the prior up and make it consider softer solutions when given less data. Returns True on success, False on failure - failure can happen if there is not enough data contained for automatic calculation (Think singular covariance matrix). This must be called before any solve methods are called."""
    # Handle mean/covar being None...
    if mean==None or covar==None:
      inc = gcp.GaussianInc(self.dims)
      dm = self.getDM()
      for i in xrange(dm.shape[0]): inc.add(dm[i,:])
      ggd = inc.fetch()
      if mean==None: mean = ggd.getMean()
      if covar==None: covar = ggd.getCovariance() * scale

    if numpy.linalg.det(covar)<1e-12: return False

    # Update the prior...
    self.prior.reset()
    self.prior.addPrior(mean, covar, weight)
    self.priorT = self.prior.intProb()

    return True


  def setConcGamma(self, alpha, beta):
    """Sets the parameters for the Gamma prior over the concentration. Note that whilst alpha and beta are used for the parameter names, in accordance with standard terminology for Gamma distributions, they are not related to the model variable names. Default values are (1,1). The concentration parameter controls how much information the model requires to justify using a stick, such that lower numbers result in fewer sticks, higher numbers in larger numbers of sticks. The concentration parameter is learnt from the data, under the Gamma distribution prior set with this method, but this prior can still have a strong effect. If your data is not producing as many clusters as you expect then adjust this parameter accordingly, (e.g. increase alpha or decrease beta, or both.), but don't go too high or it will start hallucinating patterns where none exist!"""
    self.beta[0] = alpha
    self.beta[1] = beta

  def setThreshold(self, epsilon):
    """Sets the threshold for parameter change below which it considers it to have converged, and stops iterating."""
    self.epsilon = epsilon

  def add(self, sample):
    """Adds either a single sample or several samples - either give a single sample as a 1D array or a 2D array as a data matrix, where each sample is [i,:]. (Sample = feature. I refer to them as samples as that more effectivly matches the concept of this modeling the probability distribution from which the features are drawn.)"""
    sample = numpy.asarray(sample, dtype=numpy.float32)
    if len(sample.shape)==1:
      self.data.append(numpy.reshape(sample, (1,self.dims)))
    else:
      assert(len(sample.shape)==2)
      assert(sampler.shape[1]==self.dims)
      self.data.append(sample)

  def getDM(self):
    """Returns a data matrix containing all the samples that have been added."""
    if len(self.data)==1: return self.data[0]
    if len(self.data)==0: return None

    self.data = [numpy.vstack(self.data)]
    return self.data[0]

  def size(self):
    """Returns the number of samples that have been added."""
    dm = self.getDM()
    if dm!=None: return dm.shape[0]
    else: return 0


  def lock(self, num=0):
    """Prevents the algorithm updating the component weighting for the first num samples in the database - potentially useful for incrimental learning if in a rush. If set to 0, the default, everything is updated."""
    self.skip = num

  def solve(self, iterCap=None):
    """Iterates updating the parameters until the model has converged. Note that the system is designed such that you can converge, add more samples, and converge again, i.e. incrimental learning. Alternativly you can converge, add more sticks, and then convegre again without issue, which makes finding the correct number of sticks computationally reasonable.. Returns the number of iterations required to acheive convergance. You can optionally provide a cap on the number of iterations it will perform."""

    # Deal with the z array being incomplete - enlarge/create as needed. Random initialisation is used...
    dm = self.getDM()
    if self.z==None or self.z.shape[0]<dm.shape[0]:
      newZ = numpy.empty((dm.shape[0],self.stickCap), dtype=numpy.float32)
      
      if self.z==None: offset = 0
      else:
        offset = self.z.shape[0]
        newZ[:offset,:] = self.z
      self.z = newZ

      self.z[offset:,:] = numpy.random.mtrand.dirichlet(32.0*numpy.ones(self.stickCap), size=self.z.shape[0]-offset) # 32 is to avoid extreme values, which can lock it in place, without the distribution being too flat as to cause problems converging.

    # Iterate until convergance...
    prev = self.z.copy()
    iters = 0
    while True:
      iters += 1

      # Update the concentration parameter...
      self.alpha[0] = self.beta[0] + self.stickCap
      self.alpha[1] = self.beta[1] - self.vExpNegLog.sum()

      # Record the expected values of a stick given the prior alone - needed to normalise the z values...
      expLogStick = -scipy.special.psi(1.0 + self.alpha[0]/self.alpha[1])
      expNegLogStick = expLogStick
      expLogStick += scipy.special.psi(1.0)
      expNegLogStick += scipy.special.psi(self.alpha[0]/self.alpha[1])

      # Update the stick breaking weights...
      self.v[:,0] = 1.0
      self.v[:,1] = self.alpha[0]/self.alpha[1]

      sums = self.z.sum(axis=0)
      self.v[:,0] += sums
      
      self.v[:,1] += self.z.shape[0]
      self.v[:,1] -= numpy.cumsum(sums)

      # Calculate the log expectations on the stick breaking weights...
      self.vExpLog[:] = -scipy.special.psi(self.v.sum(axis=1))
      self.vExpNegLog[:] = self.vExpLog

      self.vExpLog[:] += scipy.special.psi(self.v[:,0])
      self.vExpNegLog[:] += scipy.special.psi(self.v[:,1])

      # Update the Gaussian conjugate priors, extracting the student-t distributions as well...
      for k in xrange(self.stickCap):
        self.n[k].reset()
        self.n[k].addGP(self.prior)
        self.n[k].addSamples(dm, self.z[:,k])

        self.nT[k] = self.n[k].intProb()

      # Update the z values...
      prev[self.skip:,:] = self.z[self.skip:,:]
      
      vExpNegLogCum = self.vExpNegLog.cumsum()
      base = self.vExpLog.copy()
      base[1:] += vExpNegLogCum[:-1]
      self.z[self.skip:,:] = numpy.exp(base).reshape((1,self.stickCap))

      for k in xrange(self.stickCap):
        self.z[self.skip:,k] *= self.nT[k].batchProb(dm[self.skip:,:])

      norm = self.priorT.batchProb(dm[self.skip:,:])
      norm *= math.exp(expLogStick + vExpNegLogCum[-1]) / (1.0 - math.exp(expNegLogStick))

      self.z[self.skip:,:] /= (self.z[self.skip:,:].sum(axis=1) + norm).reshape((self.z.shape[0]-self.skip,1))

      # Check for convergance...
      change = numpy.abs(prev[self.skip:,:]-self.z[self.skip:,:]).sum(axis=1).max()
      if change<self.epsilon: break
      if iters==iterCap: break

    # Return the number of iterations that were required to acheive convergance...
    return iters

  def solveGrow(self, iterCap=None):
    """This method works by solving for the current stick cap, and then it keeps increasing the stick cap until there is no longer an improvement in the model quality. If using this method you should probably initialise with a stick cap of 1. By using this method the stick cap parameter is lost, and you no longer have to guess what a good value is."""
    it = 0
    prev = None
    while True:
      it += self.solve(iterCap)
      value = self.nllData()
      if prev==None or value<prev:
        prev = value
        self.incStickCap()
      else: return it


  def sampleMixture(self):
    """Once solve has been called and a distribution over models determined this allows you to draw a specific model. Returns a 2-tuple, where the first entry is an array of weights and the second entry a list of Gaussian distributions - they line up, to give a specific Gaussian mixture model. For density estimation the probability of a specific point is then the sum of each weight multiplied by the probability of it comming from the associated Gaussian. For clustering the probability of a specific point belonging to a cluster is the weight multiplied by the probability of it comming from a specific Gaussian, normalised for all clusters. Note that this includes an additional term to cover the infinite number of terms that follow, which is really an approximation, but tends to be such a small amount as to not matter. Be warned that if doing clustering a point could be assigned to this 'null cluster', indicating that the model thinks the point belongs to an unknown cluster (i.e. one that it doesn't have enough information, or possibly sticks, to instanciate.)."""
    weight = numpy.empty(self.stickCap+1, dtype=numpy.float32)
    stick = 1.0
    for i in xrange(self.stickCap):
      val = random.betavariate(self.v[i,0], self.v[i,1])
      weight[i] = stick * val
      stick *= 1.0 - val
    weight[-1] = stick

    gauss = map(lambda x: x.sample(), self.n)
    gauss.append(self.prior.sample())
    
    return (weight,gauss)

  def intMixture(self):
    """Returns the details needed to calculate the probability of a point given the model (density estimation), or its probability of belonging to each stick (clustering), but with the actual draw of a mixture model from the model integrated out. It is an apprximation, though not a bad one. Basically you get a 2-tuple - the first entry is an array of weights, the second a list of student-t distributions. The weights and distributions align, such that for density estimation the probability for a point is the sum over all entrys of the weight multiplied by the probability of the sample comming from the student-t distribution. The prob method of this class calculates the use of this for a sample directly. For clustering the probability of belonging to each cluster is calculated as the weight multiplied by the probability of comming from the associated student-t, noting that you will need to normalise. stickProb allows you to get this assesment directly. Do not edit the returned value; also, it will not persist if solve is called again. This must only be called after solve is called at least once. Note that an extra element is included to cover the remainder of the infinite number of elements - be warned that a sample could have the highest probability of belonging to this dummy element, indicating that it probably belongs to something for which there is not enough data to infer a reasonable model."""
    weights = numpy.empty(self.stickCap+1, dtype=numpy.float32)
    
    stick = 1.0
    for i in xrange(self.stickCap):
      ev = self.v[i,0] / self.v[i,:].sum()
      weights[i] = stick * ev
      stick *= 1.0 - ev
    weights[-1] = stick
      
    return (weights, self.nT + [self.priorT])
  
  def prob(self, x):
    """Given a sample this returns its probability, with the actual draw from the model integrated out. Must not be called until after solve has been called. This is the density estimate if using this model for density estimation. Will also accept a data matrix, in which case it will return a 1D array of probabilities aligning with the input data matrix."""
    x = numpy.asarray(x)
    if len(x.shape)==1:
      ret = 0.0
      stick = 1.0
      for i in xrange(self.stickCap):
        bp = self.nT[i].prob(x)
        ev = self.v[i,0] / self.v[i,:].sum()
        ret += bp * stick * ev
        stick *= 1.0 - ev
      bp = self.priorT.prob(x)
      ret += bp * stick
      return ret
    else:
      ret = numpy.zeros(x.shape[0])
      stick = 1.0
      for i in xrange(self.stickCap):
        bp = self.nT[i].batchProb(x)
        ev = self.v[i,0] / self.v[i,:].sum()
        ret += bp * stick * ev
        stick *= 1.0 - ev
      bp = self.priorT.batchProb(x)
      ret += bp * stick
      return ret
  
  def stickProb(self, x):
    """Given a sample this returns its probability of belonging to each of the components, as a 1D array, including a dummy element at the end to cover the infinite number of sticks not being explicitly modeled. This is the probability of belonging to each cluster if using the model for clustering. Must not be called until after solve has been called. Will also accept a data matrix, in which case it will return a matrix with a row for each vector in the input data matrix."""
    x = numpy.asarray(x)
    if len(x.shape)==1:
      ret = numpy.empty(self.stickCap+1, dtype=numpy.float32)
      stick = 1.0
      for i in xrange(self.stickCap):
        bp = self.nT[i].prob(x)
        ev = self.v[i,0] / self.v[i,:].sum()
        ret[i] = bp * stick * ev
        stick *= 1.0 - ev
      bp = self.priorT.prob(x)
      ret[self.stickCap] = bp * stick
      ret /= ret.sum()
      return ret
    else:
      ret = numpy.empty((x.shape[0],self.stickCap+1), dtype=numpy.float32)
      stick = 1.0
      for i in xrange(self.stickCap):
        bp = self.nT[i].batchProb(x)
        ev = self.v[i,0] / self.v[i,:].sum()
        ret[:,i] = bp * stick * ev
        stick *= 1.0 - ev
      bp = self.priorT.batchProb(x)
      ret[:,self.stickCap] = bp * stick
      ret /= ret.sum(axis=1).reshape((-1,1))
      return ret


  def reset(self, alphaParam = True, vParam = True, zParam = True):
    """Allows you to reset the parameters associated with each variable - good for doing a restart if your doing multiple restarts, or if you suspect it has got stuck in a local minima whilst doing incrimental stuff."""
    if alphaParam:
      self.alpha[:] = 1.0

    if vParam:
      self.v[:,:] = 1.0
      self.vExpLog[:] = -1.0
      self.vExpNegLog[:] = -1.0

    if zParam:
      self.z = None

  def nllData(self):
    """Returns the negative log likelihood of the data given the current distribution over models, with the model integrated out - good for comparing multiple restarts/different numbers of sticks to find which is the best."""
    dm = self.getDM()
    model = self.intMixture()

    probs = numpy.empty((dm.shape[0],model[0].shape[0]), dtype=numpy.float32)
    for i, st in enumerate(model[1]):
      probs[:,i] = st.batchLogProb(dm) + math.log(model[0][i])
    offsets = probs.max(axis=1)
    probs -= offsets.reshape((-1,1))

    ret = offsets.sum()
    ret += numpy.log(numpy.exp(probs).sum(axis=1)).sum()
    return -ret

  def multiSolve(self, runs, testIter=256):
    """Clones this object a number of times, given by the runs parameter, and then runs each for the testIters parameter number of iterations, to give them time to converge a bit. It then selects the one with the best nllData() score and runs that to convergance, before returning that specific clone. This is basically a simple way of avoiding getting stuck in a really bad local minima, though chances are you will end up in another one, just not a terrible one. Obviously testIter limits the effectivness of this, but as it tends to converge faster if your closer to the correct answer hopefully not by much. (To be honest I have not found this method to be of much use - in testing when this techneque converges to the wrong answer it does so consistantly, indicating that there is insufficient data regardless of initialisation.)"""
    best = None
    bestNLL = None
    for _ in xrange(runs):
      clone = DPGMM(self)
      clone.solve(testIter)
      score = clone.nllData()
      if bestNLL==None or score<bestNLL:
        best = clone
        bestNLL = score

    best.solve()
    return best

  def multiGrowSolve(self, runs):
    """This effectivly does multiple calls of growSolve, as indicated by runs, and returns a clone of this object that has converged to the best solution found. This is without a doubt the best solving techneque provided by this method, just remember to use the default stickCap of 1 when setting up the object. Also be warned - this can take an aweful long time to produce its awesomness. Can return self, if the starting number of sticks is the best (Note that self will always be converged after this method returns.)."""
    self.solve()
    best = self
    bestNLL = self.nllData()
    
    for _ in xrange(runs):
      current = self
      lastScore = None
      while True:
        current = DPGMM(current)
        current.incStickCap()
        current.solve()
        score = current.nllData()
        if score<bestNLL:
          best = current
          bestNLL = score
        if lastScore!=None and score>lastScore: break
        lastScore = score

    return best
