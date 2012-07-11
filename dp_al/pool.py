# Copyright 2011 Tom SF Haines

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.



import math
import random
import numpy
import collections

from p_cat.p_cat import ProbCat

from concentration_dp import ConcentrationDP



Entity = collections.namedtuple('Entity', ['sample', 'prob', 'ident'])



class Pool:
  """Represents a pool of entities that can be used for trainning with active learning. Simply contains the entities, their category probabilities and some arbitary identifier (For testing the identifier is often set to be the true category.). Provides active learning methods to extract the entities via various techneques based on the category probabilites. The category probabilites are a dictionary, indexed by category names, and includes 'None' as the probability of it being draw from the prior. Each term consists of P(data|category,model). The many select methods remove an item from the pool based on an active learning approach - the user is then responsible for querying the oracle for its category and updating the model accordingly. Before calling a select method you need to call update to update the probabilities associated with each entity, providing it with the current model, though you can batch things by calling update once before several select calls. The select methods return the named tuple Entity, which is (sample, prob, ident)."""
  def __init__(self):
    self.entities = [] # Each entity is a 5-list, where the first entry is the thing being stored, the second the associated category probabilities dictionary, and the third the identifier of the thing, which for testing is often the true category. These are basically Entity objects, but left editable as lists. The 4th item is then the state, used to optimise repeated calls to update, and the 5th is a list of probability dictionaries, for if the classifier supports that.
    
    self.prior = collections.defaultdict(lambda: 1.0)
    self.count = None
    self.conc = ConcentrationDP()

    self.cats = None
    

  def store(self, sample, ident=None):
    """Stores the provided sample into the pool, for later extraction. An arbitary identifier can optionally be provided for testing purposes. The probability distribution is left empty at this time - a call to update will fix that for all objects currently in the pool."""
    self.entities.append([sample, None, ident, dict(), None])


  def update(self, classifier, dp_ready = True, qbc = False):
    """This is given an object that impliments the ProbCat interface from the p_cat module - it then uses that object to update the probabilities for all entities in the pool. Assumes the sample provided to store can be passed into the getProb method of the classifier. dp_ready should be left True if one of the select methods that involves dp's is going to be called, so it can update the concentration. qbc needs to be set True if methods based on query by comittee are to be used."""
    for entity in self.entities:
      entity[1] = classifier.getDataProb(entity[0], entity[3])
      if classifier.listMode() and qbc:
        entity[4] = classifier.getDataProbList(entity[0], entity[3])

    self.count = dict(classifier.getCatCounts())

    if dp_ready: self.conc.update(len(self.count), sum(self.count.itervalues()))

    self.cats = classifier.getCatList()


  def empty(self):
    """For testing if the pool is empty."""
    return len(self.entities)==0

  def size(self):
    """Returns how many entities are currently stored."""
    return len(self.entities)

  def data(self):
    """Returns the Entity objects representing the current pool, as a list. Safe to edit."""
    return map(lambda r: Entity._make(r[:3]), self.entities)

  def getConcentration(self):
    """Pass through to get the DP concentration."""
    return self.conc.getConcentration()


  def setPrior(self, prior=None):
    """Sets the prior used to swap P(data|class) by some select methods - if not provided a uniform prior is used. Automatically normalised."""
    if prior!=None:
      self.prior = dict(prior)
      div = float(sum(self.prior.values()))
      for key in self.prior.iterkeys(): self.prior[key] /= div
    else:
      self.prior = collections.defaultdict(lambda: 1.0)
    

  def selectRandom(self):
    """Returns an Entity randomly - effectivly the dumbest possible algorithm, though it has a nasty habit of doing quite well."""
    pos = random.randrange(len(self.entities))

    ret = self.entities[pos]
    self.entities = self.entities[:pos] + self.entities[pos+1:]
    return Entity._make(ret[:3])

  def selectRandomIdent(self, ident):
    """Selects randomly from all entities in the pool with the given identifier. It is typically used when the identifiers are the true categories, to compare with algorithms that are not capable of making a first choice, where the authors of the test have fixed the first item to be drawn. Obviously this is cheating, but it is sometimes required to do a fair comparison."""
    selFrom = []
    for i,entity in enumerate(self.entities):
      if entity[2]==ident:
        selFrom.append(i)

    pos = random.randrange(len(selFrom))
    pos = selFrom[pos]

    ret = self.entities[pos]
    self.entities = self.entities[:pos] + self.entities[pos+1:]
    return Entity._make(ret[:3])


  def selectOutlier(self, beta = None):
    """Returns the least likelly member. You can also make it probalistic by providing a beta value - it then weights the samples by exp(-beta * outlier) for random selection."""
    if len(self.cats)==0: return self.selectRandom()
    
    prob = numpy.zeros(len(self.entities), dtype=numpy.float32)
    for i, entity in enumerate(self.entities):
      for cat, p in entity[1].iteritems():
        if cat!=None:
          prob[i] += p * self.prior[cat]

    if beta==None:
      pos = numpy.argmin(prob)
    else:
      prob *= -beta
      prob = numpy.exp(prob)

      r = random.random() * prob.sum()
      pos = 0
      while pos<(prob.shape[0]-1):
        r -= prob[pos]
        if r<0.0: break
        pos += 1

    ret = self.entities[pos]
    self.entities = self.entities[:pos] + self.entities[pos+1:]
    return Entity._make(ret[:3])

  def selectEntropy(self, beta = None):
    """Selects the sample with the greatest entropy - the most common uncertainty-based sampling method. If beta is provided instead of selecting the maximum it makes a random selection by weighting each sample by exp(-beta * entropy)."""
    if len(self.cats)==0: return self.selectRandom()
    
    ent = numpy.empty(len(self.entities), dtype=numpy.float32)
    for i, entity in enumerate(self.entities):
      vals = []
      for cat, p in entity[1].iteritems():
        if cat!=None:
          pp = p * self.prior[cat]
          if pp>1e-6: vals.append(pp)
      div = sum(vals)
      ent[i] = -sum(map(lambda pp: (pp/div) * math.log(pp/div), vals))

    if beta==None:
      pos = numpy.argmax(ent)
    else:
      ent *= -beta
      ent = numpy.exp(ent)

      r = random.random() * ent.sum()
      pos = 0
      while pos<(ent.shape[0]-1):
        r -= ent[pos]
        if r<0.0: break
        pos += 1

    ret = self.entities[pos]
    self.entities = self.entities[:pos] + self.entities[pos+1:]
    return Entity._make(ret[:3])


  def selectDP(self, hardChoice = False):
    """Selects the entity, that, according to the DP assumption, is most likelly to be an instance of a new class. Can be made to select randomly, using the probabilities as weights, or to simply select the entry with the highest probability of being new."""

    # Calculate the P(new) probabilities...
    prob = numpy.empty(len(self.entities))
    for i, entity in enumerate(self.entities):
      new = entity[1][None] * self.conc.getConcentration()
      div = new
      for cat, p in entity[1].iteritems():
        if cat!=None: div += p * self.count[cat]
      prob[i] = new / div

    # Select an entry...
    if hardChoice: pos = numpy.argmax(prob)
    else:
      r = random.random() * prob.sum()
      pos = 0
      while pos<(prob.shape[0]-1):
        r -= prob[pos]
        if r<0.0: break
        pos += 1

    # Remove it from the pool, package it up and return it...
    ret = self.entities[pos]
    self.entities = self.entities[:pos] + self.entities[pos+1:]
    return Entity._make(ret[:3])


  def selectWrong(self, softSelect = False, hardChoice = False, dp = True, dw = False):
    """24 different selection strategies, all rolled into one. Bite me! All work on the basis of selecting the entity in the pool with the greatest chance of being misclassified by the current classifier. There are four binary flags that control the behaviour, and their defaults match up with the algorithm presented in the paper 'Active Learning using Dirichlet Processes for Rare Class Discovery and Classification'. softSelect indicates if the classifier selects the category with the highest probability (False) or selects the category probalistically from P(class|data) (True). hardChoice comes into play once P(wrong) has been calculated for each entity in the pool - when True the entity with the highest P(wrong) is selected, otherwise the P(wrong) are used as weights for a probabilistic selection. dp indicates if the Dirichlet process assumption is to be used, such that we consider the probability that the entity belongs to a new category in addition to the existing categories. Note that the classifier cannot select an unknown class, so an entity with a high probability of belonging to a new class has a high P(wrong) score when the dp assumption is True. dw indicates if it should weight the metric by a density estimate over the data set, and hence bias selection towards areas with lots of samples. Appendum: Also supports expected hinge loss, if you set softSelect to None (False is equivalent to expected 0-1 loss, True to something without a name.)."""
    if len(self.cats)==0 and dp==False: return self.selectRandom()
    
    wrong = numpy.ones(len(self.entities))
    for i, entity in enumerate(self.entities):
      # Calculate the probability of selecting each of the known classes...
      probSel = dict()
      div = 0.0
      for cat, p in entity[1].iteritems():
        if cat!=None:
          pp = p * self.prior[cat]
          probSel[cat] = pp
          div += pp

      if div<1e-5:
        for cat in probSel.iterkeys():
          probSel[cat] = self.prior[cat]
          div += probSel[cat]
      
      for cat in probSel.iterkeys(): probSel[cat] /= div

      # Calculate the probability of it being each of the options...
      probIs = dict()
      div = 0.0
      for cat, p in entity[1].iteritems():
        if cat!=None or dp:
          probIs[cat] = p * (self.count[cat] if cat!=None else self.conc.getConcentration())
          div += probIs[cat]
          
      if div<1e-5:
        for cat in probIs.iterkeys():
          probIs[cat] = self.prior[cat]
          div += probIs[cat]

      for cat in probIs.iterkeys(): probIs[cat] /= div

      # Calculate the probability of getting it wrong...
      if softSelect==None: # 1 - Expected hinge loss, sort of.
        if len(probSel)>0: maxSel = max(probSel.itervalues())
        else: maxSel = 1.0
        wrong[i] -= maxSel * probIs[None]
        for cat, p in probSel.iteritems():
          wrong[i] -= (maxSel - p) * probIs[cat]
      elif softSelect:
        for cat, p in probSel.iteritems():
          wrong[i] -= p * probIs[cat]
      else:
        best = -1.0
        for cat, p in probSel.iteritems():
          if p>best:
            best = p
            wrong[i] = 1.0 - probIs[cat]
      
      # If requested include a weighting by density...
      if dw: wrong[i] *= entity[1][None]

    if hardChoice:
      pos = numpy.argmax(wrong)
    else:
      r = random.random() * wrong.sum()
      pos = 0
      while pos<(wrong.shape[0]-1):
        r -= wrong[pos]
        if r<0.0: break
        pos += 1

    ret = self.entities[pos]
    self.entities = self.entities[:pos] + self.entities[pos+1:]
    return Entity._make(ret[:3])
  
  
  def selectWrongQBC(self, softSelect = False, hardChoice = False, dp = True, dw = False):
    """A query by comittee version of selectWrong - its parameters are equivalent. Requires that update is called with qbc set to True."""
    if len(self.cats)==0 and dp==False: return self.selectRandom()
    
    wrong = numpy.zeros(len(self.entities))
    for i, entity in enumerate(self.entities):
      # Calculate a list of estimates of the probability of selecting each of the known classes...
      probSelList = []
      for prob in entity[4]:
        probSel = dict()
        div = 0.0
        for cat, p in prob.iteritems():
          if cat!=None:
            pp = p * self.prior[cat]
            probSel[cat] = pp
            div += pp

        if div<1e-5:
          for cat in probSel.iterkeys():
            probSel[cat] = self.prior[cat]
            div += probSel[cat]
      
        for cat in probSel.iterkeys(): probSel[cat] /= div
        
        probSelList.append(probSel)
        
      # Calculate a list of estimates of the probability of it being each of the options...
      probIsList = []
      for prob in entity[4]:
        probIs = dict()
        div = 0.0
        for cat, p in prob.iteritems():
          if cat!=None or dp:
            probIs[cat] = p * (self.count[cat] if cat!=None else self.conc.getConcentration())
            div += probIs[cat]

        if div<1e-5:
          for cat in probIs.iterkeys():
            probIs[cat] = self.prior[cat]
            div += probIs[cat]

        for cat in probIs.iterkeys(): probIs[cat] /= div
        
        probIsList.append(probIs)
      
      # Now do the combinatorics of the two lists to generate a P(wrong) estimate for each pair, for which the average is taken...
      for probSel in probSelList:
        for probIs in probIsList:
          if softSelect==None: # 1 - Expected hinge loss, sort of.
            if len(probSel)>0: maxSel = max(probSel.itervalues())
            else: maxSel = 1.0
            w = 1.0 - maxSel * probIs[None]
            for cat, p in probSel.iteritems():
              w -= (maxSel - p) * probIs[cat]
          elif softSelect:
            w = 1.0
            for cat, p in probSel.iteritems():
              w -= p * probIs[cat]
          else:
            best = -1.0
            w = 0.0
            for cat, p in probSel.iteritems():
              if p>best:
                best = p
                w = 1.0 - probIs[cat]
          wrong[i] += w
      wrong[i] /= len(probSelList) * len(probIsList)

      # If requested include a weighting by density...
      if dw: wrong[i] *= entity[1][None]
      
    if hardChoice:
      pos = numpy.argmax(wrong)
    else:
      r = random.random() * wrong.sum()
      pos = 0
      while pos<(wrong.shape[0]-1):
        r -= wrong[pos]
        if r<0.0: break
        pos += 1

    ret = self.entities[pos]
    self.entities = self.entities[:pos] + self.entities[pos+1:]
    return Entity._make(ret[:3])


  @staticmethod
  def methods(incQBC = False):
    """Returns a list of the method names that can be passed to the select method. Read the select method to work out which they each are. p_wrong_soft is the published techneque. By default it does not include the query by comittee versions, which can be switched on using the relevent flag."""
    return ['random', 'outlier', 'entropy', 'p_new_hard', 'p_new_soft', 'p_wrong_hard', 'p_wrong_soft', 'p_wrong_hard_pcat', 'p_wrong_soft_pcat', 'p_wrong_hard_naive', 'p_wrong_soft_naive', 'p_wrong_hard_pcat_naive', 'p_wrong_soft_pcat_naive', 'dxp_wrong_hard', 'dxp_wrong_soft', 'dxp_wrong_hard_pcat', 'dxp_wrong_soft_pcat', 'dxp_wrong_hard_naive', 'dxp_wrong_soft_naive', 'dxp_wrong_hard_pcat_naive', 'dxp_wrong_soft_pcat_naive', 'p_wrong_hard_hinge', 'p_wrong_soft_hinge', 'p_wrong_hard_hinge_naive', 'p_wrong_soft_hinge_naive', 'dxp_wrong_hard_hinge', 'dxp_wrong_soft_hinge', 'dxp_wrong_hard_hinge_naive', 'dxp_wrong_soft_hinge_naive'] + ([] if incQBC==False else ['qbc_p_wrong_hard', 'qbc_p_wrong_soft', 'qbc_p_wrong_hard_pcat', 'qbc_p_wrong_soft_pcat', 'qbc_p_wrong_hard_naive', 'qbc_p_wrong_soft_naive', 'qbc_p_wrong_hard_pcat_naive', 'qbc_p_wrong_soft_pcat_naive', 'qbc_dxp_wrong_hard', 'qbc_dxp_wrong_soft', 'qbc_dxp_wrong_hard_pcat', 'qbc_dxp_wrong_soft_pcat', 'qbc_dxp_wrong_hard_naive', 'qbc_dxp_wrong_soft_naive', 'qbc_dxp_wrong_hard_pcat_naive', 'qbc_dxp_wrong_soft_pcat_naive', 'qbc_p_wrong_hard_hinge', 'qbc_p_wrong_soft_hinge', 'qbc_p_wrong_hard_hinge_naive', 'qbc_p_wrong_soft_hinge_naive', 'qbc_dxp_wrong_hard_hinge', 'qbc_dxp_wrong_soft_hinge', 'qbc_dxp_wrong_hard_hinge_naive', 'qbc_dxp_wrong_soft_hinge_naive'])

  def select(self, method):
    """Pass through for all of the select methods that have no problamatic parameters - allows you to select the method using a string. You can get a list of all method strings from the methods() method."""
    if method=='random': return self.selectRandom()
    elif method=='outlier': return self.selectOutlier()
    elif method=='entropy': return self.selectEntropy()
    elif method=='p_new_hard': return self.selectDP(True)
    elif method=='p_new_soft': return self.selectDP(False)
    elif method=='p_wrong_hard': return self.selectWrong(False,True,True,False)
    elif method=='p_wrong_soft': return self.selectWrong(False,False,True,False)
    elif method=='p_wrong_hard_pcat': return self.selectWrong(True,True,True,False)
    elif method=='p_wrong_soft_pcat': return self.selectWrong(True,False,True,False)
    elif method=='p_wrong_hard_naive': return self.selectWrong(False,True,False,False)
    elif method=='p_wrong_soft_naive': return self.selectWrong(False,False,False,False)
    elif method=='p_wrong_hard_pcat_naive': return self.selectWrong(True,True,False,False)
    elif method=='p_wrong_soft_pcat_naive': return self.selectWrong(True,False,False,False)
    elif method=='dxp_wrong_hard': return self.selectWrong(False,True,True,True)
    elif method=='dxp_wrong_soft': return self.selectWrong(False,False,True,True)
    elif method=='dxp_wrong_hard_pcat': return self.selectWrong(True,True,True,True)
    elif method=='dxp_wrong_soft_pcat': return self.selectWrong(True,False,True,True)
    elif method=='dxp_wrong_hard_naive': return self.selectWrong(False,True,False,True)
    elif method=='dxp_wrong_soft_naive': return self.selectWrong(False,False,False,True)
    elif method=='dxp_wrong_hard_pcat_naive': return self.selectWrong(True,True,False,True)
    elif method=='dxp_wrong_soft_pcat_naive': return self.selectWrong(True,False,False,True)
    elif method=='p_wrong_hard_hinge': return self.selectWrong(None,True,True,False)
    elif method=='p_wrong_soft_hinge': return self.selectWrong(None,False,True,False)
    elif method=='p_wrong_hard_hinge_naive': return self.selectWrong(None,True,False,False)
    elif method=='p_wrong_soft_hinge_naive': return self.selectWrong(None,False,False,False)
    elif method=='dxp_wrong_hard_hinge': return self.selectWrong(None,True,True,True)
    elif method=='dxp_wrong_soft_hinge': return self.selectWrong(None,False,True,True)
    elif method=='dxp_wrong_hard_hinge_naive': return self.selectWrong(None,True,False,True)
    elif method=='dxp_wrong_soft_hinge_naive': return self.selectWrong(None,False,False,True)
    elif method=='qbc_p_wrong_hard': return self.selectWrongQBC(False,True,True,False)
    elif method=='qbc_p_wrong_soft': return self.selectWrongQBC(False,False,True,False)
    elif method=='qbc_p_wrong_hard_pcat': return self.selectWrongQBC(True,True,True,False)
    elif method=='qbc_p_wrong_soft_pcat': return self.selectWrongQBC(True,False,True,False)
    elif method=='qbc_p_wrong_hard_naive': return self.selectWrongQBC(False,True,False,False)
    elif method=='qbc_p_wrong_soft_naive': return self.selectWrongQBC(False,False,False,False)
    elif method=='qbc_p_wrong_hard_pcat_naive': return self.selectWrongQBC(True,True,False,False)
    elif method=='qbc_p_wrong_soft_pcat_naive': return self.selectWrongQBC(True,False,False,False)
    elif method=='qbc_dxp_wrong_hard': return self.selectWrongQBC(False,True,True,True)
    elif method=='qbc_dxp_wrong_soft': return self.selectWrongQBC(False,False,True,True)
    elif method=='qbc_dxp_wrong_hard_pcat': return self.selectWrongQBC(True,True,True,True)
    elif method=='qbc_dxp_wrong_soft_pcat': return self.selectWrongQBC(True,False,True,True)
    elif method=='qbc_dxp_wrong_hard_naive': return self.selectWrongQBC(False,True,False,True)
    elif method=='qbc_dxp_wrong_soft_naive': return self.selectWrongQBC(False,False,False,True)
    elif method=='qbc_dxp_wrong_hard_pcat_naive': return self.selectWrongQBC(True,True,False,True)
    elif method=='qbc_dxp_wrong_soft_pcat_naive': return self.selectWrongQBC(True,False,False,True)
    elif method=='qbc_p_wrong_hard_hinge': return self.selectWrongQBC(None,True,True,False)
    elif method=='qbc_p_wrong_soft_hinge': return self.selectWrongQBC(None,False,True,False)
    elif method=='qbc_p_wrong_hard_hinge_naive': return self.selectWrongQBC(None,True,False,False)
    elif method=='qbc_p_wrong_soft_hinge_naive': return self.selectWrongQBC(None,False,False,False)
    elif method=='qbc_dxp_wrong_hard_hinge': return self.selectWrongQBC(None,True,True,True)
    elif method=='qbc_dxp_wrong_soft_hinge': return self.selectWrongQBC(None,False,True,True)
    elif method=='qbc_dxp_wrong_hard_hinge_naive': return self.selectWrongQBC(None,True,False,True)
    elif method=='qbc_dxp_wrong_soft_hinge_naive': return self.selectWrongQBC(None,False,False,True)
    else: raise Exception('Unknown selection method')
