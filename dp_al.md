# Dirichlet Process Active Learning #

## Overview ##
**Dirichlet process active learning**

Implements the active learning algorithm from the paper
'Active Learning using Dirichlet Processes for Rare Class Discovery and Classification' by T. S. F. Haines and T. Xiang
and also a bunch of other active learning algorithms, most of which are variants.

Contains two classes - one is just a helper for estimating the concentration parameter of a Dirichlet process, the other represents a pool of entities for an active learner to select from. The pool provides many active learning methods, and is designed to interface tightly with a classifier from the p\_cat module. Usage is a bit on the non-obvious side - best to look at the test code to see how to use it.

Files included:

`dp_al.py` - File that imports all parts of the system, to be imported by users.


`concentration_dp.py` - Contains ConcentrationDP, a class to assist with estimating the concentration parameter of a Dirichlet process.

`pool.py` - Contains Pool, which you fill with entities to learn from. It then provides various active learning algorithms to select which entity to give to the oracle next. Note that the user is responsible for updating the classifier and interfacing with the oracle.



`test_iris.py` - Simple visualisation of the P(wrong) algorithm.


`test_synth.py` - Simple synthetic comparison between all of the active learning algorithms. Potentially a bit misleading due to its synthetic nature.



`readme.txt` - This file, which is copied into the html documentation.


`make_doc.py` - Generates the html documentation.


---


# Classes #

## Pool() ##
> Represents a pool of entities that can be used for trainning with active learning. Simply contains the entities, their category probabilities and some arbitary identifier (For testing the identifier is often set to be the true category.). Provides active learning methods to extract the entities via various techneques based on the category probabilites. The category probabilites are a dictionary, indexed by category names, and includes 'None' as the probability of it being draw from the prior. Each term consists of P(data|category,model). The many select methods remove an item from the pool based on an active learning approach - the user is then responsible for querying the oracle for its category and updating the model accordingly. Before calling a select method you need to call update to update the probabilities associated with each entity, providing it with the current model, though you can batch things by calling update once before several select calls. The select methods return the named tuple Entity, which is (sample, prob, ident).

**`__init__(self)`**
> None

**`data(self)`**
> Returns the Entity objects representing the current pool, as a list. Safe to edit.

**`empty(self)`**
> For testing if the pool is empty.

**`getConcentration(self)`**
> Pass through to get the DP concentration.

**`select(self, method, sd)`**
> Pass through for all of the select methods that have no problamatic parameters - allows you to select the method using a string. You can get a list of all method strings from the methods() method. Actually, allows you to provide a sd parameter for the P(wrong) methods that support it.

**`selectDP(self, hardChoice = False)`**
> Selects the entity, that, according to the DP assumption, is most likelly to be an instance of a new class. Can be made to select randomly, using the probabilities as weights, or to simply select the entry with the highest probability of being new.

**`selectEntropy(self, beta)`**
> Selects the sample with the greatest entropy - the most common uncertainty-based sampling method. If beta is provided instead of selecting the maximum it makes a random selection by weighting each sample by exp(-beta **entropy).**

**`selectOutlier(self, beta)`**
> Returns the least likelly member. You can also make it probalistic by providing a beta value - it then weights the samples by exp(-beta **outlier) for random selection.**

**`selectRandom(self)`**
> Returns an Entity randomly - effectivly the dumbest possible algorithm, though it has a nasty habit of doing quite well.

**`selectRandomIdent(self, ident)`**
> Selects randomly from all entities in the pool with the given identifier. It is typically used when the identifiers are the true categories, to compare with algorithms that are not capable of making a first choice, where the authors of the test have fixed the first item to be drawn. Obviously this is cheating, but it is sometimes required to do a fair comparison.

**`selectWrong(self, softSelect = False, hardChoice = False, dp = True, dw = False, sd)`**
> 24 different selection strategies, all rolled into one. Bite me! All work on the basis of selecting the entity in the pool with the greatest chance of being misclassified by the current classifier. There are four binary flags that control the behaviour, and their defaults match up with the algorithm presented in the paper 'Active Learning using Dirichlet Processes for Rare Class Discovery and Classification'. softSelect indicates if the classifier selects the category with the highest probability (False) or selects the category probalistically from P(class|data) (True). hardChoice comes into play once P(wrong) has been calculated for each entity in the pool - when True the entity with the highest P(wrong) is selected, otherwise the P(wrong) are used as weights for a probabilistic selection. dp indicates if the Dirichlet process assumption is to be used, such that we consider the probability that the entity belongs to a new category in addition to the existing categories. Note that the classifier cannot select an unknown class, so an entity with a high probability of belonging to a new class has a high P(wrong) score when the dp assumption is True. dw indicates if it should weight the metric by a density estimate over the data set, and hence bias selection towards areas with lots of samples. Appendum: Also supports expected hinge loss, if you set softSelect to None (False is equivalent to expected 0-1 loss, True to something without a name.). If sd is not None then the wrong score for each entity is boosted by neighbours, on the grounds that knowing about an entity will affect its neighbours classification - its uses the unnormalised weighting of a Gaussian (The centre carries a weight of 1.) with the given sd.

**`selectWrongQBC(self, softSelect = False, hardChoice = False, dp = True, dw = False)`**
> A query by comittee version of selectWrong - its parameters are equivalent. Requires that update is called with qbc set to True.

**`setPrior(self, prior)`**
> Sets the prior used to swap P(data|class) by some select methods - if not provided a uniform prior is used. Automatically normalised.

**`size(self)`**
> Returns how many entities are currently stored.

**`store(self, sample, ident)`**
> Stores the provided sample into the pool, for later extraction. An arbitary identifier can optionally be provided for testing purposes. The probability distribution is left empty at this time - a call to update will fix that for all objects currently in the pool.

**`update(self, classifier, dp_ready = True, qbc = False)`**
> This is given an object that impliments the ProbCat interface from the p\_cat module - it then uses that object to update the probabilities for all entities in the pool. Assumes the sample provided to store can be passed into the getProb method of the classifier. dp\_ready should be left True if one of the select methods that involves dp's is going to be called, so it can update the concentration. qbc needs to be set True if methods based on query by comittee are to be used.

## Entity(tuple, object) ##
> Entity(sample, nll, ident)

**`__getnewargs__(self)`**
> Return self as a plain tuple.  Used by copy and pickle.

**`__repr__(self)`**
> Return a nicely formatted representation string

**`_asdict(self)`**
> Return a new OrderedDict which maps field names to their values

**`_make(cls, iterable, new = <built-in method __new__ of type object at 0x88dd80>, len = <built-in function len>)`**
> Make a new Entity object from a sequence or iterable

**`_replace(_self, **kwds)`**
> Return a new Entity object replacing specified fields with new values

## ConcentrationDP() ##
> Represents the concentration parameter of a Dirichlet process - contains the parameters of its prior and updates its estimate as things change. The estimate is actually the mean of many Gibbs samples of the parameter.

**`__init__(self)`**
> Initialises with both parameters of the prior set to 1 - i.e. both alpha and beta of the gamma distribution.

**`getConcentration(self)`**
> Returns the most recent estimate of the concentration.

**`setParms(self, burnIn, samples)`**
> Sets the Gibbs sampling parameters for updating the estimate. They both default to 128.

**`setPrior(self, alpha, beta)`**
> Sets the alpha and beta parameters of the concentrations gamma prior. They both default to 1.

**`update(self, k, n)`**
> Given k, the number of dp instances, and n, the number of samples drawn from the dp, updates and returns the concentration parameter.