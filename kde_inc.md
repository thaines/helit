# Incrimental kernel density estimation #

## Overview ##
**Incremental Kernel Density Estimation**

An implementation of the method from the paper 'Incremental One-Class learning with Bounded Computational Complexity', by R. R. Sillito and R. B. Fisher.

`loo_cov.py` - Code to calculate an optimal covariance matrix from a bunch of samples using leave one out.

`gmm.py` - Gaussian mixture model representation, used internally.

`kde_inc.py` - Wrapper for GMM that adds the ability to update the model with new samples as they come in, using Fishers incremental technique.


`test_1d_beta.py` - Simple test of KDE\_INC class.

`test_loo.py` - Simple test of the loo class.


`make_doc.py` - Generates the help file.

`readme.txt` - This file, which gets copied into the html documentation.


---


# Classes #

## PrecisionLOO() ##
> Given a large number of samples this uses leave one out to calculate the optimal symetric precision matrix. Standard griding solution.

**`__init__(self)`**
> Initilises with no samples but a default grid of 10^-2 to 10 in 128 incriments that are linear in log base 10 space.

**`addSample(self, sample)`**
> Adds one or more samples to the set used for loo optimisation. Can either be a single vector or a data matrix, where the first dimension indexes the individual samples.

**`calcVar(self, var, subset)`**
> Internal method really - given a variance calculates its leave one out nll. Has an optional subset parameter, which indexes a subset of data point to be used from the data matrix.

**`dataMatrix(self)`**
> More for internal use - collates all the samples into a single data matrix, which is put in the internal samples array such that it does not break things - the data matrix is then returned.

**`getBest(self)`**
> Returns the best precision matrix.

**`setLogGrid(self, low = -4.0, high = 1.0, step = 128)`**
> Sets the grid of variances to test to contain values going from 10<sup>low to 10</sup>high, with inclusive linear interpolation of the exponents to obtain step values.

**`solve(self, callback)`**
> Trys all the options, and selects the one that provides the best nll.

## SubsetPrecisionLOO(PrecisionLOO) ##
> This class performs the same task as PrecisionLOO, except it runs on a subset of data points, and in effect tunes the precision matrix for a kernel density estimate constructed using less samples than are provided to the class. Takes the mean of multiple runs with different subsets.

**`__init__(self)`**
> Initilises with no samples but a default grid of 10^-2 to 10 in 128 incriments that are linear in log base 10 space.

**`addSample(self, sample)`**
> Adds one or more samples to the set used for loo optimisation. Can either be a single vector or a data matrix, where the first dimension indexes the individual samples.

**`calcVar(self, var, subset)`**
> Internal method really - given a variance calculates its leave one out nll. Has an optional subset parameter, which indexes a subset of data point to be used from the data matrix.

**`dataMatrix(self)`**
> More for internal use - collates all the samples into a single data matrix, which is put in the internal samples array such that it does not break things - the data matrix is then returned.

**`getBest(self)`**
> Returns the best precision matrix.

**`setLogGrid(self, low = -4.0, high = 1.0, step = 128)`**
> Sets the grid of variances to test to contain values going from 10<sup>low to 10</sup>high, with inclusive linear interpolation of the exponents to obtain step values.

**`solve(self, runs, size, callback)`**
> Trys all the options, and selects the one that provides the best nll. runs is the number of runs to do, with it taking the average score for each run, whilst size is how many samples to have in each run, i.e. the size to tune for.

## GMM() ##
> Contains a Gaussian mixture model - just a list of weights, means and precision matrices. List is of fixed size, and it has functions to determine the probability of a point in space. Components with a weight of zero are often computationally ignored. Initialises empty, which is not good for normalisation of weights - don't do it until data is avaliable! Designed to be used directly by any entity that is filling it in - interface is mostly user only.

**`__init__(self, dims, count)`**
> dims is the dimension of the mixture model, count the number of mixture components it will consider using.

**`calcNorm(self, i)`**
> Sets the normalising constant for a specific entry.

**`calcNorms(self)`**
> Fills in the normalising constants for all components with weight.

**`clone(self)`**
> Returns a clone of this object.

**`marginalise(self, dims)`**
> Given a list of dimensions this keeps those dimensions and drops the rest, i.e. marginalises them out. New version of this object will have the old indices remapped as indicated by dims.

**`nll(self, sample)`**
> Given a sample vector, as something that numpy.asarray can interpret, return the negative log liklihood of the sample. All values must be correct for this to work. Has inline C, but if that isn't working the implimentation is fully vectorised, so should be quite fast despite being in python.

**`normWeights(self)`**
> Scales the weights so they sum to one, as is required for correctness.

**`prob(self, sample)`**
> Given a sample vector, as something that numpy.asarray can interpret, return the normalised probability of the sample. All values must be correct for this to work. Has inline C, but if that isn't working the implimentation is fully vectorised, so should be quite fast despite being in python.

## KDE\_INC() ##
> Provides an incrimental kernel density estimate system that uses Gaussians. A kernel density estimate system with Gaussian kernels that, on reaching a cap, starts merging kernels to limit the number of kernels to a constant - done in such a way as to minimise error whilst capping computation. (Computation is quite high however - this is not a very efficient implimentation.)

**`__init__(self, prec, cap = 32)`**
> Initialise with the precision matrix to use for the kernels, which implicitly provides the number of dimensions, and the cap on the number of kernels to allow.

**`add(self, sample)`**
> Adds a sample, updating the kde accordingly.

**`marginalise(self, dims)`**
> Returns an object on which you can call prob(), but with only a subset of the dimensions. The set of dimensions is given as something that can be interpreted as a numpy array of integers - it is the dimensions to keep, it marginalises away everything else. The indexing of the returned object will match up with that in dims. Note that you must not have any repetitions in dims - that would create absurdity.

**`nll(self, sample)`**
> Returns the negative log liklihood of the given sample - must not be called until at least one sample has been added, though it will return a positive constant if called with no samples provided.

**`prob(self, sample)`**
> Returns the probability of the given sample - must not be called until at least one sample has been added, though it will return a positive constant if called with no samples provided.

**`samples(self)`**
> Returns how many samples have been added to the object.

**`setPrec(self, prec)`**
> Changes the precision matrix - must be called before any samples are added, and must have the same dimensions as the current one.