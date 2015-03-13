# Gaussian Conjugate Prior #

## Overview ##
**Gaussian Conjugate Prior**

Basic library for handling the conjugate prior to the Gaussian. Contains Gaussian, Wishart and Student-T distributions, alongside a class to represent the conjugate prior, with all the relevant operations. Also includes an incremental Gaussian calculator. Lacking optimisation this is more for demonstration/education or for when speed just does not matter (Or there are lots of dimensions, so the python costs become insignificant compared to the actual operations, which are all done by scipy.) - implemented using pure python with scipy. The test scripts provide good examples of use.


`gcp.py` - Helper file that pulls all the parts of the system into a single namespace.


`gaussian.py` - Contains the Gaussian distributions class.

`gaussian_inc.py` - Contains GaussianInc, an incremental class for which you provide samples, from which at any point a Gaussian object can be extracted.

`wishart.py` - The Wishart distribution.

`student_t.py` - The student T distribution, provided as it is the distribution obtained when you integrate out a draw from the Gaussians conjugate prior.

`gaussian_prior.py` - The point of the entire system - provides a conjugate prior to the Gaussian, over both mean and covariance, with lots of methods to update it, draw from it or integrate out draws from it etc.


`test_1d.py` - Simple 1D test of the system. Uses open cv for its output.

`test_2d.py` - Simple 2D test of the system. Uses open cv for its output.

`test_inc.py` - Tests that incrementally adding and removing samples works.

`test_weighted.py` - Tests that it can handle weighted samples.


`readme.txt` - This file, which is copied into the automatically generated documentation.

`make_doc.py` - creates the gcp.html help file.


---


# Classes #

## Gaussian() ##
> A basic multivariate Gaussian class. Has caching to avoid duplicate calculation.

**`__init__(self, dims)`**
> dims is the number of dimensions. Initialises with mu at the origin and the identity matrix for the precision/covariance. dims can also be another Gaussian object, in which case it acts as a copy constructor.

**`__str__(self)`**
> None

**`getCovariance(self)`**
> Returns the covariance matrix.

**`getMean(self)`**
> Returns the mean.

**`getNorm(self)`**
> Returns the normalising constant of the distribution. Typically for internal use only.

**`getPrecision(self)`**
> Returns the precision matrix.

**`prob(self, x)`**
> Given a vector x evaluates the probability density function at that point.

**`sample(self)`**
> Draws and returns a sample from the distribution.

**`setCovariance(self, covariance)`**
> Sets the covariance matrix. Alternativly you can use the setPrecision method.

**`setMean(self, mean)`**
> Sets the mean - you can use anything numpy will interprete as a 1D array of the correct length.

**`setPrecision(self, precision)`**
> Sets the precision matrix. Alternativly you can use the setCovariance method.

## GaussianInc() ##
> Allows you to incrimentally calculate a Gaussian distribution by providing lots of samples.

**`__init__(self, dims)`**
> You provide the number of dimensions - you must add at least dims samples before there is the possibility of extracting a gaussian from this. Can also act as a copy constructor.

**`add(self, sample, weight = 1.0)`**
> Updates the state given a new sample - sample can have a weight, which obviously defaults to 1, but can be set to other values to indicate repetition of a single point, including fractional.

**`fetch(self)`**
> Returns the Gaussian distribution calculated so far.

**`makeSafe(self)`**
> Bodges the internal representation so it can provide a non-singular covariance matrix - obviously a total hack, but potentially useful when insufficient information exists. Works by taking the svd, nudging zero entrys away from 0 in the diagonal matrix, then multiplying the terms back together again. End result is arbitary, but won't be inconsistant with the data provided.

**`safe(self)`**
> Returns True if it has enough data to provide an actual Gaussian, False if it does not.

## Wishart() ##
> Simple Wishart distribution class, quite basic really, but has caching to avoid duplicate computation.

**`__init__(self, dims)`**
> dims is the number of dimensions - it initialises with the dof set to 1 and the scale set to the identity matrix. Has copy constructor support.

**`__str__(self)`**
> None

**`getDof(self)`**
> Returns the degrees of freedom.

**`getInvScale(self)`**
> Returns the inverse of the scale matrix.

**`getNorm(self)`**
> Returns the normalising constant of the distribution, typically not used by users.

**`getScale(self)`**
> Returns the scale matrix.

**`prob(self, mat)`**
> Returns the probability of the provided matrix, which must be the same shape as the scale matrix and also symmetric and positive definite.

**`sample(self)`**
> Returns a draw from the distribution - will be a symmetric positive definite matrix.

**`setDof(self, dof)`**
> Sets the degrees of freedom of the distribution.

**`setScale(self, scale)`**
> Sets the scale matrix, must be symmetric positive definite

## StudentT() ##
> A feature incomplete multivariate student-t distribution object - at this time it only supports calculating the probability of a sample, and not the ability to make a draw.

**`__init__(self, dims)`**
> dims is the number of dimensions - initalises it to default values with the degrees of freedom set to 1, the location as the zero vector and the identity matrix for the scale. Suports copy construction.

**`__str__(self)`**
> None

**`batchLogProb(self, dm)`**
> Same as batchProb, but returns the logarithm of the probability instead.

**`batchProb(self, dm)`**
> Given a data matrix evaluates the density function for each entry and returns the resulting array of probabilities.

**`getDOF(self)`**
> Returns the degrees of freedom.

**`getInvScale(self)`**
> Returns the inverse of the scale matrix.

**`getLoc(self)`**
> Returns the location vector.

**`getLogNorm(self)`**
> Returns the logarithm of the normalising constant of the distribution. Typically for internal use only.

**`getScale(self)`**
> Returns the scale matrix.

**`logProb(self, x)`**
> Returns the logarithm of prob - faster than a straight call to prob.

**`prob(self, x)`**
> Given a vector x evaluates the density function at that point.

**`setDOF(self, dof)`**
> Sets the degrees of freedom.

**`setInvScale(self, invScale)`**
> Sets the scale matrix by providing its inverse.

**`setLoc(self, loc)`**
> Sets the location vector.

**`setScale(self, scale)`**
> Sets the scale matrix.

## GaussianPrior() ##
> The conjugate prior for the multivariate Gaussian distribution. Maintains the 4 values and supports various operations of interest - initialisation of prior, Bayesian update, drawing a Gaussian and calculating the probability of a data point comming from a Gaussian drawn from the distribution. Not a particularly efficient implimentation, and it has no numerical protection against extremelly large data sets. Interface is not entirly orthogonal, due to the demands of real world usage.

**`__init__(self, dims)`**
> Initialises with everything zeroed out, such that a prior must added before anything interesting is done. Supports cloning.

**`__str__(self)`**
> None

**`addGP(self, gp)`**
> Adds another Gaussian prior, combining the two.

**`addPrior(self, mean, covariance, weight)`**
> Adds a prior to the structure, as an estimate of the mean and covariance matrix, with a weight which can be interpreted as how many samples that estimate is worth. Note the use of 'add' - you can call this after adding actual samples, or repeatedly. If weight is omitted it defaults to the number of dimensions, as the total weight in the system must match or excede this value before draws etc can be done.

**`addSample(self, sample, weight = 1.0)`**
> Updates the prior given a single sample drawn from the Gaussian being estimated. Can have a weight provided, in which case it will be equivalent to repetition of that data point, where the repetition count can be fractional.

**`addSamples(self, samples, weight)`**
> Updates the prior given multiple samples drawn from the Gaussian being estimated. Expects a data matrix ([sample, position in sample]), or an object that numpy.asarray will interpret as such. Note that if you have only a few samples it might be faster to repeatedly call addSample, as this is designed to be efficient for hundreds+ of samples. You can optionally weight the samples, by providing an array to the weight parameter.

**`getInverseLambda(self)`**
> Returns the inverse of lambda.

**`getK(self)`**
> Returns k.

**`getLambda(self)`**
> Returns lambda.

**`getMu(self)`**
> Returns mu.

**`getN(self)`**
> Returns n.

**`intProb(self)`**
> Returns a multivariate student-t distribution object that gives the probability of drawing a sample from a Gaussian drawn from this prior, with the Gaussian integrated out. You may then call the prob method of this object on each sample obtained.

**`make_safe(self)`**
> Checks for a singular inverse shape matrix - if singular replaces it with the identity. Also makes sure n and k are not less than the number of dimensions, clamping them if need be. obviously the result of this is quite arbitary, but its better than getting a crash from bad data.

**`prob(self, gauss)`**
> Returns the probability of drawing the provided Gaussian from this prior.

**`remSample(self, sample)`**
> Does the inverse of addSample, to in effect remove a previously added sample. Note that the issues of floating point (in-)accuracy mean its not perfect, and removing all samples is bad if there is no prior. Does not support weighting - effectvily removes a sample of weight 1.

**`reset(self)`**
> Resets as though there is no data, other than the dimensions of course.

**`reweight(self, newN, newK)`**
> A slightly cheaky method that reweights the gp such that it has the new values of n and k, effectivly adjusting the relevant weightings of the samples - can be useful for generating a prior for some GPs using the data stored in those GPs. If a new k is not provided it is set to n; if a new n is not provided it is set to the number of dimensions.

**`safe(self)`**
> Returns true if it is possible to sample the prior, work out the probability of samples or work out the probability of samples being drawn from a collapsed sample - basically a test that there is enough information.

**`sample(self)`**
> Returns a Gaussian, drawn from this prior.