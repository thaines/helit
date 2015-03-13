# Dirichlet Process Gaussian Mixture Model #

## Overview ##
**A Dirichlet Process Gaussian Mixture Model, implemented using the mean-field variational method.**

The main paper describing this method is:
'Variational Inference for Dirichlet Process Mixtures' by D. M. Blei and M. I. Jordan.
but it uses the capping method given by
'Accelerated Variational Dirichlet Process Mixtures' by K. Kurihara, M. Welling and N. Vlassis.
so it can operate incrementally.

Its a non-parametric Bayesian density estimator, using Gaussian kernels - its the best general purpose density estimation method that I know of (In the sense of having an entirely rigorous justification, neat solution method, and parameters that can be mostly ignored.), though it does not scale too well for larger data sets, both in terms of computation and memory. It can also be used for clustering.

Implementation is incredibly neat, given how complex the maths behind it is, and also 100% python, for maximum portability (Algorithm makes good use of vectorisation, so no significant speed advantage would be gained from using scipy.weave.). It also only requires a single 400 line file, though does make use of the gcp module.

Typical usage consists of creating the object, providing the number of dimensions of the feature vectors/samples and then using add() to add the said samples. After adding samples setPrior() is called, often with no parameters, and then optionally setConcGamma() can be called - this last method controlls its preference towards having few/lots of clusters. After this a solving method is called - solveGrow() is a good default choice. Note that, due to the somewhat erratic convergance speed no attempt is made to predict run time and provide any kind of progress bar. Once the model is fitted two common modes of usage are as a density estimate and as a clustering algorithm. For density estimation the prob() method will provide the probability of a presented sample being drawn from the model, whilst for clustering the stickProb() method will provide the probability of a provided feature vector having been drawn from each stick (cluster) in the model.


`dpgmm.py` - Contains the DPGMM class - pure awesome.


`test_1d_1mode.py` - Tests it for 1D data with a single mode.

`test_1d_2mode.py` - Two mode version of the above.

`test_1d_3mode.py` - Three mode version of the above.

`test_grow.py` - Tests the ability to grow the number of sticks to model, until the optimal number is found.

`test_stick_inc.py` - Like the above, but does things another way.

`test_cluster.py` - Tests the models ability to do clustering.


`readme.txt` - This file, which gets copied into the documentation.

`make_doc.py` - Creates the html documentation.


---


# Classes #

## DPGMM() ##
> A Dirichlet process Gaussian mixture model, implimented using the mean-field variational method, with the stick tying rather than capping method such that incrimental usage works. For those unfamiliar with Dirichlet processes the key thing to realise is that each stick corresponds to a mixture component for density estimation or a cluster for clustering, so the stick cap is the maximum number of these entities (Though it can choose to use less sticks than supplied.). As each stick has a computational cost standard practise is to start with a low number of sticks and then increase the count, updating the model each time, until no improvement to the model is obtained with further sticks. Note that because the model is fully Bayesian the output is in fact a probability distribution over the probability distribution from which the data is drawn, i.e. instead of the point estimate of a Gaussian mixture model you get back a probability distribution from which you can draw a Gaussian mixture model, though shortcut methods are provided to get the probability/component membership probability of features. Regarding the model it actually has an infinite number of sticks, but as computers don't have an infinite amount of memory or computation the sticks count is capped. The infinite number of sticks past the cap are still modeled in a limited manor, such that you can get the probability in clustering of a sample/feature belonging to an unknown cluster (i.e. one of the infinite number past the stick cap.). It also means that there is no explicit cluster count, as it is modelling a probability distribution over the number of clusters - if you need a cluster count the best choice is to threshold on the component weights, as returned by sampleMixture(...) or intMixture(...), i.e. keep only thise that are higher than a percentage of the highest weight.

**`__init__(self, dims, stickCap = 1)`**
> You initialise with the number of dimensions and the cap on the number of sticks to have. Note that the stick cap should be high enough for it to represent enough components, but not so high that you run out of memory. The better option is to set the stick cap to 1 (The default) and use the solve grow methods, which in effect find the right number of sticks. Alternativly if only given one parameter of the same type it acts as a copy constructor.

**`add(self, sample)`**
> Adds either a single sample or several samples - either give a single sample as a 1D array or a 2D array as a data matrix, where each sample is [i,:]. (Sample = feature. I refer to them as samples as that more effectivly matches the concept of this modeling the probability distribution from which the features are drawn.)

**`getDM(self)`**
> Returns a data matrix containing all the samples that have been added.

**`getStickCap(self)`**
> Returns the current stick cap.

**`incStickCap(self, inc = 1)`**
> Increases the stick cap by the given number of entrys. Can be used in collaboration with nllData to increase the number of sticks until insufficient improvement, indicating the right number has been found.

**`intMixture(self)`**
> Returns the details needed to calculate the probability of a point given the model (density estimation), or its probability of belonging to each stick (clustering), but with the actual draw of a mixture model from the model integrated out. It is an apprximation, though not a bad one. Basically you get a 2-tuple - the first entry is an array of weights, the second a list of student-t distributions. The weights and distributions align, such that for density estimation the probability for a point is the sum over all entrys of the weight multiplied by the probability of the sample comming from the student-t distribution. The prob method of this class calculates the use of this for a sample directly. For clustering the probability of belonging to each cluster is calculated as the weight multiplied by the probability of comming from the associated student-t, noting that you will need to normalise. stickProb allows you to get this assesment directly. Do not edit the returned value; also, it will not persist if solve is called again. This must only be called after solve is called at least once. Note that an extra element is included to cover the remainder of the infinite number of elements - be warned that a sample could have the highest probability of belonging to this dummy element, indicating that it probably belongs to something for which there is not enough data to infer a reasonable model.

**`lock(self, num = 0)`**
> Prevents the algorithm updating the component weighting for the first num samples in the database - potentially useful for incrimental learning if in a rush. If set to 0, the default, everything is updated.

**`multiGrowSolve(self, runs)`**
> This effectivly does multiple calls of growSolve, as indicated by runs, and returns a clone of this object that has converged to the best solution found. This is without a doubt the best solving techneque provided by this method, just remember to use the default stickCap of 1 when setting up the object. Also be warned - this can take an aweful long time to produce its awesomness. Can return self, if the starting number of sticks is the best (Note that self will always be converged after this method returns.).

**`multiSolve(self, runs, testIter = 256)`**
> Clones this object a number of times, given by the runs parameter, and then runs each for the testIters parameter number of iterations, to give them time to converge a bit. It then selects the one with the best nllData() score and runs that to convergance, before returning that specific clone. This is basically a simple way of avoiding getting stuck in a really bad local minima, though chances are you will end up in another one, just not a terrible one. Obviously testIter limits the effectivness of this, but as it tends to converge faster if your closer to the correct answer hopefully not by much. (To be honest I have not found this method to be of much use - in testing when this techneque converges to the wrong answer it does so consistantly, indicating that there is insufficient data regardless of initialisation.)

**`nllData(self)`**
> Returns the negative log likelihood of the data given the current distribution over models, with the model integrated out - good for comparing multiple restarts/different numbers of sticks to find which is the best.

**`prob(self, x)`**
> Given a sample this returns its probability, with the actual draw from the model integrated out. Must not be called until after solve has been called. This is the density estimate if using this model for density estimation. Will also accept a data matrix, in which case it will return a 1D array of probabilities aligning with the input data matrix.

**`reset(self, alphaParam = True, vParam = True, zParam = True)`**
> Allows you to reset the parameters associated with each variable - good for doing a restart if your doing multiple restarts, or if you suspect it has got stuck in a local minima whilst doing incrimental stuff.

**`sampleMixture(self)`**
> Once solve has been called and a distribution over models determined this allows you to draw a specific model. Returns a 2-tuple, where the first entry is an array of weights and the second entry a list of Gaussian distributions - they line up, to give a specific Gaussian mixture model. For density estimation the probability of a specific point is then the sum of each weight multiplied by the probability of it comming from the associated Gaussian. For clustering the probability of a specific point belonging to a cluster is the weight multiplied by the probability of it comming from a specific Gaussian, normalised for all clusters. Note that this includes an additional term to cover the infinite number of terms that follow, which is really an approximation, but tends to be such a small amount as to not matter. Be warned that if doing clustering a point could be assigned to this 'null cluster', indicating that the model thinks the point belongs to an unknown cluster (i.e. one that it doesn't have enough information, or possibly sticks, to instanciate.).

**`setConcGamma(self, alpha, beta)`**
> Sets the parameters for the Gamma prior over the concentration. Note that whilst alpha and beta are used for the parameter names, in accordance with standard terminology for Gamma distributions, they are not related to the model variable names. Default values are (1,1). The concentration parameter controls how much information the model requires to justify using a stick, such that lower numbers result in fewer sticks, higher numbers in larger numbers of sticks. The concentration parameter is learnt from the data, under the Gamma distribution prior set with this method, but this prior can still have a strong effect. If your data is not producing as many clusters as you expect then adjust this parameter accordingly, (e.g. increase alpha or decrease beta, or both.), but don't go too high or it will start hallucinating patterns where none exist!

**`setPrior(self, mean, covar, weight, scale = 1.0)`**
> Sets a prior for the mixture components - basically a pass through for the addPrior method of the GaussianPrior class. If None (The default) is provided for the mean or the covar then it calculates these values for the currently contained sample set and uses them. Note that the prior defaults to nothing - this must be called before fitting the model, and if mean/covar are not provided then there must be enough data points to avoid problems. weight defaults to the number of dimensions if not specified. If covar is not given then scale is a multiplier for the covariance matrix - setting it high will soften the prior up and make it consider softer solutions when given less data. Returns True on success, False on failure - failure can happen if there is not enough data contained for automatic calculation (Think singular covariance matrix). This must be called before any solve methods are called.

**`setThreshold(self, epsilon)`**
> Sets the threshold for parameter change below which it considers it to have converged, and stops iterating.

**`size(self)`**
> Returns the number of samples that have been added.

**`solve(self, iterCap)`**
> Iterates updating the parameters until the model has converged. Note that the system is designed such that you can converge, add more samples, and converge again, i.e. incrimental learning. Alternativly you can converge, add more sticks, and then convegre again without issue, which makes finding the correct number of sticks computationally reasonable.. Returns the number of iterations required to acheive convergance. You can optionally provide a cap on the number of iterations it will perform.

**`solveGrow(self, iterCap)`**
> This method works by solving for the current stick cap, and then it keeps increasing the stick cap until there is no longer an improvement in the model quality. If using this method you should probably initialise with a stick cap of 1. By using this method the stick cap parameter is lost, and you no longer have to guess what a good value is.

**`stickProb(self, x)`**
> Given a sample this returns its probability of belonging to each of the components, as a 1D array, including a dummy element at the end to cover the infinite number of sticks not being explicitly modeled. This is the probability of belonging to each cluster if using the model for clustering. Must not be called until after solve has been called. Will also accept a data matrix, in which case it will return a matrix with a row for each vector in the input data matrix.