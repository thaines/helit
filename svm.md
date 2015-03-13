# Support Vector Machine #

## Overview ##
**Support Vector Machine implementation, using the Sequential Minimal Optimization technique.**

Implemented in python using numpy and scipy.weave.
Only provides classification.

The key papers behind this are:
'Support-vector networks' by Cortes and Vapnik provides the basic soft svm idea.
'Fast Training of Support Vector Machines using Sequential Minimal Optimization' by Platt provides the core solution method.
'Working Set Selection Using Second Order Information for Training Support Vector Machines' by Fan, Chen and Lin provides a better pair selection method for the above algorithm.

svm.py provides all the functionality, and is the only file you need import. Standard usage is to create 3 objects - a parameters object, a dataset object and a model object.

The model object should nearly always be MultiModel, even if you only have two classes, as it provides the best interface (Mostly because it supports model selection and multi-processor solving.).

The Dataset object is constructed using one or more data matrix/label vector pairs.

For parameters you either choose the parameters explicitly using a Params() object, or you construct a ParamsSet() object and do model selection. See their respective interfaces for details.

All objects are designed to be compatible with pickle, for file i/o purposes.
If you are reading readme.txt then you can generate documentation by running make\_doc.py, which will include all the relevant interfaces.



`svm.py` - import file, as covered above.


`params.py` - parameters object

`params_sets.py` - objects that emulate being lists of parameters, for model selection.

`dataset.py` - dataset, of feature vectors with labels.


`smo.py` - actual solver object.

`smo_aux.py` - C code strings for solver.

`loo.py` - Does leave one out, to access how good specific parameters are. Primarilly used for model selection.


`model.py` - Two class model as provided by the solver.

`multiclass.py` - Wrapper for multiple models to support one vs one classification; additionally provides model selection and multiprocessor support.


`readme.txt` - this file, which is copied into the start of svm.html if generated.

`make_doc.py` - creates the svm.html help file.


`test_simple.py` - Very simple test of a linear classifier on overlapping uniform distributions.

`test_mult.py` - Model selection test with four classes with overlapping uniform distributions.

`test_curve.py` - Test on a sin curve with model selection. Note this requires opencv 2.x to be installed, as it renders out some images.

`test_2d.py` - Test on a four-class spiral with model selection; also outputs images requiring opencv 2.x.

---


# Functions #

**`solvePair(params, dataset, negLabel, posLabel)`**
> Solves for a pair of labels - you provide a parameters object, a data set and the labels to assign to -1 and +1 respectivly. It then returns a Model object.

**`looPair(params, data)`**
> Given a parameters object and a pair of data matrix and y (As returned by dataset.getTrainData.) this returns a (good) approximation of the leave one out negative log likellihood, and a model trained on **all** the data as a pair. Makes the assumption that losing a non-supporting vector does not require retraining, which is correct the vast majority of the time, and as a bonus avoids retrainning for most of the data, making this relativly fast.

**`looPairSelect(paramsList, data)`**
> Given an iterator of parameters this returns a pair of the loo score and model of the best set of parameters - just loops over looPair.


# Classes #

## Kernel() ##
> Enum of supported kernel types, with some helpful static methods.

**`gbf`** = 3

**`homo_polynomial`** = 1

**`linear`** = 0

**`polynomial`** = 2

**`rbf`** = 3

**`sigmoid`** = 4

## Params() ##
> Parameters for the svm algorithm - softness and kernel. Defaults to a C value of 10 and a linear kernel.

**`__init__(self)`**
> None

**`__str__(self)`**
> None

**`getC(self)`**
> Returns c, the softness parameter.

**`getCode(self)`**
> Returns the code for a function that impliments the specified kernel, with the parameters hard coded in.

**`getKernel(self)`**
> Returns which kernel is being used; see the Kernel enum for transilations of the value.

**`getP1(self)`**
> returns kernel parameter 1, not always used.

**`getP2(self)`**
> returns kernel parameter 2, not always used.

**`getRebalance(self)`**
> Returns whether the c value is rebalanced or not - defaults to true.

**`kernelKey(self)`**
> Returns a string unique to the kernel/kernel parameters combo.

**`setC(self, c)`**
> Sets the c value, whcih indicates how soft the answer can be. (0 don't care, infinity means perfect seperation.) Default is 10.0

**`setGBF(self, sd)`**
> Sets it to use a gaussian basis function, with the given standard deviation. (This is equivalent to a RBF with the scale set to 1/(2\*sd^2))

**`setHomoPoly(self, degree)`**
> Sets it to use a homogenous polynomial, with the given degree.

**`setKernel(self, kernel, p1, p2)`**
> Sets the kernel to use, and the parameters if need be.

**`setLinear(self)`**
> Sets it to use the linear kernel.

**`setP1(self, p1)`**
> Sets parameter p1.

**`setP2(self, p2)`**
> Sets parameter p2.

**`setPoly(self, degree)`**
> Sets it to use a polynomial, with the given degree.

**`setRBF(self, scale)`**
> Sets it to use a radial basis function, with the given distance scale.

**`setRebalance(self, rebalance)`**
> Sets if c is rebalanced or not.

**`setSigmoid(self, scale, offset)`**
> Sets it to be a sigmoid, with the given parameters.

## ParamsRange() ##
> A parameters object where each variable takes a list rather than a single value - it then pretends to be a list of Params objects, which consists of every combination implied by the ranges.

**`__init__(self)`**
> Initialises to contain just the default.

**`__iter__(self)`**
> None

**`getCList(self)`**
> Returns the list of c parameters.

**`getKernelList(self)`**
> Returns the list of kernels.

**`getP1List(self)`**
> returns the list of kernel parameters 1, not always used.

**`getP2List(self)`**
> returns the list of kernel parameters 2, not always used.

**`getRebalanceList(self)`**
> Returns the list of rebalance options - can only ever be two

**`permutations(self)`**
> None

**`setCList(self, c)`**
> Sets the list of c values.

**`setKernelList(self, kernel)`**
> Sets the list of kernels.

**`setP1List(self, p1)`**
> Sets the list of P1 values.

**`setP2List(self, p2)`**
> Sets the list of P2 values.

**`setRebalanceList(self, rebalance)`**
> Sets if c is rebalanced or not.

## ParamsSet() ##
> Pretends to be a list of parameters, when instead it is a list of parameter ranges, where each set of ranges defines a search grid - used for model selection, typically by being passed as the params input to the MultiModel class.

**`__init__(self, incDefault = False, incExtra = False)`**
> Initialises the parameter set - with the default constructor this is empty. However, initalising it with paramsSet(True) gets you a good default set to model select with (That is the addLinear and addPoly methods are called with default parameters.), whilst paramsSet(True,True) gets you an insanely large default set for if your feeling particularly patient (It being all the add methods with default parameters.).

**`__iter__(self)`**
> None

**`addBasisFuncs(self, rExpHigh = 6, rExp = 2.0, sdExpHigh = 6, sdExp = 2.0, cExpLow = -3, cExpHigh = 3, cExp = 10.0, rebalance = True)`**
> Adds the basis functions to the set, both Radial and Gaussian. The parameter for the radial basis functions go from rExp<sup>0 to rExp</sup>rExpHigh, whilst the parameter for the Gaussian does the same thing, but with the sd parameters. Same c controls as for addLinear.

**`addHomoPoly(self, maxDegree = 6, cExpLow = -3, cExpHigh = 3, cExp = 10.0, rebalance = True)`**
> Adds the homogenous polynomial to the set, from an exponent of 2 to the given value inclusive, which defaults to 8. Same c controls as for addLinear.

**`addLinear(self, cExpLow = -3, cExpHigh = 3, cExp = 10.0, rebalance = True)`**
> Adds a standard linear model to the set, with a range of c values. These values will range from cExp<sup>cExpLow to cExp</sup>cExpHigh, and by default are the set {0.001,0.01,0.1,1,10,100,1000}, which is typically good enough.

**`addPoly(self, maxDegree = 6, cExpLow = -3, cExpHigh = 3, cExp = 10.0, rebalance = True)`**
> Adds the polynomial to the set, from an exponent of 2 to the given value inclusive, which defaults to 8. Same c controls as for addLinear.

**`addRange(self, ran)`**
> Adds a new ParamsRange to the set.

**`addSigmoid(self, sExpLow = -3, sExpHigh = 3, sExp = 10.0, oExpLow = -3, oExpHigh = 3, oExp = 10.0, cExpLow = -3, cExpHigh = 3, cExp = 10.0, rebalance = True)`**
> Add sigmoids to the set - the parameters use s for the scale component and o for the offset component; these parameters use the same exponential scheme as for c and others. Same c controls as for addLinear.

**`permutations(self)`**
> None

## Dataset() ##
> Contains a dataset - lots of pairs of feature vectors and labels. For conveniance labels can be arbitrary python objects, or at least python objects that work for indexing a dictionary.

**`__init__(self)`**
> None

**`add(self, featVect, label)`**
> Adds a single feature vector and label.

**`addMatrix(self, dataMatrix, labels)`**
> This adds a data matrix alongside a list of labels for it. The number of rows in the matrix should match the number of labels in the list.

**`getCounts(self)`**
> Returns a how many features with each label have been seen - as a list which aligns with the output of getLabels.

**`getLabels(self)`**
> Returns a list of all the labels in the data set.

**`getTrainData(self, lNeg, lPos)`**
> Given two labels this returns a pair of a data matrix and a y vector, where lPos features have +1 and lNeg features have -1. Features that do not have one of these two labels will not be included.

**`subsampleData(self, count)`**
> Returns a new dataset object which contains count instances of the data, sampled from the data contained within without repetition. Returned Dataset could miss some of the classes.

## Model() ##
> Defines a model - this will consist of a parameters object to define the kernel (C is ignored, but will be the same as the trainning parameter if needed for reference.), a list of support vectors in a dataMatrix and then a vector of weights, plus the b parameter. The weights are the multiple of the y value and alpha value. Uses weave to make evaluation of new features fast.

**`__init__(self, params, supportVectors, supportWeights, b)`**
> Sets up a model given the parameters. Note that if given a linear kernel and multiple support vectors it does the obvious optimisation.

**`classify(self, feature)`**
> Classifies a single feature vector - returns -1 or +1 depending on its class. Just the sign of the decision method.

**`decision(self, feature)`**
> Given a feature vector this returns its decision boundary evaluation, specifically the weighted sum of each of the kernel evaluations for the support vectors against the given feature vector, plus b.

**`getB(self)`**
> Returns the addative offset of the function defined by the support vectors to locate the decision boundary at 0.

**`getParams(self)`**
> Returns the parameters the svm was trainned with.

**`getSupportVectors(self)`**
> Returns a 2D array where each row is a support vector.

**`getSupportWeights(self)`**
> Returns the vector of weights matching the support vectors.

**`multiClassify(self, features)`**
> Given a matrix where every row is a feature returns - returns -1 or +1 depending on the class of each vector, as an array. Just the sign of the multiDecision method. Be warned the classification vector is returned with a type of int8.

**`multiDecision(self, features)`**
> Given a matrix where every row is a feature returns the decision boundary evaluation for each feature as an array of values.

## MultiModel() ##
> This represents a model with multiple labels - uses one against one voting. Even if you only have two labels you are best off using this interface, as it makes everything neat. Supports model selection as well.

**`__init__(self, params, dataset, weightSVM = True, callback, pool, looDist = 1.1)`**
> Trains the model given the dataset and either a params object or a iterator of params objects. If a list it trys all entrys of the list for each pairing, and selects the one that gives the best loo, i.e. does model selection. If weightSVM is True (The default) then it makes use of the leave one out scores calculated during model selection to weight the classification boundaries - this can result in slightly better behavour at the meeting points of multiple classes in feature space. The pool parameter can be passed in a Pool() object from the multiprocessing python module, or set to True to have it create an instance itself. This enables multiprocessor mode for doing each loo calculation required - good if you have lots of models to test and/or lots of labels.

**`classify(self, feature)`**
> Classifies a single feature vector - returns the most likelly label.

**`getLabels(self)`**
> Returns a list of the labels supported.

**`getModel(self, labA, labB)`**
> Returns a tuple of (model,neg label,pos label, loo) where model is the model between the pair and the two labels indicate which label is associated with the negative result and which with the positive result. loo is the leave one out score of this particular boundary.

**`multiClassify(self, features)`**
> Given a matrix where every row is a feature - returns a list of labels for the rows.

**`paramsList(self)`**
> Returns a list of parameters objects used by the model - good for curiosity.