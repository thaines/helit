# Probabilistic Classification #

## Overview ##
**Probabilistic Classifiers**

A simple set of classifiers, all working on the principal of building a density estimate for each category and then using Bayes rule to work out the probability of belonging to each, before finally selecting the category the sample is most likely to belong to (The word category is used rather than class to avoid the problem of class being a keyword in Python and many other languages.). Whilst a discriminative model will typically get better results (e.g. the support vector machine (svm) model that is also available.) the advantage of getting a probability distribution is not to be sniffed at. Features such as incremental learning and estimating the odds that the sample belongs to none of the available categories are also included.

Provides a standard-ish interface for a classifier, and then 3 actual implementations, using 3 different density estimation methods. The methods range in complexity and speed - the Gaussian method is obviously very simple, but also extremely fast, and has proper prior handling. The KDE (Kernel density estimation) method is reasonably fast, but requires some parameter tuning. The DPGMM (Dirichlet process Gaussian mixture model.) method is defiantly the best, but extremely expensive to run.

Requires the relevant density estimation modules be available to run, though the header (p\_cat.py) that you include is coded to only load models that are available, so you can chop things down to size if desired. The entire module, and its dependencies, are coded in Python (As of this minute - its likely I might optionally accelerate them via scipy.weave in the future.), making it easy to get running. Its actually very little code, as most of the code is in the density estimation modules on which it depends.


`p_cat.py` - The file that conveniently provides everything.


`prob_cat.py` - The standard interface that all classifiers implement.


`classify_gaussian.py` - Contains ClassifyGaussian, which uses a Gaussian for each category, albeit in a fully Bayesian manor.

`classify_kde.py` - Contains ClassifyKDE, which models its categories using a kernel density estimate.

`classify_dpgmm.py` - Contains ClassifyDPGMM, which uses the heavy weight that is the Dirichlet process gaussian mixture model for its density estimates.


`test_iris.py` - Test file, to verify it all works.


`make_doc.py` - Generates the HTML documentation.

`readme.txt` - This file, which gets copied into the HTML documentation.


---


# Classes #

## ProbCat() ##
> A standard interface for a probabilistic classifier. Provides an interface for a system where you add a bunch of 'samples' and their categories, and can then request the probability that new samples belong to each category. Designed so it can be used in an incrimental manor - if a classifier does not support that it is responsible for prentending that it does, by retrainning each time it is used after new data has been provided. Being probabilistic the interface forces the existance of a prior - it can be a fake-prior that is never used or updated, but it must exist. Obviously it being handled correctly is ideal. The 'sample' object often passed in is category specific, though in a lot of cases will be something that can be interpreted as a 1D numpy array, i.e. a feature vector. The 'cat' object often passed in represents a category - whilst it can be model specific it is generally considered to be a python object that is hashable and can be tested for equality. Being hashable is important to allow it to key a dictionary.

**`add(self, sample, cat)`**
> Given a sample and its category this updates the model, hopefully incrimentally during this method call, but otherwise when the model is next needed. Stuff added to a specific category should **not** be added to the prior by this method - a user can decide to do that by additionally calling priorAdd if they so choose.

**`getCat(self, sample, weight = False, conc, state)`**
> Simply calls through to getNLL and returns the category with the highest probability. If conc is provided it can be None, to indicate a new category is more likelly than any of the existing ones. A simple conveniance method. state is passed through to the getDataProb call.

**`getCatCounts(self)`**
> Returns a dictionary indexable by each of the categories that goes to how many samples of that category have been provided. The returned dictionary must not be edited by the user.

**`getCatList(self, sample, weight = False, conc, state)`**
> List version of getCat, will only work if listMode returns True. Same as getCat but it returns a list of answers, as samples of a possible answer.

**`getCatTotal(self)`**
> Returns the total number of categories that have been provided to the classifier.

**`getDataNLL(self, sample, state)`**
> Identical to getDataProb, except it returns negative log liklihood instead of probabilities. Default implimentation converts the return value of getDataProb, so either that or this needs to be overriden.

**`getDataNLLList(self, sample, state)`**
> The negative log liklihood version of getDataProbList.

**`getDataProb(self, sample, state)`**
> Returns a dictionary indexed by the various categories, with the probabilities of the sample being drawn from the respective categories. Must also include an entry indexed by 'None' that represents the probability of the sample comming from the prior. Note that this is P(data|category,model) - you probably want it reversed, which requires Bayes rule be applied. state is an optional dictionary - if you are calling this repeatedly on the same sample, e.g. during incrimental learning, then state allows an algorithm to store data to accelerate future calls. There should be a dictionary for each sample, and it should be empty on the first call. The implimentation will presume that the sample is identical for each call but that the model will not be, though as it would typically be used for incrimental learning the speed up can be done under the assumption that the model has only changed a little bit.

**`getDataProbList(self, sample, state)`**
> Is only implimented if listMode returns True. Does the same things as getDataProb, but returns a list of answers.

**`getNLL(self, sample, weight = False, conc, state)`**
> Negative log liklihood equivalent of getProb.

**`getNLLList(self, sample, weight = False, conc, state)`**
> Negative log liklihood version of getProbList

**`getProb(self, sample, weight = False, conc, state)`**
> This calls through to getDataProb and then applies Bayes rule to return a dictionary of values representing P(data,category|model) = P(data|category,model)P(category). Note that whilst the two terms going into the return value will be normalised the actual return value will not - you will typically normalise to get P(category|data,model). The weight parameter indicates the source of P(class) - False (The default) indicates to use a uniform prior, True to weight by the number of instances of each category that have been provided to the classifier. Alternativly a dictionary indexed by the categories can be provided of weights, which will be normalised and used. By default the prior probability is ignored, but if a concentration (conc) value is provided it assumes a Dirichlet process, and you will have an entry in the return value, indexed by None, indicating the probability that it belongs to a previously unhandled category. For normalisation purposes conc is always assumed to be in relation to the number of samples that have been provided to the classifier, regardless of weight. state is passed through to the getDataProb call.

**`getProbList(self, sample, weight = False, conc, state)`**
> List version of getProb, will only work if listMode returns True. Same as getProb but it returns a list of answers, as samples of a possible answer.

**`getSampleTotal(self)`**
> Returns the total number of samples that have been provided to train the classifier (Does not include the samples provided to build the prior.).

**`listMode(self)`**
> Returns true if list mode is supported, False otherwise. List mode basically provides some of the same methods, with the 'List' postfix, where the return is a list of answers which the usual method would provide just one of. Used when there are multiple estimates, e.g. multiple draws from a posterior on the model.

**`priorAdd(self, sample)`**
> Adds a sample to the prior; being that it is the prior no category is provided with. This can do nothing, but is typically used to build a prior over samples when the category is unknown.

## ClassifyGaussian(ProbCat) ##
> A simplistic Gaussian classifier, that uses a single Gaussian to represent each category/the prior. It is of course fully Bayesian. It keeps a prior that is worth the number of dimensions with the mean and covariance of all the samples provided for its construction. Implimentation is not very efficient, though includes some caching to stop things being too slow.

**`__init__(self, dims)`**
> dims is the number of dimensions.

**`add(self, sample, cat)`**
> Given a sample and its category this updates the model, hopefully incrimentally during this method call, but otherwise when the model is next needed. Stuff added to a specific category should **not** be added to the prior by this method - a user can decide to do that by additionally calling priorAdd if they so choose.

**`getCat(self, sample, weight = False, conc, state)`**
> Simply calls through to getNLL and returns the category with the highest probability. If conc is provided it can be None, to indicate a new category is more likelly than any of the existing ones. A simple conveniance method. state is passed through to the getDataProb call.

**`getCatCounts(self)`**
> Returns a dictionary indexable by each of the categories that goes to how many samples of that category have been provided. The returned dictionary must not be edited by the user.

**`getCatList(self)`**
> List version of getCat, will only work if listMode returns True. Same as getCat but it returns a list of answers, as samples of a possible answer.

**`getCatTotal(self)`**
> Returns the total number of categories that have been provided to the classifier.

**`getDataNLL(self, sample, state)`**
> Identical to getDataProb, except it returns negative log liklihood instead of probabilities. Default implimentation converts the return value of getDataProb, so either that or this needs to be overriden.

**`getDataNLLList(self, sample, state)`**
> The negative log liklihood version of getDataProbList.

**`getDataProb(self, sample, state)`**
> Returns a dictionary indexed by the various categories, with the probabilities of the sample being drawn from the respective categories. Must also include an entry indexed by 'None' that represents the probability of the sample comming from the prior. Note that this is P(data|category,model) - you probably want it reversed, which requires Bayes rule be applied. state is an optional dictionary - if you are calling this repeatedly on the same sample, e.g. during incrimental learning, then state allows an algorithm to store data to accelerate future calls. There should be a dictionary for each sample, and it should be empty on the first call. The implimentation will presume that the sample is identical for each call but that the model will not be, though as it would typically be used for incrimental learning the speed up can be done under the assumption that the model has only changed a little bit.

**`getDataProbList(self, sample, state)`**
> Is only implimented if listMode returns True. Does the same things as getDataProb, but returns a list of answers.

**`getNLL(self, sample, weight = False, conc, state)`**
> Negative log liklihood equivalent of getProb.

**`getNLLList(self, sample, weight = False, conc, state)`**
> Negative log liklihood version of getProbList

**`getProb(self, sample, weight = False, conc, state)`**
> This calls through to getDataProb and then applies Bayes rule to return a dictionary of values representing P(data,category|model) = P(data|category,model)P(category). Note that whilst the two terms going into the return value will be normalised the actual return value will not - you will typically normalise to get P(category|data,model). The weight parameter indicates the source of P(class) - False (The default) indicates to use a uniform prior, True to weight by the number of instances of each category that have been provided to the classifier. Alternativly a dictionary indexed by the categories can be provided of weights, which will be normalised and used. By default the prior probability is ignored, but if a concentration (conc) value is provided it assumes a Dirichlet process, and you will have an entry in the return value, indexed by None, indicating the probability that it belongs to a previously unhandled category. For normalisation purposes conc is always assumed to be in relation to the number of samples that have been provided to the classifier, regardless of weight. state is passed through to the getDataProb call.

**`getProbList(self, sample, weight = False, conc, state)`**
> List version of getProb, will only work if listMode returns True. Same as getProb but it returns a list of answers, as samples of a possible answer.

**`getSampleTotal(self)`**
> Returns the total number of samples that have been provided to train the classifier (Does not include the samples provided to build the prior.).

**`getStudentT(self)`**
> Returns a dictionary with categories as keys and StudentT distributions as values, these being the probabilities of samples belonging to each class with the actual draw from the posterior integrated out. Also stores the prior, under a key of None.

**`listMode(self)`**
> Returns true if list mode is supported, False otherwise. List mode basically provides some of the same methods, with the 'List' postfix, where the return is a list of answers which the usual method would provide just one of. Used when there are multiple estimates, e.g. multiple draws from a posterior on the model.

**`priorAdd(self, sample)`**
> Adds a sample to the prior; being that it is the prior no category is provided with. This can do nothing, but is typically used to build a prior over samples when the category is unknown.

## ClassifyKDE(ProbCat) ##
> A classifier that uses the incrimental kernel density estimate model for each category. It keeps a 'psuedo-prior', a KDE\_INC with an (optionally) larger variance that contains all the samples. Uses entities that can index a dictionary for categories.

**`__init__(self, prec, cap = 32, mult = 1.0)`**
> You provide the precision that is to be used (As a 2D numpy array, so it implicitly provides the number of dimensions.), the cap on the number of components in the KDE\_INC objects and the multiplier for the standard deviation of the components in the 'psuedo-prior'.

**`add(self, sample, cat)`**
> Given a sample and its category this updates the model, hopefully incrimentally during this method call, but otherwise when the model is next needed. Stuff added to a specific category should **not** be added to the prior by this method - a user can decide to do that by additionally calling priorAdd if they so choose.

**`getCat(self, sample, weight = False, conc, state)`**
> Simply calls through to getNLL and returns the category with the highest probability. If conc is provided it can be None, to indicate a new category is more likelly than any of the existing ones. A simple conveniance method. state is passed through to the getDataProb call.

**`getCatCounts(self)`**
> Returns a dictionary indexable by each of the categories that goes to how many samples of that category have been provided. The returned dictionary must not be edited by the user.

**`getCatList(self)`**
> List version of getCat, will only work if listMode returns True. Same as getCat but it returns a list of answers, as samples of a possible answer.

**`getCatTotal(self)`**
> Returns the total number of categories that have been provided to the classifier.

**`getDataNLL(self, sample, state)`**
> Identical to getDataProb, except it returns negative log liklihood instead of probabilities. Default implimentation converts the return value of getDataProb, so either that or this needs to be overriden.

**`getDataNLLList(self, sample, state)`**
> The negative log liklihood version of getDataProbList.

**`getDataProb(self, sample, state)`**
> Returns a dictionary indexed by the various categories, with the probabilities of the sample being drawn from the respective categories. Must also include an entry indexed by 'None' that represents the probability of the sample comming from the prior. Note that this is P(data|category,model) - you probably want it reversed, which requires Bayes rule be applied. state is an optional dictionary - if you are calling this repeatedly on the same sample, e.g. during incrimental learning, then state allows an algorithm to store data to accelerate future calls. There should be a dictionary for each sample, and it should be empty on the first call. The implimentation will presume that the sample is identical for each call but that the model will not be, though as it would typically be used for incrimental learning the speed up can be done under the assumption that the model has only changed a little bit.

**`getDataProbList(self, sample, state)`**
> Is only implimented if listMode returns True. Does the same things as getDataProb, but returns a list of answers.

**`getNLL(self, sample, weight = False, conc, state)`**
> Negative log liklihood equivalent of getProb.

**`getNLLList(self, sample, weight = False, conc, state)`**
> Negative log liklihood version of getProbList

**`getProb(self, sample, weight = False, conc, state)`**
> This calls through to getDataProb and then applies Bayes rule to return a dictionary of values representing P(data,category|model) = P(data|category,model)P(category). Note that whilst the two terms going into the return value will be normalised the actual return value will not - you will typically normalise to get P(category|data,model). The weight parameter indicates the source of P(class) - False (The default) indicates to use a uniform prior, True to weight by the number of instances of each category that have been provided to the classifier. Alternativly a dictionary indexed by the categories can be provided of weights, which will be normalised and used. By default the prior probability is ignored, but if a concentration (conc) value is provided it assumes a Dirichlet process, and you will have an entry in the return value, indexed by None, indicating the probability that it belongs to a previously unhandled category. For normalisation purposes conc is always assumed to be in relation to the number of samples that have been provided to the classifier, regardless of weight. state is passed through to the getDataProb call.

**`getProbList(self, sample, weight = False, conc, state)`**
> List version of getProb, will only work if listMode returns True. Same as getProb but it returns a list of answers, as samples of a possible answer.

**`getSampleTotal(self)`**
> Returns the total number of samples that have been provided to train the classifier (Does not include the samples provided to build the prior.).

**`listMode(self)`**
> Returns true if list mode is supported, False otherwise. List mode basically provides some of the same methods, with the 'List' postfix, where the return is a list of answers which the usual method would provide just one of. Used when there are multiple estimates, e.g. multiple draws from a posterior on the model.

**`priorAdd(self, sample)`**
> Adds a sample to the prior; being that it is the prior no category is provided with. This can do nothing, but is typically used to build a prior over samples when the category is unknown.

**`setPrec(self, prec)`**
> Changes the precision matrix - must be called before any samples are added, and must have the same dimensions as the current one.

## ClassifyDPGMM(ProbCat) ##
> A classifier that uses a Dirichlet process Gaussian mixture model (DPGMM) for each category. Also includes a psuedo-prior in the form of an extra DPGMM that you can feed. Trains them incrimentally, increasing the mixture component cap when that results in an improvement in model performance. Be aware that whilst this is awesome its memory consumption can be fierce, and its a computational hog. Includes the ability to switch off incrimental learning, which can save some time if your not using the model between trainning samples.

**`__init__(self, dims, runs = 1)`**
> dims is the number of dimensions the input vectors have, whilst runs is how many starting points to converge from for each variational run. Increasing runs helps to avoid local minima at the expense of computation, but as it often converges well enough with the first attempt, so this is only for the paranoid.

**`add(self, sample, cat)`**
> Given a sample and its category this updates the model, hopefully incrimentally during this method call, but otherwise when the model is next needed. Stuff added to a specific category should **not** be added to the prior by this method - a user can decide to do that by additionally calling priorAdd if they so choose.

**`getCat(self, sample, weight = False, conc, state)`**
> Simply calls through to getNLL and returns the category with the highest probability. If conc is provided it can be None, to indicate a new category is more likelly than any of the existing ones. A simple conveniance method. state is passed through to the getDataProb call.

**`getCatCounts(self)`**
> Returns a dictionary indexable by each of the categories that goes to how many samples of that category have been provided. The returned dictionary must not be edited by the user.

**`getCatList(self)`**
> List version of getCat, will only work if listMode returns True. Same as getCat but it returns a list of answers, as samples of a possible answer.

**`getCatTotal(self)`**
> Returns the total number of categories that have been provided to the classifier.

**`getDataNLL(self, sample, state)`**
> Identical to getDataProb, except it returns negative log liklihood instead of probabilities. Default implimentation converts the return value of getDataProb, so either that or this needs to be overriden.

**`getDataNLLList(self, sample, state)`**
> The negative log liklihood version of getDataProbList.

**`getDataProb(self, sample, state)`**
> Returns a dictionary indexed by the various categories, with the probabilities of the sample being drawn from the respective categories. Must also include an entry indexed by 'None' that represents the probability of the sample comming from the prior. Note that this is P(data|category,model) - you probably want it reversed, which requires Bayes rule be applied. state is an optional dictionary - if you are calling this repeatedly on the same sample, e.g. during incrimental learning, then state allows an algorithm to store data to accelerate future calls. There should be a dictionary for each sample, and it should be empty on the first call. The implimentation will presume that the sample is identical for each call but that the model will not be, though as it would typically be used for incrimental learning the speed up can be done under the assumption that the model has only changed a little bit.

**`getDataProbList(self, sample, state)`**
> Is only implimented if listMode returns True. Does the same things as getDataProb, but returns a list of answers.

**`getNLL(self, sample, weight = False, conc, state)`**
> Negative log liklihood equivalent of getProb.

**`getNLLList(self, sample, weight = False, conc, state)`**
> Negative log liklihood version of getProbList

**`getProb(self, sample, weight = False, conc, state)`**
> This calls through to getDataProb and then applies Bayes rule to return a dictionary of values representing P(data,category|model) = P(data|category,model)P(category). Note that whilst the two terms going into the return value will be normalised the actual return value will not - you will typically normalise to get P(category|data,model). The weight parameter indicates the source of P(class) - False (The default) indicates to use a uniform prior, True to weight by the number of instances of each category that have been provided to the classifier. Alternativly a dictionary indexed by the categories can be provided of weights, which will be normalised and used. By default the prior probability is ignored, but if a concentration (conc) value is provided it assumes a Dirichlet process, and you will have an entry in the return value, indexed by None, indicating the probability that it belongs to a previously unhandled category. For normalisation purposes conc is always assumed to be in relation to the number of samples that have been provided to the classifier, regardless of weight. state is passed through to the getDataProb call.

**`getProbList(self, sample, weight = False, conc, state)`**
> List version of getProb, will only work if listMode returns True. Same as getProb but it returns a list of answers, as samples of a possible answer.

**`getSampleTotal(self)`**
> Returns the total number of samples that have been provided to train the classifier (Does not include the samples provided to build the prior.).

**`listMode(self)`**
> Returns true if list mode is supported, False otherwise. List mode basically provides some of the same methods, with the 'List' postfix, where the return is a list of answers which the usual method would provide just one of. Used when there are multiple estimates, e.g. multiple draws from a posterior on the model.

**`priorAdd(self, sample)`**
> Adds a sample to the prior; being that it is the prior no category is provided with. This can do nothing, but is typically used to build a prior over samples when the category is unknown.

**`setInc(self, state)`**
> With a state of False it disables incrimental learning until further notice, with a state of True it reenables it, and makes sure that it is fully up to date by updating everything. Note that when reenabled it assumes that enough data is avaliable, and will crash if not, unlike the incrimental approach that just twiddles its thumbs - in a sense this is safer if you want to avoid bad results.

## ClassifyDF(ProbCat) ##
> A classifier that uses decision forests. Includes the use of a density estimate decision forest as a psuedo-prior. The incrimental method used is rather simple, but still works reasonably well. Provides default parameters for the decision forests, but allows access to them for if you want to mess around. Internally the decision forests have two channels - the first is the data, the second the class.

**`__init__(self, dims, treeCount, incAdd = 1, testDims = 3, dimCount = 4, rotCount = 32)`**
> dims is the number of dimensions in each sample. treeCount is how many trees to use whilst incAdd is how many to train for each new sample. testDims is the number of dimensions to use for each test, dimCount the number of combinations of dimensions to try for generating each nodes decision and rotCount the number of orientations to try for each nodes test generation.

**`add(self, sample, cat)`**
> Given a sample and its category this updates the model, hopefully incrimentally during this method call, but otherwise when the model is next needed. Stuff added to a specific category should **not** be added to the prior by this method - a user can decide to do that by additionally calling priorAdd if they so choose.

**`getCat(self, sample, weight = False, conc, state)`**
> Simply calls through to getNLL and returns the category with the highest probability. If conc is provided it can be None, to indicate a new category is more likelly than any of the existing ones. A simple conveniance method. state is passed through to the getDataProb call.

**`getCatCounts(self)`**
> Returns a dictionary indexable by each of the categories that goes to how many samples of that category have been provided. The returned dictionary must not be edited by the user.

**`getCatList(self)`**
> List version of getCat, will only work if listMode returns True. Same as getCat but it returns a list of answers, as samples of a possible answer.

**`getCatTotal(self)`**
> Returns the total number of categories that have been provided to the classifier.

**`getClassifier(self)`**
> Returns the decision forest used for classification.

**`getDataNLL(self, sample, state)`**
> Identical to getDataProb, except it returns negative log liklihood instead of probabilities. Default implimentation converts the return value of getDataProb, so either that or this needs to be overriden.

**`getDataNLLList(self, sample, state)`**
> The negative log liklihood version of getDataProbList.

**`getDataProb(self, sample, state)`**
> Returns a dictionary indexed by the various categories, with the probabilities of the sample being drawn from the respective categories. Must also include an entry indexed by 'None' that represents the probability of the sample comming from the prior. Note that this is P(data|category,model) - you probably want it reversed, which requires Bayes rule be applied. state is an optional dictionary - if you are calling this repeatedly on the same sample, e.g. during incrimental learning, then state allows an algorithm to store data to accelerate future calls. There should be a dictionary for each sample, and it should be empty on the first call. The implimentation will presume that the sample is identical for each call but that the model will not be, though as it would typically be used for incrimental learning the speed up can be done under the assumption that the model has only changed a little bit.

**`getDataProbList(self, sample, state)`**
> Is only implimented if listMode returns True. Does the same things as getDataProb, but returns a list of answers.

**`getDensityEstimate(self)`**
> Returns the decision forest used for density estimation, as a psuedo-prior.

**`getNLL(self, sample, weight = False, conc, state)`**
> Negative log liklihood equivalent of getProb.

**`getNLLList(self, sample, weight = False, conc, state)`**
> Negative log liklihood version of getProbList

**`getProb(self, sample, weight = False, conc, state)`**
> This calls through to getDataProb and then applies Bayes rule to return a dictionary of values representing P(data,category|model) = P(data|category,model)P(category). Note that whilst the two terms going into the return value will be normalised the actual return value will not - you will typically normalise to get P(category|data,model). The weight parameter indicates the source of P(class) - False (The default) indicates to use a uniform prior, True to weight by the number of instances of each category that have been provided to the classifier. Alternativly a dictionary indexed by the categories can be provided of weights, which will be normalised and used. By default the prior probability is ignored, but if a concentration (conc) value is provided it assumes a Dirichlet process, and you will have an entry in the return value, indexed by None, indicating the probability that it belongs to a previously unhandled category. For normalisation purposes conc is always assumed to be in relation to the number of samples that have been provided to the classifier, regardless of weight. state is passed through to the getDataProb call.

**`getProbList(self, sample, weight = False, conc, state)`**
> List version of getProb, will only work if listMode returns True. Same as getProb but it returns a list of answers, as samples of a possible answer.

**`getSampleTotal(self)`**
> Returns the total number of samples that have been provided to train the classifier (Does not include the samples provided to build the prior.).

**`listMode(self)`**
> Returns true if list mode is supported, False otherwise. List mode basically provides some of the same methods, with the 'List' postfix, where the return is a list of answers which the usual method would provide just one of. Used when there are multiple estimates, e.g. multiple draws from a posterior on the model.

**`priorAdd(self, sample)`**
> Adds a sample to the prior; being that it is the prior no category is provided with. This can do nothing, but is typically used to build a prior over samples when the category is unknown.

**`setDensityMinTrain(self, count)`**
> None