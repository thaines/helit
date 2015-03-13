# Dual Hierarchical Dirichlet Processes #

## Overview ##
**Dual Hierarchical Dirichlet Processes**

An implementation of the algorithm given in 'Trajectory analysis and semantic region modeling using a nonparametric Bayesian model' by X. Wang, K. T. Ma, G-W. Ng and W. E. L. Grimson. Also supports a fallback mode allowing it to do something that is to all intents and purposes normal HDP.

Implemented using scipy.weave - being that it is Gibbs sampled the lions share of the code is in C++.

Files:

`dhdp.py` - Includes everything needed to use the system.


`dp_conc.py` - Contains the values required for the concentration parameter of a DP - its prior and initial value.

`params.py` - Provides the parameters for the solver that are not specific to the problem.

`document.py` - Provides an object representing a document and the words it contains.

`corpus.py` - Contains an object to represent a Corpus - basically all the documents and a bunch of parameters to define exactly what the problem does.

`model.py` - Contains all the objects that represent a model of the data, as provided by the solvers.



`solvers.py` - Internal use file; contains the code to detect the best solver available.

`solve_shared.py` - Contains the python-side representation of the state of a Gibbs sampler.

`solve_weave.py` - Solver that uses scipy.weave.

`solve_weave_mp.py` - Multiprocess version of the scipy.weave based solver.


`ds_cpp.py` - Contains the cpp side representation of the state of a Gibbs sampler.

`ds_link_cpp.py` - Contains the code to convert between the python and cpp states.


`test_lines.py` - Simple test file, to verify that it works. Also acts as an example of usage.


`readme.txt` - This file, which is included in the html documentation.

`make_doc.py` - Creates the html documentation.


---


# Functions #

**`getAlgorithm()`**
> Returns a text string indicating which implimentation of the fitting algorithm is being used by default, which will be the best avaliable.


# Classes #

## PriorConcDP() ##
> Contains the parameters required for the concentration parameter of a DP - specifically its Gamma prior and initial concentration value.

**`__init__(self, other)`**
> None

**`getAlpha(self)`**
> Getter for alpha.

**`getBeta(self)`**
> Getter for beta.

**`getConc(self)`**
> Getter for the initial concentration.

**`setAlpha(self, alpha)`**
> Setter for alpha.

**`setBeta(self, beta)`**
> Setter for beta.

**`setConc(self, conc)`**
> Setter for the initial concentration.

## Params() ##
> Parameters for running the fitter that are universal to all fitters - basically the parameters you would typically associate with Gibbs sampling.

**`__init__(self, toClone)`**
> Sets the parameters to reasonable defaults. Will act as a copy constructor if given an instance of this object.

**`fromArgs(self, args, prefix = )`**
> Extracts from an arg string, typically sys.argv[1:], the parameters, leaving them untouched if not given. Uses --runs, --samples, --burnIn and --lag. Can optionally provide a prefix which is inserted after the '--'

**`getBurnIn(self)`**
> Returns the burn in length.

**`getLag(self)`**
> Returns the lag length.

**`getRuns(self)`**
> Returns the number of runs.

**`getSamples(self)`**
> Returns the number of samples.

**`setBurnIn(self, burnIn)`**
> Number of Gibbs iterations to do for burn in before sampling starts.

**`setLag(self, lag)`**
> Number of Gibbs iterations to do between samples.

**`setRuns(self, runs)`**
> Sets the number of runs, i.e. how many seperate chains are run.

**`setSamples(self, samples)`**
> Number of samples to extract from each chain - total number of samples extracted will hence be samples\*runs.

## Document() ##
> Representation of a document as used by the system. Consists of a list of words - each is referenced by a natural number and is associated with a count of how many of that particular word exist in the document.

**`__init__(self, dic)`**
> Constructs a document given a dictionary (Or equivalent) dic[ident](ident.md) = count, where ident is the natural number that indicates which word and count is how many times that word exists in the document. Excluded entries are effectivly assumed to have a count of zero. Note that the solver will construct an array 0..{max word ident} and assume all words in that range exist, going so far as smoothing in words that are never actually seen.

**`getDic(self)`**
> Returns a dictionary object that represents the document, basically a recreated version of the dictionary handed in to the constructor.

**`getIdent(self)`**
> Ident - just the offset into the array in the corpus where this document is stored, or None if its yet to be stored anywhere.

**`getSampleCount(self)`**
> Returns the number of samples in the document, which is equivalent to the number of words, counting duplicates.

**`getWord(self, index)`**
> Given an index 0..getWordCount()-1 this returns the tuple (ident,count) for that word.

**`getWordCount(self)`**
> Returns the number of unique words in the document, i.e. not counting duplicates.

## Corpus() ##
> Contains a set of Document-s, plus parameters for the graphical models priors - everything required as input to build a model, except a Params object.

**`__init__(self)`**
> Basic setup - sets a whole bunch of stuff to sensible defaults.

**`add(self, doc)`**
> Adds a document to the corpus.

**`documentList(self)`**
> Returns a list of all documents.

**`getAlpha(self)`**
> Returns the PriorConcDP for the alpha parameter.

**`getBeta(self)`**
> Returns the current beta value. Defaults to 1.0.

**`getCalcBeta(self)`**
> Returns False to leave the beta prior on topic word multinomials as is, True to indicate that it should be optimised

**`getCluInstsDNR(self)`**
> Returns False if the cluster instances are going to be resampled, True if they are not.

**`getDocInstsDNR(self)`**
> Returns False if the document instances are going to be resampled, True if they are not.

**`getDocument(self, ident)`**
> Returns the Document associated with the given ident.

**`getDocumentCount(self)`**
> Number of documents.

**`getGamma(self)`**
> Returns the PriorConcDP for the gamma parameter.

**`getMu(self)`**
> Returns the PriorConcDP for the mu parameter.

**`getOneCluster(self)`**
> Returns False for normal behaviour, True if only one cluster will be used - this forces the algorithm to be normal HDP, with an excess level, rather than dual HDP.

**`getRho(self)`**
> Returns the PriorConcDP for the rho parameter.

**`getSampleCount(self)`**
> Returns the number of samples stored in all the documents contained within.

**`getSeperateClusterConc(self)`**
> True if each cluster has its own seperate concentration parameter, false if they are shared.

**`getSeperateDocumentConc(self)`**
> True if each document has its own concentration parameter, False if they all share a single concentration parameter.

**`getWordCount(self)`**
> Number of words as far as a fitter will be concerned; doesn't mean that they have all actually been sampled within documents however.

**`sampleModel(self, params, callback, mp = True)`**
> Given parameters to run the Gibbs sampling with this does the sampling, and returns the resulting Model object. If params is not provided it uses the default. callback can be a function to report progress, and mp can be set to False if you don't want to make use of multiprocessing.

**`setAlpha(self, alpha, beta, conc)`**
> Sets the concentration details for the per-document DP from which the topics for words are drawn.

**`setBeta(self, beta)`**
> Parameter for the symmetric Dirichlet prior on the multinomial distribution from which words are drawn, one for each topic. (Symmetric therefore a single float as input.)

**`setCalcBeta(self, val)`**
> Leave as False to have beta constant as the algorithm runs, True if you want it recalculated based on the topic multinomials drawn from it. Defaults to False.

**`setCluInstsDNR(self, val)`**
> False to resample the cluster instances, True to not. Defaults to False, but can be set True to save quite a bit of computation. Its debatable if switching this to True causes the results to degrade in any way, but left on by default as indicated in the paper.

**`setDocInstsDNR(self, val)`**
> False to resample the document instances, True to not. Defaults to False, but can be set True to save a bit of computation. Not recomended to be changed as convergance is poor without it.

**`setGamma(self, alpha, beta, conc)`**
> Sets the concentration details for the topic DP, from which topics are drawn

**`setMu(self, alpha, beta, conc)`**
> Sets the concentration details used for the DP from which clusters are drawn for documents.

**`setOneCluster(self, val)`**
> Leave as False to keep the default cluster behaviour, but set to True to only have a single cluster - this results in a HDP implimentation that has an extra pointless layer, making a it a bit inefficient, but not really affecting the results relative to a normal HDP implimentation.

**`setRho(self, alpha, beta, conc)`**
> Sets the concentration details used for each cluster instance.

**`setSeperateClusterConc(self, val)`**
> True if you want clusters to each have their own concentration parameter, False, the default, if you want a single concentration parameter shared between all clusters. Note that setting this True doesn't really work in my experiance.

**`setSeperateDocumentConc(self, val)`**
> True if you want each document to have its own concentration value, False if you want a single value shared between all documents. Experiance shows that the default, False, is the only sensible option most of the time, though when single cluster is on True can give advantages.

**`setWordCount(self, wordCount)`**
> Because the system autodetects words as being the identifiers 0..max where max is the largest identifier seen it is possible for you to tightly pack words but to want to reserve some past the end. Its also possible for a data set to never contain the last word, creating problems. This allows you to set the number of words, forcing the issue. Note that setting the number less than actually exists will be ignored.

## DocSample() ##
> Stores the sample information for a given document - the DP from which topics are drawn and which cluster it is a member of. Also calculates and stores the negative log liklihood of the document.

**`__init__(self, doc)`**
> Given the specific DocState object this copies the relevant information. Note that it doesn't calculate the nll - a method does that. Also supports cloning.

**`calcNLL(self, doc, state)`**
> Calculates the negative log likelihood of the document, given the relevant information. This is the DocState object again (It needs the samples, which are not copied into this object by the constructor.), but this time with the entire state object as well. Probability (Expressed as negative log likelihood.) is specificly calculated using all terms that contain a variable in the document, but none that would be identical for all documents. That is, it contains the probability of the cluster, the probability of the DP given the cluster, and the probability of the samples, which factor in both the drawing of the topic and the drawing of the word. The ordering of the samples is considered irrelevant, with both the topic and word defining uniqueness. Some subtle approximation is made - see if you can spot it in the code!

**`getCluster(self)`**
> Returns the sampled cluster assignment.

**`getInstConc(self)`**
> Returns the sampled concentration parameter, as used by the document DP.

**`getInstCount(self)`**
> Returns the number of cluster instances in the documents model.

**`getInstDual(self)`**
> Returns a 2D numpy array of integers where the first dimension indexes the topic instances for the document and the the second dimension has two entrys, the first (0) the topic index, the second (1) the number of samples assigned to the given topic instance. Do not edit the return value for this method - copy it first.

**`getInstTopic(self, i)`**
> Returns the topic index for the given instance.

**`getInstWeight(self, i)`**
> Returns the number of samples that have been assigned to the given topic instance.

**`getNLL(self)`**
> Returns the negative log liklihood of the document given the model, if it has been calculated.

## Sample() ##
> Stores a single sample drawn from the model - the topics, clusters and each document being sampled over. Stores counts and parameters required to make them into distributions, rather than final distributions. Has clonning capability.

**`__init__(self, state, calcNLL = True)`**
> Given a state this draws a sample from it, as a specific parametrisation of the model.

**`docCount(self)`**
> Returns the number of documents stored within. Should be the same as the corpus from which the sample was drawn.

**`getAlphaPrior(self)`**
> Returns the PriorConcDP that was used for the alpha parameter, which is the concentration parameter for the DP in each document.

**`getBeta(self)`**
> Returns the beta prior, which is a vector representing a Dirichlet distribution from which the multinomials for each topic are drawn, from which words are drawn.

**`getClusterCount(self)`**
> Returns how many clusters there are.

**`getClusterDrawConc(self)`**
> Returns the sampled concentration parameter for drawing cluster instances for documents.

**`getClusterDrawWeight(self, c)`**
> Returns how many times the given cluster has been instanced by a document.

**`getClusterDrawWeights(self)`**
> Returns an array, indexed by cluster id, that contains how many times each cluster has been instanciated by a document. Do not edit the return value - copy it first.

**`getClusterInstConc(self, c)`**
> Returns the sampled concentration that goes with the DP from which members of each documents DP are drawn.

**`getClusterInstCount(self, c)`**
> Returns how many instances of topics exist in the given cluster.

**`getClusterInstDual(self, c)`**
> Returns a 2D array, where the first dimension is indexed by the topic instance, and  the second contains two columns - the first the topic index, the second the weight. Do not edit the return value - copy before use.

**`getClusterInstTopic(self, c, ti)`**
> Returns which topic the given cluster topic instance is an instance of.

**`getClusterInstWeight(self, c, ti)`**
> Returns how many times the given cluster topic instance has been instanced by a documents DP.

**`getDoc(self, d)`**
> Given a document index this returns the appropriate DocSample object. These indices should align up with the document indices in the Corpus from which this Sample was drawn.

**`getGammaPrior(self)`**
> Returns the PriorConcDP that was used for the gamma parameter, which is the concentration parameter for the global DP from which topics are drawn.

**`getMuPrior(self)`**
> Returns the PriorConcDP that was used for the mu parameter, which is the concentration parameter for the Dp from which clusters are drawn.

**`getRhoPrior(self)`**
> Returns the PriorConcDP that was used for the rho parameter, which is the concentration parameter for each specific clusters DP.

**`getTopicConc(self)`**
> Returns the sampled concentration parameter for drawing topic instances from the global DP.

**`getTopicCount(self)`**
> Returns the number of topics in the sample.

**`getTopicMultinomial(self, t)`**
> Returns the calculated multinomial for a given topic ident.

**`getTopicMultinomials(self)`**
> Returns the multinomials for all topics, in a single array - indexed by [topic,word] to give P(word|topic).

**`getTopicUseWeight(self, t)`**
> Returns how many times the given topic has been instanced in a cluster.

**`getTopicUseWeights(self)`**
> Returns an array, indexed by topic id, that contains how many times each topic has been instanciated in a cluster. Do not edit the return value - copy it first.

**`getTopicWordCount(self, t)`**
> Returns the number of samples assigned to each word for the given topic, as an integer numpy array. Do not edit the return value - make a copy first.

**`getTopicWordCounts(self, t)`**
> Returns the number of samples assigned to each word for all topics, indexed [topic,word], as an integer numpy array. Do not edit the return value - make a copy first.

**`getWordCount(self)`**
> Returns the number of words in the topic multinomial.

**`nllAllDocs(self)`**
> Returns the negative log likelihood of all the documents in the sample - a reasonable value to compare various samples with.

## Model() ##
> Simply contains a list of samples taken from the state during Gibbs iterations. Has clonning capability.

**`__init__(self, obj)`**
> None

**`absorbModel(self, model)`**
> Given another model this absorbs all its samples, leaving the given model baren.

**`bestSampleOnly(self)`**
> Calculates the document nll for each sample and prunes all but the one with the highest - very simple way of 'merging' multiple samples together.

**`fitDoc(self, doc, params, callback, mp = True)`**
> Given a document this returns a DocModel calculated by Gibbs sampling the document with the samples in the model as priors. Returns a DocModel. Note that it samples using params for **each** sample in the Model, so you typically want to use less than the defaults in Params, typically only a single run and sample, which is the default. mp can be set to False to force it to avoid multi-processing behaviour.

**`getSample(self, s)`**
> Returns the sample associated with the given index.

**`sampleCount(self)`**
> Returns the number of samples.

**`sampleList(self)`**
> Returns a list of samples, for iterating.

**`sampleState(self, state)`**
> Samples the state, storing the sampled model within.

## DocModel() ##
> A Model that just contains DocSample-s for a single document. Obviously incomplete without a full Model, this is typically used when sampling a document relative to an already trained Model, such that the topic/cluster indices will match up with the original Model. Note that if the document has enough data to justify the creation of an extra topic/cluster then that could exist with an index above the indices of the topics/clusters in the source Model.

**`__init__(self, obj)`**
> Supports cloning.

**`absorbModel(self, dModel)`**
> Absorbs samples from the given DocModel, leaving it baren.

**`addFrom(self, model, index = 0)`**
> Given a model and a document index number extracts all the relevant DocSample-s, adding them to this DocModel. It does not edit the Model but the DocSample-s transfered over are the same instances.

**`getNLL(self)`**
> Returns the average nll of all the contained samples - does a proper mean of the probability of the samples.

**`getSample(self, s)`**
> Returns the sample with the given index, in the range 0..sampleCount()-1

**`sampleCount(self)`**
> Returns the number of samples contained within.

**`sampleList(self)`**
> Returns a list of samples, for iterating.