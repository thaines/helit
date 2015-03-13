# Region Latent Dirichlet Allocation #

## Overview ##
**RLDA implementation**

The code for the paper 'Video Topic Modelling with Behavioural Segmentation' by T. S. F. Haines and T. Xiang.

Theoretically performs a behavioural segmentation whilst constructing a topic model, in a single formulation. In practise its better described as a topic model with a slightly more complicated topic distribution, with some crazy interactions. Whilst the idea is sound this model and implementation do not perform as well as I would of liked (Tested on traffic cctv data.) - its better than straight LDA, but not by much (Mostly because it has better generalisation to larger regions, i.e. its less sensitive to vehicles that are large or driving in the right way but slightly out of position.).

Implemented using Gibbs sampling, using python with scipy, including scipy.weave. All of the tests use open cv for visualisation. Interface is extremely similar to my LDA implementation.

`rlda.py` - pulls together everything into a single convenient namespace, and handles details with regards to selecting a solver.


`document.py` - contains the data structure to represent a document.

`corpus.py` - defines a corpus, as a set of documents and a model if it has been solved.

`params.py` - the object that represents the parameters for running the algorithm.


`solve_shared.py` - The data structure and other shared stuff used by all solving methods (Even though their is really only one.).

`solve_weave.py` - The single process solver.

`solve_weave_mp.py` - The multi-process solver.


`test_cross.py` - A very simple test case. Probably a good reference to figuring out how to use the implementation.

`test_cross_dual.py` - A variant of the simple test case.

`test_cross_fat.py` - Another variant of the simple test case.

`test_junction.py` - More sophisticated test that simulates a traffic junction.


`readme.txt` - This file, which is also copied into the html documentation.

`make_doc.py` - The file that generates the html documentation.

---


# Functions #

**`getAlgorithm()`**
> Returns a text string indicating which implimentation of the fitting algorithm is being used.


# Classes #

## Document() ##
> A document, consists of a list of all the word/identifier pairs in the document.

**`__init__(self, dic)`**
> Constructs a document given a dictionary dic[(identifier num, word num)] = count, where identifier num is the natural number that indicates which identifier, and word num the natural number which indicates which word. Count is how many times that identifier-word pair exist in the document. Excluded entries are effectivly assumed to have a count of zero.

**`fit(self, ir, wrt, params = <params.Params instance at 0x19a28c0>, alpha = 1.0, norm = 100.0)`**
> Given the model provided by a corpus (ir and wrt.) this fits the documents model, independent of the corpus itself. Uses Gibbs sampling as you would expect.

**`getDic(self)`**
> Returns a dictionary object that represents the document, basically a recreated version of the dictionary handed in to the constructor.

**`getMaxIdentNum(self)`**
> Returns the largest ident number it has seen.

**`getMaxWordNum(self)`**
> Returns the largest word number it has seen.

**`getModel(self)`**
> Returns the vector defining the multinomial from which topics are drawn, P(topic), if it has been calculated, or None if it hasn't.

**`getNum(self)`**
> Number - just the offset into the array in the corpus where this document is stored, or None if its yet to be stored anywhere.

**`getSampleCount(self)`**
> Returns the number of identifier-word pairs in the document, counting duplicates.

**`getWords(self)`**
> Returns an array of all the words in the document, row per word with the columns [identifier,word,count].

**`negLogLikeRegion(self, region)`**
> Returns the negative log likelihood of the words being drawn in the region, sampled and calculated during a call of fit. Do not call if fit has not been run, noting that fitting an entire corpus does not count.

**`negLogLikeRegionAlt(self, region, ir, wrt, sampleCount = 64)`**
> Returns the negative log likelihood of the given region, alternate calculation - designed to decide if a region is being normal or not. Assuming that the documents model and the given model are all correct, rather than averages of samples from a distribution. This is obviously incorrect, but gives a good enough approximation for most uses. Has to use sampling in part, hence the sampleCount parameter.

**`negLogLikeRegionVec(self)`**
> Returns a vector of negative log likelihhods for each region in the document.

**`probTopic(self, topic)`**
> Returns the probability of the document emitting the given topic, where topics are represented by their ident. Do not call if model not calculated.

**`regionSize(self, region)`**
> Returns the average size of the region, as sampled when sampling the region probabilities.

**`regionSizeVec(self)`**
> Returns a vector of the average size of each region, as sampled when sampling the region probabilities.

**`setModel(self, model)`**
> Sets the model for the document. For internal use only really.

## Corpus() ##
> Defines a corpus, i.e. the input to the rLDA algorithm. Consists of documents, identifiers and words, plus counts of how many regions and topics a fitted model should have. Has a method to fit a model, after which you can retrieve the models parameters.

**`__init__(self, regions, topics)`**
> Construct a corpus - you are required to provide the number of regions and topics to be used by the fitting model.

**`add(self, doc)`**
> Adds a document to the corpus.

**`documentList(self)`**
> Returns a list of all documents.

**`fit(self, params = <params.Params instance at 0x1f7cb48>, callback)`**
> Fits a model to this Corpus.

**`getAlpha(self)`**
> Returns the current alpha value.

**`getBeta(self)`**
> Returns the current beta value.

**`getGamma(self)`**
> Returns the current gamma value.

**`getIR(self)`**
> Returns an unnormalised multinomial distribution indexed by [identifier,region]

**`getMaxIdentNum(self)`**
> Returns the largest ident number it has seen.

**`getMaxWordNum(self)`**
> Returns the largest word number it has seen.

**`getRegionCount(self)`**
> Returns the number of regions that will be used.

**`getSampleCount(self)`**
> Returns the number of identifier-word pairs in all the documents, counting duplicates.

**`getTopicCount(self)`**
> Returns the number of topics that will be used.

**`getWRT(self)`**
> Returns an unnormalised multinomial distribution indexed by [word,region,topic]

**`setAlpha(self, alpha)`**
> Sets the alpha value - 1 is more often than not a good value, and is the default.

**`setBeta(self, beta)`**
> Sets the beta value. Defaults to 1.0.

**`setGamma(self, gamma)`**
> Sets the gamma value. Defaults to 1.0. One will note that it doesn't actually get used in the formulation, so in a slight abuse it is used in place of beta during the r-step - this provides a touch more control to the user.

**`setIdentWordCounts(self, identCount, wordCount)`**
> Because the system autodetects identifiers and words as being the range 0..max where max is the largest number seen it is possible for you to tightly pack words but to want to reserve some past the end. Its also possible for a data set to never contain the last entity, creating problems. This allows you to set the numbers, forcing the issue. Note that setting the number less than actually exist is a guaranteed crash, at a later time.

**`setModel(self, wrt, ir)`**
> Sets the model, in terms of the wrt and ir count matrices. For internal use only really.

**`setRegionTopicCounts(self, regions, topics)`**
> Sets the number of regions and topics. Note that this will reset the model, so after doing this all the model variables will be None.

## Params() ##
> Parameters for running the fitter that are universal to all fitters.

**`__init__(self)`**
> None

**`fromArgs(self, args, prefix = )`**
> Extracts from an arg string, typically sys.argv[1:], the parameters, leaving them untouched if not given. Uses --runs, --samples, --burnIn, --lag, --iterT and --iterR. Can optionally provide a prefix which is inserted after the '--'

**`getBurnIn(self)`**
> Returns the burn in length.

**`getIterR(self)`**
> Return the number of r iterations.

**`getIterT(self)`**
> Return the number of t iterations.

**`getLag(self)`**
> Returns the lag length.

**`getRuns(self)`**
> Returns the number of runs.

**`getSamples(self)`**
> Returns the number of samples.

**`setBurnIn(self, burnIn)`**
> Number of Gibbs iterations to do for burn in before sampling starts.

**`setIterR(self, iterR)`**
> Number of iterations of updating r to do for each inner loop.

**`setIterT(self, iterT)`**
> Number of iterations of updating t to do for each inner loop.

**`setLag(self, lag)`**
> Number of Gibbs iterations to do between samples.

**`setRuns(self, runs)`**
> Sets the number of runs, i.e. how many seperate chains are run.

**`setSamples(self, samples)`**
> Number of samples to extract from each chain - total number of samples going into the final estimate will then be sampels\*runs.