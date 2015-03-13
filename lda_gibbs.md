# Latent Dirichlet Allocation (Gibbs) #

## Overview ##
**LDA implementation using Gibbs sampling.**

Implemented in python using scipy; makes use of scipy.weave if available. (Highly recommended, to the point you would be insane to not. Under Linux this is almost a given if you have gcc installed; under Windows you will probably have to run some installers. If your using a mac give up and get a virtual machine - whilst possible it is a total bitch.)

Based on the ideas from the paper 'Latent Dirichlet Allocation' by D. M. Blei, A. Y. Ng and M. I. Jordan.
Makes use of the solving method from 'Finding scientific topics' by T. L. Griffiths and M. Steyvers

Note: When you take multiple samples from the Gibbs sampler it averages them together, rather than providing the separate models. This is obviously incorrect, but still works perfectly well in practise (And is hardly an uncommon trick.). If you care about having multiple sample then clone the corpus a number of times and do a single run/sample on each of them.


`lda.py` - packaging file, pulls it all together into a single convenient namespace and chooses the correct solver depending on detected capabilities. The only file you need to import, unless you choose the below version.

`lda_nmp.py` - same as lda.py except it does not consider the multiprocessing solvers when auto-detecting which solver to use. Useful if your planning to do the multiprocessing yourself, i.e. have lots of solvers running in parallel.


`document.py` - Document object for building a model, includes its multinomial distribution over Topic-s once solved for.

`topic.py` - Multinomial distribution for a Topic.

`corpus.py` - Collection of Document-s; includes the models for the topics if calculated.


`solve-shared.py` - Stuff shared between all solvers.

`solve-python.py` - Pure python with scipy solver; really just for testing/verification.

`solve-python-mp.py` - Effectively solve-python with multiprocessing added in.

`solve-weave.py` - Implementation with weave to make it go really fast, but with the obvious dependency of a C/C++ compiler working with scipy.weave.

`solve-weave-mp.py` - Both weave and multiprocess - super fast. Only crazy people fail to use this.


`test_tiny.py` - test file. Very simple text output of results for 4 words and 2 topics.

`test_junction.py` - test file. Uses a simulation of traffic at a 4 way junction.

`test_grid.py` - test file. Uses images for testing, outputs images and requires the opencv library.

`test_ap.py` - test file. Uses the associated press data set obtainable from http://www.cs.princeton.edu/~blei/lda-c/ , which must be decompressed into a sub-folder with the name 'ap'. Outputs into a test file ap/results-gibbs.txt the top 20 words in 100 topics, exactly as for the original paper.

(The above test files all make good examples of how to use this module.)

`make_doc.py` - creates/overwrites the documentation file lda.html.

`readme.txt` - this file, which gets copied into lda.html.

---


# Functions #

**`getAlgorithm()`**
> Returns a text string indicating which implimentation of the fitting algorithm is being used.


# Classes #

## Document() ##
> Representation of a document used by the system. Consists of two parts: a) A list of words; each is referenced by a natural number and is associated with a count of how many of that particular word exist in the document. Stored in a matrix. b) The vector parameterising the multinomial distribution from which topics are drawn for the document, if this has been calculated.

**`__init__(self, dic)`**
> Constructs a document given a dictionary dic[ident](ident.md) = count, where ident is the natural number that indicates which word and count is how many times that word exists in the document. Excluded entries are effectivly assumed to have a count of zero. Note that the solver will construct an array 0..{max word ident} and assume all words in that range exist, going so far as smoothing in words that are never actually seen.

**`dupWords(self)`**
> Returns the number of words in the document, counting duplicates.

**`fit(self, topicsWords, alpha = 1.0, params = <solve_shared.Params instance at 0x253fa70>)`**
> Calculates a model for this document given a topics-words array, alpha value and a Params object. Note that the topic-words array is technically a point approximation of what is really a prior over a multinomial distribution, so this is not technically correct, but it is good enough for most purposes.

**`getDic(self)`**
> Returns a dictionary object that represents the document, basically a recreated version of the dictionary handed in to the constructor.

**`getIdent(self)`**
> Ident - just the offset into the array in the corpus where this document is stored, or None if its yet to be stored anywhere.

**`getModel(self)`**
> Returns the vector defining the multinomial from which topics are drawn, P(topic), if it has been calculated, or None if it hasn't.

**`getTopic(self)`**
> Returns the pre-assigned topic, as in integer offset into the topic list, or None if not set.

**`getWord(self, word)`**
> Given an index 0..uniqueWords()-1 this returns the tuple (ident,count) for that word.

**`negLogLikelihood(self, topicsWords)`**
> Returns the negative log likelihood of the document given a topics-words array. (This document can be in the corpus that generated the list or not, just as long as it has a valid model. Can use fit if need be.) Ignores the priors given by alpha and beta - just the probability of the words given the topic multinomials and the documents multinomial. Note that it is assuming that the topics-words array and document model are both exactly right, rather than averages of samples taken from the distribution over these parameters, i.e. this is not corrrect, but is generally a good enough approximation.

**`probTopic(self, topic)`**
> Returns the probability of the document emitting the given topic, where topics are represented by their ident. Do not call if model not calculated.

**`setModel(self, model)`**
> None

**`setTopic(self, topic)`**
> Allows you to 'set the topic' for the document, which is by default not set. This simply results in an increase in the relevant entry of the prior dirichlet distribution, the size of which is decided by a parameter in the Corpus object. The purpose of this is to allow (semi/weakly-) supervised classification problems to be done, rather than just unsupervised. Defaults to None, which is no topic bias. This is of course not really setting - it is only a prior, and the algorithm could disagree with you. This is arguably an advantage, for if there are mistakes in your trainning set. Note that this is only used for trainning a complete topic model - for fitting a document to an existing model this is ignored. The input should be None to unset (The default) or an integer offset into the topic list.

**`uniqueWords(self)`**
> Returns the number of unique words in the document, i.e. not counting duplicates.

## Topic() ##
> Simple wrapper class for a topic - contains just the parameter vector for the multinomial distribution from which words in that topic are drawn from. The index into the vector is the ident of the word associated with each specific probability value.

**`__init__(self, ident)`**
> Initialises the model to be None, so it can be later calculated. ident is the offset of this topic into the Corpus in which this topic is stored. Only Corpus-s should initialise this object and hence should know.

**`getIdent(self)`**
> Ident - just the offset into the array in the Corpus where this topic is stored.

**`getModel(self)`**
> Returns the unnormalised parameter vector for the multinomial distribution from which words generated by the topic are drawn. (The probabilities are actually a list of P(topic,word) for this topic, noting that there are all the other topics. You may normalise it to get P(word|topic), or take the other vectors and manipulate them to get all the relevant distributions, i.e. P(topic|word), P(topic), P(word). )

**`getNormModel(self)`**
> Returns the model but normalised so it is the multinomial P(word|topic).

**`getTopWords(self)`**
> Returns an array of word identifiers ordered by the probability of the topic emitting them.

**`probWord(self, ident)`**
> Returns the probability of the topic emitting the given word. Only call if the model has been calculated.

**`setModel(self, model)`**
> None

## Corpus() ##
> Contains a set of Document-s and a set of Topic-s associated with those Document-s. Also stores the alpha and beta parameters associated with the model.

**`__init__(self, topicCount)`**
> Basic setup, only input is the number of topics. Chooses default values for alpha and beta which you can change later before fitting the model.

**`add(self, doc)`**
> Adds a document to the corpus.

**`documentCount(self)`**
> Number of documents.

**`documentList(self)`**
> Returns a list of all documents.

**`fit(self, params = <solve_shared.Params instance at 0x256b638>, callback)`**
> Fits a model to this Corpus. params is a Params object from solve-shared. callback if provided should take two numbers - the first is the number of iterations done, the second the number of iterations that need to be done; used to report progress. Note that it will probably not be called for every iteration for reasons of efficiency.

**`getAlpha(self)`**
> Returns the current alpha value.

**`getAlphaMult(self)`**
> Returns the current alpha multiplier.

**`getBeta(self)`**
> Returns the current beta value.

**`getDocument(self, ident)`**
> Returns the Document associated with the given ident.

**`getTopic(self, ident)`**
> Returns the Topic associated with the given ident.

**`maxDocumentIdent(self)`**
> Returns the highest ident; documents will then be found in the range {0..max ident}. Returns -1 if no documents exist.

**`maxTopicIdent(self)`**
> Returns the highest ident; topics will then be found in the range {0..max ident}. Returns -1 if no topics exist.

**`maxWordIdent(self)`**
> Returns the maximum word ident currently in the system; note that unlike Topic-s and Document-s this can have gaps in as its user set. Only a crazy user would do that though as it affects the result due to the system presuming that the gap words exist.

**`setAlpha(self, alpha)`**
> Sets the alpha value - 1 is more often than not a good value, and is the default.

**`setAlphaMult(self, alphaMult)`**
> Sets a multiplier of the alpha parameter used when the topic of a document is given - for increasing the prior for a given entry - can be used for semi-supervised classification. Defaults to a factor of 10.0

**`setBeta(self, beta)`**
> The authors of the paper observe that this is effectivly a scale parameter - use a low value to get a fine grained division into topics, or a high value to get just a few topics. Defaults to 1.0, which is a good number for most situations.

**`setTopicCount(self, topicCount)`**
> Sets the number of topics. Note that this will reset the model, so after doing this all the model variables will be None etc.

**`setWordCount(self, wordCount)`**
> Because the system autodetects words as being the identifiers 0..max where max is the largest identifier seen it is possible for you to tightly pack words but to want to reserve some past the end. Its also possible for a data set to never contain the last word, creating problems. This allows you to set the number of words, forcing the issue. Note that setting the number less than actually exist is a guaranteed crash, at a later time.

**`topicCount(self)`**
> Number of topics.

**`topicList(self)`**
> Returns a list of all topics.

**`topicsWords(self)`**
> Constructs and returns a topics X words array that represents the learned models key part. Simply an array topics X words of P(topic,word). This is the data best saved for analysing future data - you can use the numpy.save/.load functions. Note that you often want P(word|topic), which you can obtain by normalising the rows - (a.T/a.sum(axis=1)).T

**`totalWordCount(self)`**
> Returns the total number of words used by all the Document-s - is used by the solver, but may be of interest to curious users.

**`wordCount(self)`**
> Number of words as far as a fitter will be concerned; doesn't mean that they all actually exist however.

## Params() ##
> Parameters for running the fitter that are universal to all fitters.

**`__init__(self)`**
> None

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
> Sets the number fom runs, i.e. how many seperate chains are run.

**`setSamples(self, samples)`**
> Number of samples to extract from each chain - total number of samples going into the final estimate will then be sampels\*runs.