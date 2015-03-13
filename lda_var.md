# Latent Dirichlet Allocation (variational) #

## Overview ##
**A variational LDA implementation, using the global variational method.**

Fairly standard really, though it should be noted that it uses a different solving method to the original LDA paper ('Latent Dirichlet Allocation', by D. M. Blei, A. Y. Ng and M. I. Jordan.). It also has a proper Dirichlet prior over beta, to smooth in unseen words.

Implementation is all in straight python, so not as fast as it could be, but still pretty decent due to vectorisation. Makes use of scipy, and both tests use open cv. Unlike the Gibbs implementation it does not support multiprocessing, it also does not have progress bar support, but then its a lot faster, so these features are not as necessary.


The graphical model includes the current variables:

alpha - Known prior on theta, a Dirichlet distribution.
theta\_d - The multinomial over topics for each document. Has a prior alpha; and the z values are drawn from this.
z\_dn - Label assigned to each word, indicating which topic it was drawn from. Drawn from the documents theta, indexes the beta from which w is drawn.
w\_dn - The known words, i.e. the data that is actually provided to the algorithm. Many of these for each document. Works are drawn from the beta distribution associated with the topic for that word, which is drawn from the documents theta.
beta\_t - One for each topic - a multinomial from which words are drawn from.
gamma - Fixed prior over beta, a Dirichlet distribution.

d - subscript for document.
t - subscript for topic.
n - subscript for word within a document.


The distribution can be written:

P(theta,z,w,beta) = P(theta;alpha) P(z|theta) P(w|z,beta) P(beta;gamma)

and the variational factorisation approximation is:

q(theta,z,beta) = q(theta) q(z) q(beta)


The files included are:

`lda.py` - Everything basically - implementation and interface. Its a lot simpler than the Gibbs version!


`test_grid.py` - A test program.

`test_junction.py` - Another test program.


`readme.txt` - This file, which is copied into lda\_var.html when generated.

`make_doc.py` - Generates the documentation.


---


# Classes #

## VLDA() ##
> A variational implimentation of LDA - everything is in a single class. Has an extensive feature set, and is capable of more than simply fitting a model.

**`__init__(self, topics, words)`**
> Initialises the object - you provide the number of topics and the number of words.

**`add(self, dic)`**
> Adds a document. Given a dictionary indexed by word identifier (An integer [0,wordCount-1]) that leads to a count of how many of the given words exist in the document. Omitted words are assumed to have no instances. Returns an identifier (An integer) that can be used to request information about the document at a later stage.

**`clone(self)`**
> Returns a copy of this object.

**`docCount(self)`**
> Returns the number of documents in the system.

**`getAlpha(self)`**
> Returns the current alpha value, the parameters for the DP prior, as a numpy array. This is the prior over the per-document multinomial from which topics are drawn.

**`getBeta(self, topic)`**
> Returns the parameters for the Dirichlet distribution over the beta multinomial for the given topic, i.e. the DP from which the multinomial that words for the topic are drawn from.

**`getDelta(self)`**
> Returns the maximum change seen for any z multinomial in the most recent iteration. Useful if using maxIter to see if it has got close enough.

**`getDoc(self, doc)`**
> Given a document identifier returns a reconstruction of the dictionary originally provided to the add method.

**`getGamma(self)`**
> Returns the current gamma value, the parameters for the DP prior, as a numpy array. This is the prior over the per-topic multinomial from which words are drawn.

**`getNLL(self, doc)`**
> Given a document identifier this returns the probability of the document, given the model that has been fitted. Specifically, it returns the negative log likelyhood of the words given the model that has been fitted to it, using the expected values for theta and beta. Returns None if the value can't be calculated, i.e. solve needs to be called.

**`getNewNLL(self, dic, lock = True)`**
> A helper method, that replicates getNLL but for a document that is not in the corpus. Takes as input a dictionary, as you would provide to the add method - it then clones self, adds the document, solves the model and return getNLL. If lock is True, the default, it locks all the existing model parameters, which is computationally useful, but if its False it lets them all change.

**`getTheta(self, doc)`**
> Returns the parameter vector for the DP over the theta variable associated with the requested document, i.e. the DP from which the per-document multinomial over topics is drawn. Returns None if it has not been calculated.

**`getZ(self, doc)`**
> Returns for the provided document a dictionary that is indexed by word index and obtains multinomials over the value of Z for the words with that value in the document. See getDoc for how many times you would need to draw from each distribution. Will only include multinomials for words that exist in the document (There is a hack involving putting a word count of zero in the input to get other words however.). Returns None if it has not been calculated.

**`lockAllDoc(self, lock = True)`**
> Same as lockDoc, but for all documents.

**`lockBeta(self, lock = True)`**
> Locks, or unlocks, the beta parameter from being updated during a solve - can be useful for repeated calls to solve, to effectivly lock the model and analyse new documents

**`lockDoc(self, doc, lock = True)`**
> Given a document identifier this locks it, or unlocks it if lock==False - can be used for repeated calls to solve to reduce computation if desired.

**`numTopics(self)`**
> Returns the number of topics - idents are therefore all natural numbers less than this.

**`numWords(self)`**
> Returns the number of words - idents are therefore all natural numbers less than this.

**`rem(self, doc)`**
> Removes a document, as identified by its identifier returned by add. Note that this results in some memory efficiency, though it will use newly added documents to fill the gap.

**`setAlpha(self, alpha)`**
> Sets the alpha prior - you should use either a complete numpy array, of length topics, or a single value, which will be replicated for all values of the DP.

**`setGamma(self, gamma)`**
> Sets the gamma prior - you should use either a complete numpy array, of length words, or a single value, which will be replicated for all values of the DP.

**`setThreshold(self, epsilon)`**
> Sets the threshold for parameter change below which it considers it to have converged, and stops iterating.

**`solve(self, maxIter)`**
> Solves the model, such that you can query all the model details. Returns how many passes it took to acheive convergance. You can optionally set maxIter, to avoid it running forever. Due to its incrimental nature repeated calls with maxIter whilst keeping an eye on the delta can be used to present progress.

**`solveHuman(self, step = 32)`**
> Does the exact same thing as solve except it prints out status reports and allows you to hit 'd' to cause it to exit - essentially an interactive version that reports progress so a human can decide to break early if need be. Uses curses.