LDA implementation using Gibbs sampling.

Implemented in python using scipy; makes use of scipy.weave if available. (Highly recommended, to the point you would be insane to not. Under Linux this is almost a given if you have gcc installed; under Windows you will probably have to run some installers. If your using a mac give up and get a virtual machine - whilst possible it is a total bitch.)

Based on the ideas from the paper 'Latent Dirichlet Allocation' by D. M. Blei, A. Y. Ng and M. I. Jordan.
Makes use of the solving method from 'Finding scientific topics' by T. L. Griffiths and M. Steyvers

Note: When you take multiple samples from the Gibbs sampler it averages them together, rather than providing the separate models. This is obviously incorrect, but still works perfectly well in practise (And is hardly an uncommon trick.). If you care about having multiple sample then clone the corpus a number of times and do a single run/sample on each of them.


lda.py - packaging file, pulls it all together into a single convenient namespace and chooses the correct solver depending on detected capabilities. The only file you need to import, unless you choose the below version.
lda_nmp.py - same as lda.py except it does not consider the multiprocessing solvers when auto-detecting which solver to use. Useful if your planning to do the multiprocessing yourself, i.e. have lots of solvers running in parallel.

document.py - Document object for building a model, includes its multinomial distribution over Topic-s once solved for.
topic.py - Multinomial distribution for a Topic.
corpus.py - Collection of Document-s; includes the models for the topics if calculated.

solve-shared.py - Stuff shared between all solvers.
solve-python.py - Pure python with scipy solver; really just for testing/verification.
solve-python-mp.py - Effectively solve-python with multiprocessing added in.
solve-weave.py - Implementation with weave to make it go really fast, but with the obvious dependency of a C/C++ compiler working with scipy.weave.
solve-weave-mp.py - Both weave and multiprocess - super fast. Only crazy people fail to use this.

test_tiny.py - test file. Very simple text output of results for 4 words and 2 topics.
test_junction.py - test file. Uses a simulation of traffic at a 4 way junction.
test_grid.py - test file. Uses images for testing, outputs images and requires the opencv library.
test_ap.py - test file. Uses the associated press data set obtainable from http://www.cs.princeton.edu/~blei/lda-c/ , which must be decompressed into a sub-folder with the name 'ap'. Outputs into a test file ap/results-gibbs.txt the top 20 words in 100 topics, exactly as for the original paper.
(The above test files all make good examples of how to use this module.)

make_doc.py - creates/overwrites the documentation file lda.html.
readme.txt - this file, which gets copied into lda.html.
