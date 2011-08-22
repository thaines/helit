Probabilistic Classifiers

A simple set of classifiers, all working on the principal of building a density estimate for each category and then using Bayes rule to work out the probability of belonging to each, before finally selecting the category the sample is most likely to belong to (The word category is used rather than class to avoid the problem of class being a keyword in Python and many other languages.). Whilst a discriminative model will typically get better results (e.g. the support vector machine (svm) model that is also available.) the advantage of getting a probability distribution is not to be sniffed at. Features such as incremental learning and estimating the odds that the sample belongs to none of the available categories are also included.

Provides a standard-ish interface for a classifier, and then 3 actual implementations, using 3 different density estimation methods. The methods range in complexity and speed - the Gaussian method is obviously very simple, but also extremely fast, and has proper prior handling. The KDE (Kernel density estimation) method is reasonably fast, but requires some parameter tuning. The DPGMM (Dirichlet process Gaussian mixture model.) method is defiantly the best, but extremely expensive to run.

Requires the relevant density estimation modules be available to run, though the header (p_cat.py) that you include is coded to only load models that are available, so you can chop things down to size if desired. The entire module, and its dependencies, are coded in Python (As of this minute - its likely I might optionally accelerate them via scipy.weave in the future.), making it easy to get running. Its actually very little code, as most of the code is in the density estimation modules on which it depends.


p_cat.py - The file that conveniently provides everything.

prob_cat.py - The standard interface that all classifiers implement.

classify_gaussian.py - Contains ClassifyGaussian, which uses a Gaussian for each category, albeit in a fully Bayesian manor.
classify_kde.py - Contains ClassifyKDE, which models its categories using a kernel density estimate.
classify_dpgmm.py - Contains ClassifyDPGMM, which uses the heavy weight that is the Dirichlet process gaussian mixture model for its density estimates.

test_iris.py - Test file, to verify it all works.

make_doc.py - Generates the HTML documentation.
readme.txt - This file, which gets copied into the HTML documentation.

