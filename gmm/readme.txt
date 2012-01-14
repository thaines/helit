Gaussian mixture model implementation.

Implemented in python using numpy and scipy.weave.
Only provides an isotropic distribution, but also includes BIC model selection.

Interface consists of several classes - kMeans, KMeansShort, IsotropicGMM. Each provides the methods 'train' and 'getCluster'. Train is given a set of feature vectors and a cluster count, and fits the model. getCluster is given a set of feature vectors and returns the index of the cluster it is most likely from for each feature. saving/loading via pickling is supported, and extra features are provided, such as Bayesian information criterion model selection of cluster counts.

If you are reading readme.txt then you can generate documentation by running make_doc.py



gmm.py - exports all the functionality.
kmeans.py - provides just the k-means classes, if that is all that is required.

kmeans_shared.py - Provides the interface used by all k-means implementations.
kmeans0.py - wrapper around the scipy kmeans implimentation, so it can be used with the other parts of this system.
kmeans1.py - first implementation, brute force with multiple restarts.
kmeans2.py - second implementation, still brute force but instead of multiple restarts uses a scheme of running on multiple small data sets and then initialising with kmeans on the combined clusters from all these runs.
kmeans3.py - third implementation, assumes that distance computations are slow and trys to avoid them by storing information about cluster centre movement. This is both fast and reliable.

mixture.py - provides the interface for mixture models.
isotropic.py - provides the only implementation of the Mixture interface - IsotropicGMM.

bic.py - provides a function to do Bayesian information criterion (BIC) model selection on the number of clusters, given a mixture model.

readme.txt - this file, which is copied into the start of gmm.html if generated.
make_doc.py - creates the gmm.html help file.

test_identical.py - simple test of algorithms on multiple identical clusters.
test_varied.py - slightly more sophisticated test on clusters with different scales and sampling frequencies.
test_selection.py - randomly generates a data set from fairly arbitrary Gaussians, with a random number of them, and uses BIC to select the best IsotropicGMM model.

