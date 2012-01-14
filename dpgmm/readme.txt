A Dirichlet Process Gaussian Mixture Model, implemented using the mean-field variational method.

The main paper describing this method is:
'Variational Inference for Dirichlet Process Mixtures' by D. M. Blei and M. I. Jordan.
but it uses the capping method given by
'Accelerated Variational Dirichlet Process Mixtures' by K. Kurihara, M. Welling and N. Vlassis.
so it can operate incrementally.

Its a non-parametric Bayesian density estimator, using Gaussian kernels - its the best general purpose density estimation method that I know of (In the sense of having an entirely rigorous justification, neat solution method, and parameters that can be mostly ignored.), though it does not scale too well for larger data sets, both in terms of computation and memory. It can also be used for clustering.

Implementation is incredibly neat, given how complex the maths behind it is, and also 100% python, for maximum portability (Algorithm makes good use of vectorisation, so no significant speed advantage would be gained from using scipy.weave.). It also only requires a single 400 line file, though does make use of the gcp module.

Typical usage consists of creating the object, providing the number of dimensions of the feature vectors/samples and then using add() to add the said samples. After adding samples setPrior() is called, often with no parameters, and then optionally setConcGamma() can be called - this last method controlls its preference towards having few/lots of clusters. After this a solving method is called - solveGrow() is a good default choice. Note that, due to the somewhat erratic convergance speed no attempt is made to predict run time and provide any kind of progress bar. Once the model is fitted two common modes of usage are as a density estimate and as a clustering algorithm. For density estimation the prob() method will provide the probability of a presented sample being drawn from the model, whilst for clustering the stickProb() method will provide the probability of a provided feature vector having been drawn from each stick (cluster) in the model.


dpgmm.py - Contains the DPGMM class - pure awesome.

test_1d_1mode.py - Tests it for 1D data with a single mode.
test_1d_2mode.py - Two mode version of the above.
test_1d_3mode.py - Three mode version of the above.
test_grow.py - Tests the ability to grow the number of sticks to model, until the optimal number is found.
test_stick_inc.py - Like the above, but does things another way.
test_cluster.py - Tests the models ability to do clustering.

readme.txt - This file, which gets copied into the documentation.
make_doc.py - Creates the html documentation.

