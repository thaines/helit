A Dirichlet Process Gaussian Mixture Model, implemented using the mean-field variational method.

The main paper describing this method is:
'Variational Inference for Dirichlet Process Mixtures' by D. M. Blei and M. I. Jordan.
but it uses the capping method given by
'Accelerated Variational Dirichlet Process Mixtures' by K. Kurihara, M. Welling and N. Vlassis.
so it can operate incrementally.

Its a non-parametric Bayesian density estimator, using Gaussian kernels - its the best general purpose density estimation method that I know of (In the sense of having an entirely rigorous justification, neat solution method, and parameters that can be mostly ignored.), though it does not scale too well for larger data sets, both in terms of computation and memory.

Implementation is incredibly neat, given how complex the maths behind it is, and also 100% python, for maximum portability (Algorithm makes good use of vectorisation, so no significant speed advantage would be gained from using scipy.weave.). It also only requires a single 300 line file, though does make use of the gcp module.

dpgmm.py - Contains the DPGMM class - pure awesome.

test_1d_1mode.py - Tests it for 1D data with a single mode.
test_1d_2mode.py - Two mode version of the above.
test_1d_3mode.py - Three mode version of the above.
test_grow.py - Tests the ability to grow the number of sticks to model, until the optimal number is found.
test_stick_inc.py - Like the above, but does things another way.

readme.txt - This file, which gets copied into the documentation.
make_doc.py - Creates the html documentation.



