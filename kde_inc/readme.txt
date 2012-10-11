Incremental Kernel Density Estimation

An implementation of the method from the paper 'Incremental One-Class learning with Bounded Computational Complexity', by R. R. Sillito and R. B. Fisher.

loo_cov.py - Code to calculate an optimal covariance matrix from a bunch of samples using leave one out.
gmm.py - Gaussian mixture model representation, used internally.
kde_inc.py - Wrapper for GMM that adds the ability to update the model with new samples as they come in, using Fishers incremental technique.

test_1d_beta.py - Simple test of KDE_INC class.
test_loo.py - Simple test of the loo class.

make_doc.py - Generates the help file.
readme.txt - This file, which gets copied into the html documentation.

