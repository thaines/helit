dp_utils

A selection of C++ code for use in python.weave, all related to Dirichlet processes (Though arguably much more general.), under a BSD license. It specifically includes:

linked_list_cpp.py - A linked list implementation.
funcs_cpp.py - Implementations of ln-gamma, di-gamma and tri-gamma, with testing code.
sampling_cpp.py - Sampling code, for a bunch of probability distributions.
conc_cpp.py - Code to Gibbs sample the DP concentration parameter (Actually in sampling_cpp.py) and a class to represent it.
dir_est_cpp.py - A maximum likelihood procedure for estimating a Dirichlet distribution given the multinomials drawn from it.

dp_utils.py - Provides all of the above, plus a single variable that summarises them all.

readme.txt - This file, which is included in the html documentation.
make_doc.py - This makes the documentation, which is admittedly rather limited - you have to read the code to understand what is going on.

