# Dirichlet Process Utilities #

## Overview ##
**dp\_utils**

A selection of C++ code for use in python.weave, all related to Dirichlet processes (Though arguably much more general.), under a BSD license. It specifically includes:

`linked_list_cpp.py` - A linked list implementation.

`funcs_cpp.py` - Implementations of ln-gamma, di-gamma and tri-gamma, with testing code.

`sampling_cpp.py` - Sampling code, for a bunch of probability distributions.

`conc_cpp.py` - Code to Gibbs sample the DP concentration parameter (Actually in sampling\_cpp.py) and a class to represent it.

`dir_est_cpp.py` - A maximum likelihood procedure for estimating a Dirichlet distribution given the multinomials drawn from it.


`dp_utils.py` - Provides all of the above, plus a single variable that summarises them all.


`readme.txt` - This file, which is included in the html documentation.

`make_doc.py` - This makes the documentation, which is admittedly rather limited - you have to read the code to understand what is going on.


---


# Variables #

**`sampling_code`**
> Code for sampling from various distributions - uniform, Gaussian, gamma and beta.

**`conc_code`**
> Contains code to sample a concentration parameter and two classes - one to represent the status of a concentration parameter - its prior and its estimated value, and another to do the same thing for when a concentration parameter is shared between multiple Dirichlet processes.

**`dir_est_code`**
> Contains a class for doing maximum likelihood estimation of a Dirichlet distrbution given multinomials that have been drawn from it.

**`linked_list_code`**
> A linked list implimentation - doubly linked, adds data via templated inheritance.

**`linked_list_gc_code`**
> A linked list with reference counting and garabge collection for its entries. Happens to be very good at representing a Dirichlet process.

**`dp_utils_code`**
> Combines all of the code provided in this module into a single variable.