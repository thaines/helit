Dirichlet process active learning

Implements the active learning algorithm from the paper
'Active Learning using Dirichlet Processes for Rare Class Discovery and Classification' by T. S. F. Haines and T. Xiang
and also a bunch of other active learning algorithms, most of which are variants.

Contains two classes - one is just a helper for estimating the concentration parameter of a Dirichlet process, the other represents a pool of entities for an active learner to select from. The pool provides many active learning methods, and is designed to interface tightly with a classifier from the p_cat module. Usage is a bit on the non-obvious side - best to look at the test code to see how to use it.

Files included:

dp_al.py - File that imports all parts of the system, to be imported by users.

concentration_dp.py - Contains ConcentrationDP, a class to assist with estimating the concentration parameter of a Dirichlet process.
pool.py - Contains Pool, which you fill with entities to learn from. It then provides various active learning algorithms to select which entity to give to the oracle next. Note that the user is responsible for updating the classifier and interfacing with the oracle.


test_iris.py - Simple visualisation of the P(wrong) algorithm.

test_synth.py - Simple synthetic comparison between all of the active learning algorithms. Potentially a bit misleading due to its synthetic nature.


readme.txt - This file, which is copied into the html documentation.

make_doc.py - Generates the html documentation.

