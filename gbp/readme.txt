Gaussian Belief Propagation

Simple linear solver that has an interface that proves convenient for certain problems. Allows you to construct an arbitrary graph of univariate Gaussian random variables. For each node you can specify a unary term, as a Gaussian over its value. For each edge a pairwise term, as a Gaussian over the offset between them. It outputs the marginals for each node, the means of which happen to also be the maximum likelihood assignment. Note that the entire system is implemented in terms of precision, which is defined as the inverse of the variance, or the inverse of the standard deviation squared.

This is an implementation of the core technique I used in the paper 'Integrating Stereo with Shape-from-Shading derived Orientation Information', by Tom S.F. Haines and Richard C. Wilson, but I was not the first to use it. (There is an old C++ implementation in my PhD code repository.)

If you are reading readme.txt then you can generate documentation by running make_doc.py
Note that this module includes a setup.py that allows you to package/install it (The dependency on utils is only for the tests and automatic compilation if you have not installed it - it is not required.) It is strongly recommended that you look through the various test* files to see examples of how to use the system.


Contains the following key files:

gbp.py - The file a user imports - provides a single class - GBP.


test_*.py - Some basic test scripts.

readme.txt - This file, which is included in the html documentation.
make_doc.py - Builds the html documentation.
setup.py - Allows you to create a package/build/install this module.
