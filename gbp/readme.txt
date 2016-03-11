Gaussian Belief Propagation

Simple linear solver that has an interface that proves convenient for certain problems. Allows you to construct an arbitrary graph of univariate Gaussian random variables. For each node you can specify a unary term, as a Gaussian over its value. For each edge a pairwise term, as a Gaussian over the offset between them and/or a simple precision between them. It outputs the marginals for each node, the means of which happen to also be the maximum likelihood assignment. Note that the entire system is implemented in terms of precision, which is defined as the inverse of the variance, or the inverse of the standard deviation squared.

It additionally includes a linear solver for symmetric ax=b problems, more as a demonstration than code you would actually use as it only makes sense for sparse problems, and yet does not utilise a sparse matrix class! If your problem is a chain then you're solving a Kalman smoothing problem (or filtering, if you incrementally grow the model and only request the marginal of the last value each time! That's efficient to do btw.). Also contains a script to remove curl from a normal map - good for texture preparation.

This is an implementation of the core technique I used in the paper 'Integrating Stereo with Shape-from-Shading derived Orientation Information', by T. S.F. Haines and R. C. Wilson, but I was not the first to use it. Originally, I implemented it in C++, the code of which is in my PhD code repository - this version is improved and with a Python interface. This reimplementation was done for the paper 'My Text in Your Handwriting', by T. S. F. Haines, O. Mac Aodha and G. J. Brostow.

I did add TRW-S in addition to BP when writing this code - out of curiosity to see if it makes any kind of difference. TRW-S definitely converges faster, which is not surprising. The interesting bit is that for solving linear equations it can solve problems for which BP fails; requirements on the A matrix from Ax=b are slightly reduced in other words. I have never published this result however, as it was just an experimental observation whilst testing, which I didn't have the time to explore further.

If you are reading readme.txt then you can generate documentation by running make_doc.py
Note that this module includes a setup.py that allows you to package/install it (The dependency on utils is only for the tests and automatic compilation if you have not installed it - it is not required.) It is strongly recommended that you look through the various test_*.py files to see examples of how to use the system.


Contains the following key files:

gbp.py - The file a user imports: Provides a single class, GBP.
linear.py - Contains the symmetric ax=b solver.


test_*.py - Some basic test scripts; also good examples of usage.

readme.txt - This file, which is included in the html documentation.
make_doc.py - Builds the html documentation.
setup.py - Allows you to create a package/build/install this module.

