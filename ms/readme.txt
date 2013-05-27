Mean Shift

Primarily provides a mean shift implementation, but also includes kernel density estimation and subspace constrained mean shift using the same object, such that they are all using the same underlying density estimate. Includes multiple spatial indexing schemes and kernel types, including one for directional data. Clustering is supported, with a choice of cluster intersection tests, as well as the ability to interpret exemplar indexing dimensions of the data matrix as extra features, so it can handle the traditional image segmentation scenario.

If you are reading readme.txt then you can generate documentation by running make_doc.py
Note that this module includes a setup.py that allows you to package/install it (The dependency on utils is only for the tests and automatic compilation if you have not installed it - it is not required.)
It is strongly recommended that you look through the various test* files to see examples of how to use the system.


Contains the following key files:

ms.py - The file a user imports - provides a single class - MeanShift.

info.py - File that iterates and prints out information on all the modular components of the system.

test_*.py - Many test scripts.

readme.txt - This file, which is included in the html documentation.
make_doc.py - Builds the html documentation.
setup.py - Allows you to create a package/build/install this module.

