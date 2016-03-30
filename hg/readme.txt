Homography

A simple library of functions for constructing homographies, then distorting images with them. Has the usual set of translate/rotate/scale, which can be combined using ndarray.dot, plus generating the transform for 4 pairs of coordinates. There are then some helper methods for working out the exact size of image required to contain another image after it has been through a given homography. A transform method then allows you to apply a homography (B-Spline interpolation, degree 0-5 inclusive.), though be warned that you give it the homography that converts output coordinates to input coordinates, so you will have to invert a matrix constructed to go the other way.

Also includes some additional methods for querying arbitrary locations in an image with B-Spline interpolation - just made sense to include them here so they can share the B-Spline code. There is also a Gaussian blur implementation (n-dimensional, with support for derivatives and missing data handling) that got shoved in here.

Be warned that homographies are constructed to apply to vectors [x, y, w], to be consistent with everyone else, but then the arrays are indexed [y, x] - this makes things a touch confusing at points. Also, arrays of coordinates are always ordered [y, x], which seemed like a good idea at the time.


Contains the following key files:

hg.py - File for a user to import; just a bunch of functions.

test_*.py - Some test scripts. All take at least an image as input.

readme.txt - This file, which is included in the documentation.
make_doc.py - Builds the documentation.

