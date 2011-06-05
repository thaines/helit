Gaussian Conjugate Prior

Basic library for handling the conjugate prior to the Gaussian. Contains Gaussian, Wishart and Student-T distributions, alongside a class to represent the conjugate prior, with all the relevant operations. Also includes an incremental Gaussian calculator. Lacking optimisation this is more for demonstration/education or for when speed just does not matter (Or there are lots of dimensions, so the python costs become insignificant compared to the actual operations, which are all done by scipy.) - implemented using pure python with scipy. The test scripts provide good examples of use.


gcp.py - Helper file that pulls all the parts of the system into a single namespace.

gaussian.py - Contains the Gaussian distributions class.
gaussian_inc.py - Contains GaussianInc, an incremental class for which you provide samples, from which at any point a Gaussian object can be extracted.
wishart.py - The Wishart distribution.
student_t.py - The student T distribution, provided as it is the distribution obtained when you integrate out a draw from the Gaussians conjugate prior.
gaussian_prior.py - The point of the entire system - provides a conjugate prior to the Gaussian, over both mean and covariance, with lots of methods to update it, draw from it or integrate out draws from it etc.

test_1d.py - Simple 1D test of the system. Uses open cv for its output.
test_2d.py - Simple 2D test of the system. Uses open cv for its output.
test_inc.py - Tests that incrementally adding and removing samples works.
test_weighted.py - Tests that it can handle weighted samples.

readme.txt - This file, which is copied into the automatically generated documentation.
make_doc.py - creates the gcp.html help file.

