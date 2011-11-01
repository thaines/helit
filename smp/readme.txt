Sparse Multinomial Posterior

Implements a system to estimate a multinomial given multiple samples drawn from it where those samples are 'sparse', i.e. have had some of the counts thrown away. Actually focuses on obtaining the mean draw for each parameter of the multinomial, as this often allows the draw to be collapsed out in a larger system.

Core is implemented in C++, with a Python wrapper, using scipy.weave.


Files:

smp.py - Contains the python interface to the system.
smp_cpp.py - The C++ code that does the actual work. Designed so it can be used by scipy.weave code for other modules if need be.
flag_index_array.py - A support system; handles the combinations of flags indicating which counts in a draw are known and which are unknown.

test.py - Simple test file.

readme.txt - This file, which is copied into the html documentation.
make_doc.py - Script to generate the html documentation.

