graph cuts:
-----------

Conversion of my old graph cuts implementation to have a Python interface, albeit incomplete as I only needed to solve binary labelling problems at the time, so didn't bother with alpha expansion. Nothing special.

If you are reading readme.txt then you can generate documentation by running make_doc.py


Contains the following files:

maxflow.py - Provides a max flow implementation.
binary_label.py - Wrapper around above for solving binary labelling problems on nD grids.

test_*.py - Some test scripts, that are also demos of system usage.

readme.txt - This file, which is included in the html documentation.
make_doc.py - Builds the html documentation.
