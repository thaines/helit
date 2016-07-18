ddp
---

Simple discrete dynamic programming implementation; nothing special. Supports different numbers of labels for each random variable, and has four cost function types, that optimise message passing when possible:

 * different - One cost of they have the same label, another if they have a different label.
 * linear - Cost calculated as a linear function of the label difference.
 * ordered - One cost for same label, another cost for advancing the label by one, infinity for all other options. For when you have an alignment problem.
 * full - Arbitrary cost matrix; expensive as there is no opportunity for optimisation.

If you are reading readme.txt then you can generate documentation by running make_doc.py


Contains the following files:

ddp.py - The file a user imports - provides the DDP class that contains all of the functionality.

info.py - Dynamically generated information about the cost functions.

test_*.py - Some test scripts, that are also demos of system usage.

readme.txt - This file, which is included in the html documentation.
make_doc.py - Builds the html documentation.

