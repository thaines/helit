Decision Forests

An implementation of the method popularised by the paper
'Random Forests' by L. Breiman
which is a refinement, with a decent helping of rigour, of the papers
'Random Decision Forests' by T. K. Ho and
'Shape Quantization and Recognition with Randomized Trees' by Y. Amit and D. Geman.
This particular implementation is based on the tutorial from ICCV 2011, which has a corresponding technical report:
'Decision Forests for Classification, Regression, Density Estimation, Manifold Learning and Semi-Supervised Learning' by A. Criminisi, J. Shotton and E. Konukoglu.

Ultimately this is intended to be a fully featured decision forest, but for now it only supports classification and a limited set of test generators. This early version none-the-less contains support for both continuous and discrete features, and incremental learning (Admittedly a primitive form.). Implementation is pure python, using numpy, (Future versions will include inline C++ based optimisations.) and is extremely modular, to allow for future expansion.

Usage typically consist of initially creating a DF object and configuring it by setting its goal (Only classification is currently available.), its generator (A generator provides tests with which to split the data set at a node.), and configuring its Pruner, which defines when to stop growing each tree. Once setup the learn method is called with a set of exemplars which it uses to train the model. After the model has been learned the evaluate method can be applied to new features, to get a probability distribution over class assignment (And optionally a hard assignment.). The included tests demonstrate usage and should help understand usage; if you are interested in incremental learning then test_inc.py is the file to read.

Particular attention must be paid to the generators - it is these that contain all of the really important parameters in the system, and choosing the right configuration will have a massive impact on performance. Consequentially there are a lot of generators, built on different principals and including composite generators to allow generators to be combined to build an infinite selection. This means that you should be able to find a good choice for your data, but also means that finding a good choice might take a certain amount of effort - be prepared to try many combinations.

If you are reading readme.txt then you can generate documentation by running make_doc.py


Contains the following files:

df.py - Primary python file - import this to get all the classes, particularly the DF class, around which all functionality revolves.

exemplars.py - Provides the wrapper around data. All data fed into the system must match the interface provided within, probably by being a bunch of data matrices wrapped by the MatrixFS class. Uses a very specific vocabulary - each data point provided to the system is an exemplar, where all exemplars have a set of shared channels which each contain 1 or more features. Generators create tests for specific channels; also, channels provide auxiliary information, such as the actual answer and weights for the exemplars during learning. Given a specific channel index and a specific feature index within that channel plus a specific exemplar index you get back a single value from the structure. Whilst not provided the capability to inherit from the interface and implement one specific to the problem at hand, by calculating values on the fly for instance, exists.

goals.py - Defines a generic interface for defining the goal of a decision tree. Currently provides the Classification implementation of this interface, for solving the classification problem.

generators.py - Provides the generator interface and the composite generators. The composite generators allow you to select tests from multiple generators - this is most often used to combine a generator for continuous tests and a generator for discrete tests when your data contains both data types.
gen_median.py - Generators that attempt to split the data in half. Generally not a good approach, but for some problems this works very well.
gen_random.py - Generators based on pure randomness - generate enough random tests and something will work. This is very much in the vein of 'Extremely random forests'.
gen_classify.py - Optimising generators specific to the problem of classification. These are the traditional choices, and should be the first approaches tried.

pruners.py - Defines the concept of a Pruner, which decides when to stop growing a tree. Provides a default approach that stops growth when any of several limits is approached.

test.py - Defines the actual tests, and their interface. In practise a user only cares about Generators, as each Generator inherits from the Test relevant to what it returns.
nodes.py - Defines the node of a tree, where all the magic in fact happens. Its completely internal to the system however, and never used directly.

test_discrete.py - Tests the discrete generators on some simulated discrete data.
test_continuous.py - Tests the continuous generators on some simulated continuous data.
test_mix.py - Tests the composite generators on a simulated problem that contains both discrete and continuous data.
test_inc.py - Tests incremental learning, running batch on the same data so performance can be compared.

readme.txt - This file, which is included in the html documentation.
make_doc.py - Builds the html documentation.

