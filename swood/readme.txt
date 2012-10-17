Stochastic Woodlands

Note - this module has been superseded by the 'df' module. It should not be used by new code, and porting old code to the 'df' module might prove advantageous.

An implementation of the method popularised by the paper
'Random Forests' by L. Breiman
which is a refinement, with a decent helping of rigour, of the papers
'Random Decision Forests' by T. K. Ho and
'Shape Quantization and Recognition with Randomized Trees' by Y. Amit and D. Geman.

Alternatively, and mildly comically, named on account of a trademark (If someone could explain the financial basis for getting a trademark on a name that describes a mathematical algorithm, for which no IP protection exists (Theoretically speaking - obviously people abuse the patent system all the time.), to me I might find this a lot less comical.).


A fairly basic implementation - just builds the trees and works out error estimates using the out-of-bag technique so the parameters can be tuned. Does support the weighting of samples for if getting some right is more important than others. The trees constructed are id3, with the c4.5 feature of supporting continuous attributes, in addition to discrete. Previously unseen attribute values are handled by storing the distribution over categories at every node, so that can be returned on an unknown. Final output is always a probability distribution, effectively the normalised number of votes for each category.

Implemented using pure python, with numpy, so not very efficient, but because its such an efficient algorithm anyway its still fast enough for real world use on decently sized data sets. Obviously its quite a simple algorithm, such that anyone with a basic understanding of machine learning should be able to implement it, but it is well commented and the code should be easy to understand.

Contains the following files:

swood.py - Contains the SWood object, that is basically all you need.
dec_tree.py - Contains the DecTree object, that implements a decision tree in case that is all you want. Where most of the systems functionality actually is.
test_*.py - Various test files. Also serve as examples of how to use the system.

make_doc.py - Makes the html documentation.
readme.txt - This file, which is also copied into the html documentation.

