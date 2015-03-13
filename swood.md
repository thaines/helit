# Stochastic Woodland (Depreciated) #

## Overview ##
**Stochastic Woodlands**

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

`swood.py` - Contains the SWood object, that is basically all you need.

`dec_tree.py` - Contains the DecTree object, that implements a decision tree in case that is all you want. Where most of the systems functionality actually is.

`test_*.py` - Various test files. Also serve as examples of how to use the system.


`make_doc.py` - Makes the html documentation.

`readme.txt` - This file, which is also copied into the html documentation.


---


# Classes #

## SWood() ##
> A stochastic woodland implimentation (Usually called a random forest:-P). Nothing that fancy - does classification and calculates/provides the out-of-bag error estimate so you can tune the parameters.

**`__init__(self, int_dm, real_dm, cat, tree_count = 128, option_count = 3, minimum_size = 1, weight, index, callback, compress = False)`**
> Constructs and trains the stochastic wood - basically all its doing is constructing lots of trees, each with a different bootstrap sample of the input and calculating the out-of-bound error estimates. The parameters are as follows: int\_dm & real\_dm - the data matrices, one for discrete attributes and one for continuous; you can set one to None if there are none of that kind. cat - The category vector, aligned with the data matrices, where each category is represented by an integer. tree\_count - The number of decision trees to create. option\_count - The number of attributes to consider at each level of the decision trees - maps to the rand parameter of the DecTree class. minimum\_size - Nodes in the trees do not suffer further splits once they are this size or smaller. weight - Optionally allows you to weight the trainning examples, aligned with data matrices. index - Using this you can optionally tell it which examples to use from the other matrices/vectors, and/or duplicate examples. callback - An optional function of the form (steps done,steps overall) used to report progress during construction. compress - if True trees are stored pickled and compressed, in a bid to make them consume less memory - this will obviously destroy classification performance unless multi\_classify is used with suitably large blocks. Allows the algorithm to be run with larger quantities of data, but only use as a last resort.

**`classify(self, int_vec, real_vec)`**
> Classifies an example, given the discrete and continuous feature vectors. Returns a dictionary indexed by categories that goes to the probability of that category being assigned; categories can be excluded, implying they have a value of one, but the returned value is actually a default dict setup to return 0.0 when you request an unrecognised key. The probabilities will of course sum to 1.

**`multi_classify(self, int_dm, real_dm, callback)`**
> Identical to classify, except you give it a data matrix and it classifies each entry in turn, returning a list of distributions. Note that in the cass of using the compressed version of this class using this is essential to be computationally reasonable.

**`oob_success(self)`**
> Returns the success rate ([0.0,1.0], more is better.) for the tree that has been trained. Calculated using the out-of-bag techneque, and primarilly exists so you can run with multiple values of option\_count to find the best parameter, or see the effect of tree\_count.

**`tree_list(self)`**
> Returns a list of all the decision trees in the woodland. Note that when in compressed mode these will be strings of bzip2 compressed and pickled trees, which can be resurected using pickle.loads(bz2.decompress(<>)).

## DecTree() ##
> A decision tree, uses id3 with the c4.5 extension for continuous attributes. Fairly basic - always grows fully and stores a distribution of children at every node so it can fallback for previously unseen attribute categories. Allows the trainning vectors to be weighted and can be pickled. An effort has been made to keep this small, due to the fact that its not unusual to have millions in memory.

**`__init__(self, int_dm, real_dm, cat, weight, index, rand, minimum_size = 1)`**
> Input consists of upto 5 arrays and one parameter. The first two parameters are data matrices, where each row contains the attributes for an example. Two are provided, one of numpy.int32 for the discrete features, another of numpy.float32 for the continuous features - one can be set to None to indicate none of that type, but obviously at least one has to be provided. The cat vector is then aligned with the data matrices and gives the category for each exemplar, as a numpy.int32. weight optionally provides a numpy.float32 vector that also aligns with the data matrices, and effectivly provides a continuous repeat count for each example, so some can be weighted as being more important. By default all items in the data matrices are used, however, instead an index vector can be provided that indexes the examples to be used by the tree - this not only allows a subset to be used but allows samples to be repeated if desired (This feature is actually used for building the tree recursivly, such that each DecTree object is in fact a node; also helps for creating a collection of trees with random trainning sets.). Finally, by default it considers all features at each level of the tree, however, if an integer rather than None is provided to the rand parameter it instead randomly selects a subset of attributes, of size rand, and then selects the best of this subset, with a new draw for each node in the tree. minimum\_size gives a minimum number of samples in a node for it to be split - it needs more samples than this otherwise it will become a leaf. A simple tree prunning method; defaults to 1 which is in effect it disabled.

**`classify(self, int_vec, real_vec)`**
> Given a pair of vectors, one for discrete attributes and another for continuous atributes this returns the trees estimated distribution for the exampler. This distribution will take the form of a dictionary, which you must not modify, that is indexed by categories and goes to a count of how many examples with that category were in that leaf node. 99% of the time only one category should exist, though various scenarios can result in there being more than 1.

**`entropy(self)`**
> Returns the entropy of the data that was used to train this node. Really an internal method, exposed in case of rampant curiosity. Note that it is in nats, not bits.

**`getChildren(self)`**
> Returns a dictionary of children nodes indexed by the attribute the decision is being made on if it makes a discrete decision, otherwise None. Note that any unseen attribute value will not be included.

**`getHigh(self)`**
> If it is a continuous decision node this returns the branch down which samples with the attribute higher than or equal to the threshold go to; otherwise None.

**`getIndex(self)`**
> Returns the index of either the discrete column or continuous column which it decides on, or None if it is a leaf.

**`getLow(self)`**
> If it is a continuous decision node this returns the branch down which samples with the attribute less than the threshold go to; otherwise None.

**`getThreshold(self)`**
> If it is a continuous node this returns the threshold between going down the low and high branches of the decision tree, otherwise returns None.

**`isContinuous(self)`**
> Returns True if it makes a decision by splitting a continuous node, False if its is either discrete or a leaf.

**`isDiscrete(self)`**
> Returns True if it makes its decision based on a discrete attribute, False if it is continuous or a leaf.

**`isLeaf(self)`**
> Returns True if it is a leaf node, False otherwise.

**`prob(self)`**
> Returns the distribution over the categories of the trainning examples that went through this node - if this is a leaf its likelly to be non-zero for just one category. Represented as a dictionary category -> weight that only includes entrys if they are not 0. weights are the sum of the weights for the input, and are not normalised.

**`size(self)`**
> Returns how many nodes make up the tree.