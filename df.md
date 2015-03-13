# Decision Forests #

## Overview ##
**Decision Forests**

An implementation of the method popularised by the paper
'Random Forests' by L. Breiman
which is a refinement, with a decent helping of rigour, of the papers
'Random Decision Forests' by T. K. Ho and
'Shape Quantization and Recognition with Randomized Trees' by Y. Amit and D. Geman.
This particular implementation is based on the tutorial from ICCV 2011, which has a corresponding technical report:
'Decision Forests for Classification, Regression, Density Estimation, Manifold Learning and Semi-Supervised Learning' by A. Criminisi, J. Shotton and E. Konukoglu.

Ultimately this is intended to be a fully featured decision forest, but for now it only supports classification and a limited set of test generators. This early version none-the-less contains support for both continuous and discrete features, and incremental learning (Admittedly a primitive form.). Implementation is pure python, using numpy, (Future versions will include inline C++ based optimisations.) and is extremely modular, to allow for future expansion.

Usage typically consist of initially creating a DF object and configuring it by setting its goal (Only classification is currently available.), its generator (A generator provides tests with which to split the data set at a node.), and configuring its Pruner, which defines when to stop growing each tree. Once setup the learn method is called with a set of exemplars which it uses to train the model. After the model has been learned the evaluate method can be applied to new features, to get a probability distribution over class assignment (And optionally a hard assignment.). The included tests demonstrate usage and should help understand usage; if you are interested in incremental learning then test\_inc.py is the file to read.

Particular attention must be paid to the generators - it is these that contain all of the really important parameters in the system, and choosing the right configuration will have a massive impact on performance. Consequentially there are a lot of generators, built on different principals and including composite generators to allow generators to be combined to build an infinite selection. This means that you should be able to find a good choice for your data, but also means that finding a good choice might take a certain amount of effort - be prepared to try many combinations.

If you are reading readme.txt then you can generate documentation by running make\_doc.py


Contains the following files:

`df.py` - Primary python file - import this to get all the classes, particularly the DF class, around which all functionality revolves.


`exemplars.py` - Provides the wrapper around data. All data fed into the system must match the interface provided within, probably by being a bunch of data matrices wrapped by the MatrixFS class. Uses a very specific vocabulary - each data point provided to the system is an exemplar, where all exemplars have a set of shared channels which each contain 1 or more features. Generators create tests for specific channels; also, channels provide auxiliary information, such as the actual answer and weights for the exemplars during learning. Given a specific channel index and a specific feature index within that channel plus a specific exemplar index you get back a single value from the structure. Whilst not provided the capability to inherit from the interface and implement one specific to the problem at hand, by calculating values on the fly for instance, exists.


`goals.py` - Defines a generic interface for defining the goal of a decision tree. Currently provides the Classification implementation of this interface, for solving the classification problem.


`generators.py` - Provides the generator interface and the composite generators. The composite generators allow you to select tests from multiple generators - this is most often used to combine a generator for continuous tests and a generator for discrete tests when your data contains both data types.

`gen_median.py` - Generators that attempt to split the data in half. Generally not a good approach, but for some problems this works very well.

`gen_random.py` - Generators based on pure randomness - generate enough random tests and something will work. This is very much in the vein of 'Extremely random forests'.

`gen_classify.py` - Optimising generators specific to the problem of classification. These are the traditional choices, and should be the first approaches tried.


`pruners.py` - Defines the concept of a Pruner, which decides when to stop growing a tree. Provides a default approach that stops growth when any of several limits is approached.


`test.py` - Defines the actual tests, and their interface. In practise a user only cares about Generators, as each Generator inherits from the Test relevant to what it returns.

`nodes.py` - Defines the node of a tree, where all the magic in fact happens. Its completely internal to the system however, and never used directly.


`test_discrete.py` - Tests the discrete generators on some simulated discrete data.

`test_continuous.py` - Tests the continuous generators on some simulated continuous data.

`test_mix.py` - Tests the composite generators on a simulated problem that contains both discrete and continuous data.

`test_inc.py` - Tests incremental learning, running batch on the same data so performance can be compared.


`readme.txt` - This file, which is included in the html documentation.

`make_doc.py` - Builds the html documentation.


---


# Classes #

## DF() ##
> Master object for the decision forest system - provides the entire interface. Typical use consists of setting up the system - its goal, pruner and generator(s), providing data to train a model and then using the model to analyse new exemplars. Incrimental learning is also supported however, albeit a not very sophisticated implimentation. Note that this class is compatable with pythons serialisation routines, for if you need to save/load a trained model.

**`__init__(self, other)`**
> Initialises as a blank model, ready to be setup and run. Can also act as a copy constructor if you provide an instance of DF as a single parameter.

**`addTree(self, es, weightChannel, ret = False, dummy = False)`**
> Adds an entirely new tree to the system given all of the new data. Uses all exemplars in the ExemplarSet, which can optionally include a channel with a single feature in it to weight the vectors; indicated via weightChannel. Typically this is used indirectly via the learn method, rather than by the user of an instance of this class.

**`allowC(self, allow)`**
> By default the system will attempt to compile and use C code instead of running the (much slower) python code - this allows you to force it to not use C code, or switch C back on if you had previously switched it off. Typically only used for speed comparisons and debugging, but also useful if the use of C code doesn't work on your system. Just be aware that the speed difference is galactic.

**`answer_types(self)`**
> Returns a dictionary giving all the answer types that can be requested using the which parameter of the evaluate method. The keys give the string to be provided to which, whilst the values give human readable descriptions of what will be returned. 'best' is always provided, as a point estimate of the best answer; most models also provide 'prob', which is a probability distribution over 'best', such that 'best' is the argmax of 'prob'.

**`error(self)`**
> Returns the average error of all the trees - meaning depends on the Goal at hand, but should provide an idea of how well the model is working.

**`evaluate(self, es, index = slice(None, None, None), which = best, mp = False, callback)`**
> Given some exemplars returns a list containing the output of the model for each exemplar. The returned list will align with the index, which defaults to everything and hence if not provided is aligned with es, the ExemplarSet. The meaning of the entrys in the list will depend on the Goal of the model and which: which can either be a single answer type from the goal object or a list of answer types, to get a tuple of answers for each list entry - the result is what the Goal-s answer method returns. The answer\_types method passes through to provide relevent information. Can be run in multiprocessing mode if you set the mp variable to True - only worth it if you have a lot of data (Also note that it splits by tree, so each process does all data items but for just one of the trees.). Should not be called if size()==0.

**`getGen(self)`**
> Returns the Generator object for the system.

**`getGoal(self)`**
> Returns the curent Goal object.

**`getGrow(self)`**
> Returns True if the trees will be subject to further growth during incrimental learning, when they have gained enough data to subdivide further.

**`getInc(self)`**
> Returns the status of incrimental learning - True if its enabled, False if it is not.

**`getPruner(self)`**
> Returns the current Pruner object.

**`learn(self, trees, es, weightChannel, clamp, mp = True, callback)`**
> This learns a model given data, and, when it is switched on, will also do incrimental learning. trees is how many new trees to create - for normal learning this is just how many to make, for incrimental learning it is how many to add to those that have already been made - more is always better, within reason, but it is these that cost you computation and memory. es is the ExemplarSet containing the data to train on. For incrimental learning you always provide the previous data, at the same indices, with the new exemplars appended to the end. weightChannel allows you to give a channel containing a single feature if you want to weight the importance of the exemplars. clamp is only relevent to incrimental learning - it is effectivly a maximum number of trees to allow, where it throws away the weakest trees first. This is how incrimental learning works, and so must be set for that - by constantly adding new trees as new data arrives and updating the error metrics of the older trees (The error will typically increase with new data.) the less-well trainned (and typically older) trees will be culled. mp indicates if multiprocessing should be used or not - True to do so, False to not. Will automatically switch itself off if not supported.

**`lumberjack(self, count)`**
> Once a bunch of trees have been learnt this culls them, reducing them such that there are no more than count. It terminates those with the highest error rate first, and does nothing if there are not enough trees to excede count. Typically this is used by the learn method, rather than by the object user.

**`nodes(self)`**
> Returns the total number of nodes in all the trees.

**`setGen(self, gen)`**
> Allows you to set the Generator object from which node tests are obtained - must be set before anything happens. You must not change this once trainning starts.

**`setGoal(self, goal)`**
> Allows you to set a goal object, of type Goal - must be called before doing anything, and must not be changed after anything is done.

**`setInc(self, inc, grow = False)`**
> Set this to True to support incrimental learning, False to not. Having incrimental learning on costs extra memory, but has little if any computational affect. If incrimental learning is on you can also switch grow on, in which case as more data arrives it tries to split the leaf nodes of trees that have already been grown. Requires a bit more memory be used, as it needs to keep the indices of the training set for future growth. Note that the default pruner is entirly inappropriate for this mode - the pruner has to be set such that as more data arrives it will allow future growth.

**`setPruner(self, pruner)`**
> Sets the pruner, which controls when to stop growing each tree. By default this is set to the PruneCap object with default parameters, though you might want to use getPruner to get it so you can adjust its parameters to match the problem at hand, as the pruner is important for avoiding overfitting.

**`size(self)`**
> Returns the number of trees within the forest.

## ExemplarSet() ##
> An interface for a set of feature vectors, referred to as exemplars - whilst a data matrix will typically be used this allows the possibility of exemplars for which that is impractical, i.e. calculating them on the fly if there is an unreasonable number of features within each exemplar. Also supports the concept of channels, which ties in with the test generation so you can have different generators for each channel. For trainning the 'answer' is stored in its own channel (Note that, because that channel will not exist for novel features it should always be the last channel, so that indexing is consistant, unless it is replaced with a dummy channel.), the type of which will depend on the problem being solved. Also allows the mixing of both continuous and discrete values.

**`__getitem__(self, index)`**
> Actual data access is via the [.md](.md) operator, with 3 entities - [channel, exemplar(s), feature(s)]. channel must index the channel, and indicates which channel to get the data from - must always be an integer. exemplars(s) indicates which examples to return and features(s) which features to return for each of the exemplars. For both of these 3 indexing types must be supported - a single integer, a slice, or a numpy array of indexing integers. For indexing integers the system is designed to work such that repetitions are never used, though that is in fact supported most of the time by actual implimentations. The return value must always have the type indicated by the dtype method for the channel in question. If both are indexed with an integer then it will return a single number (But still of the numpy dtype.); if only 1 is an integer a 1D numpy array; and if neither are integers a 2D numpy array, indexed [exemplar, relative feature](relative.md). Note that this last requirement is not the numpy default, which would actually continue to give a 1D array rather than the 2D subset defined by two sets of indicies.

**`channels(self)`**
> Returns how many channels of features are provided. They are indexed {0, ..., return-1}.

**`codeC(self, channel, name)`**
> Returns a dictionary containning all the entities needed to access the given channel of the exemplar from within C, using name to provide a unique string to avoid namespace clashes. Will raise a NotImplementedError if not avaliable. `['type'] = The C type of the channel, often 'float'. ['input'] = The input object to be passed into the C code, must be protected from any messing around that scipy.weave might do. ['itype'] = The input type in C, as a string, usually 'PyObject *' or 'PyArrayObject *'. ['get'] = Returns code for a function to get values from the channel of the exemplar; has calling convention <type> <name>_get(<itype> input, int exemplar, int feature). ['exemplars'] = Code for a function to get the number of exemplars; has calling convention int <name>_exemplars(<itype> input). Will obviously return the same value for all channels, so can be a bit redundant. ['features'] = Code for a function that returns how many features the channel has; calling convention is int <name>_features(<itype> input). ['name'] is also provided, which contains the base name handed to this method, for conveniance.`

**`dtype(self, channel)`**
> Returns the numpy dtype used for the given channel. numpy.float32 and numpy.int32 are considered to be the standard choices.

**`exemplars(self)`**
> Returns how many exemplars are provided.

**`features(self, channel)`**
> Returns how many features exist for each exemplar for the given channel - they can then be indexed {0, .., return-1} for the given channel.

**`key(self)`**
> Provides a unique string that can be used to hash the results of codeC, to avoid repeated generation. Must be implimented if codeC is implimented.

**`listCodeC(self, name)`**
> Helper method - returns a tuple indexed by channel that gives the dictionary returned by codeC for each channel in this exemplar. It generates the names using the provided name by adding the number indexing the channel to the end. Happens to by the required input elsewhere.

**`name(self, channel)`**
> Returns a string as a name for a channel - this is optional and provided more for human usage. Will return None if the channel in question has no name.

**`tupleInputC(self)`**
> Helper method, that can be overriden - returns a tuple containing the inputs needed for the exemplar.

## MatrixES(ExemplarSet) ##
> The most common exemplar set - basically what you use when all the feature vectors can be computed and then stored in memory without issue. Contains a data matrix for each channel, where these are provided by the user.

**`__init__(self, *args)`**
> Optionally allows you to provide a list of numpy data matrices to by the channels data matrices. Alternativly you can use the add method to add them, one after another, post construction, or some combination of both. All data matrices must be 2D numpy arrays, with the first dimension, indexing the exemplar, being the same size in all cases. (If there is only 1 exemplar then it will accept 1D arrays.)

**`__getitem__(self, index)`**
> Actual data access is via the [.md](.md) operator, with 3 entities - [channel, exemplar(s), feature(s)]. channel must index the channel, and indicates which channel to get the data from - must always be an integer. exemplars(s) indicates which examples to return and features(s) which features to return for each of the exemplars. For both of these 3 indexing types must be supported - a single integer, a slice, or a numpy array of indexing integers. For indexing integers the system is designed to work such that repetitions are never used, though that is in fact supported most of the time by actual implimentations. The return value must always have the type indicated by the dtype method for the channel in question. If both are indexed with an integer then it will return a single number (But still of the numpy dtype.); if only 1 is an integer a 1D numpy array; and if neither are integers a 2D numpy array, indexed [exemplar, relative feature](relative.md). Note that this last requirement is not the numpy default, which would actually continue to give a 1D array rather than the 2D subset defined by two sets of indicies.

**`add(self, dm)`**
> Adds a new data matrix of information as another channel. Returns its channel index. If given a 1D matrix assumes that there is only one exemplar and adjusts it accordingly.

**`append(self, *args)`**
> Allows you to add exemplars to the structure, by providing a set of data matrices that align with those contained, which contain the new exemplars. Note that this is slow and generally ill advised. If adding a single new feature the arrays can be 1D.

**`channels(self)`**
> Returns how many channels of features are provided. They are indexed {0, ..., return-1}.

**`codeC(self, channel, name)`**
> Returns a dictionary containning all the entities needed to access the given channel of the exemplar from within C, using name to provide a unique string to avoid namespace clashes. Will raise a NotImplementedError if not avaliable. `['type'] = The C type of the channel, often 'float'. ['input'] = The input object to be passed into the C code, must be protected from any messing around that scipy.weave might do. ['itype'] = The input type in C, as a string, usually 'PyObject *' or 'PyArrayObject *'. ['get'] = Returns code for a function to get values from the channel of the exemplar; has calling convention <type> <name>_get(<itype> input, int exemplar, int feature). ['exemplars'] = Code for a function to get the number of exemplars; has calling convention int <name>_exemplars(<itype> input). Will obviously return the same value for all channels, so can be a bit redundant. ['features'] = Code for a function that returns how many features the channel has; calling convention is int <name>_features(<itype> input). ['name'] is also provided, which contains the base name handed to this method, for conveniance.`

**`dtype(self, channel)`**
> Returns the numpy dtype used for the given channel. numpy.float32 and numpy.int32 are considered to be the standard choices.

**`exemplars(self)`**
> Returns how many exemplars are provided.

**`features(self, channel)`**
> Returns how many features exist for each exemplar for the given channel - they can then be indexed {0, .., return-1} for the given channel.

**`key(self)`**
> Provides a unique string that can be used to hash the results of codeC, to avoid repeated generation. Must be implimented if codeC is implimented.

**`listCodeC(self, name)`**
> Helper method - returns a tuple indexed by channel that gives the dictionary returned by codeC for each channel in this exemplar. It generates the names using the provided name by adding the number indexing the channel to the end. Happens to by the required input elsewhere.

**`name(self, channel)`**
> Returns a string as a name for a channel - this is optional and provided more for human usage. Will return None if the channel in question has no name.

**`tupleInputC(self)`**
> Helper method, that can be overriden - returns a tuple containing the inputs needed for the exemplar.

## MatrixGrow(ExemplarSet) ##
> A slightly more advanced version of the basic exemplar set that has better support for incrimental learning, as it allows appends to be more efficient. It still assumes that all of the data can be fitted in memory, and makes use of numpy arrays for internal storage.

**`__init__(self, *args)`**
> Optionally allows you to provide a list of numpy data matrices to by the channels data matrices. Alternativly you can use the add method to add them, one after another, post construction, or use append to start things going. All data matrices must be 2D numpy arrays, with the first dimension, indexing the exemplar, being the same size in all cases. (If there is only 1 exemplar then it will accept 1D arrays.)

**`__getitem__(self, index)`**
> Actual data access is via the [.md](.md) operator, with 3 entities - [channel, exemplar(s), feature(s)]. channel must index the channel, and indicates which channel to get the data from - must always be an integer. exemplars(s) indicates which examples to return and features(s) which features to return for each of the exemplars. For both of these 3 indexing types must be supported - a single integer, a slice, or a numpy array of indexing integers. For indexing integers the system is designed to work such that repetitions are never used, though that is in fact supported most of the time by actual implimentations. The return value must always have the type indicated by the dtype method for the channel in question. If both are indexed with an integer then it will return a single number (But still of the numpy dtype.); if only 1 is an integer a 1D numpy array; and if neither are integers a 2D numpy array, indexed [exemplar, relative feature](relative.md). Note that this last requirement is not the numpy default, which would actually continue to give a 1D array rather than the 2D subset defined by two sets of indicies.

**`add(self, dm)`**
> Adds a new data matrix of information as another channel. Returns its channel index. If given a 1D matrix assumes that there is only one exemplar and adjusts it accordingly.

**`append(self, *args)`**
> Allows you to add exemplars to the structure, by providing a set of data matrices that align with those contained, which contain the new exemplars. If adding a single new exemplar the arrays can be 1D.

**`channels(self)`**
> Returns how many channels of features are provided. They are indexed {0, ..., return-1}.

**`codeC(self, channel, name)`**
> Returns a dictionary containning all the entities needed to access the given channel of the exemplar from within C, using name to provide a unique string to avoid namespace clashes. Will raise a NotImplementedError if not avaliable. `['type'] = The C type of the channel, often 'float'. ['input'] = The input object to be passed into the C code, must be protected from any messing around that scipy.weave might do. ['itype'] = The input type in C, as a string, usually 'PyObject *' or 'PyArrayObject *'. ['get'] = Returns code for a function to get values from the channel of the exemplar; has calling convention <type> <name>_get(<itype> input, int exemplar, int feature). ['exemplars'] = Code for a function to get the number of exemplars; has calling convention int <name>_exemplars(<itype> input). Will obviously return the same value for all channels, so can be a bit redundant. ['features'] = Code for a function that returns how many features the channel has; calling convention is int <name>_features(<itype> input). ['name'] is also provided, which contains the base name handed to this method, for conveniance.`

**`dtype(self, channel)`**
> Returns the numpy dtype used for the given channel. numpy.float32 and numpy.int32 are considered to be the standard choices.

**`exemplars(self)`**
> Returns how many exemplars are provided.

**`features(self, channel)`**
> Returns how many features exist for each exemplar for the given channel - they can then be indexed {0, .., return-1} for the given channel.

**`key(self)`**
> Provides a unique string that can be used to hash the results of codeC, to avoid repeated generation. Must be implimented if codeC is implimented.

**`listCodeC(self, name)`**
> Helper method - returns a tuple indexed by channel that gives the dictionary returned by codeC for each channel in this exemplar. It generates the names using the provided name by adding the number indexing the channel to the end. Happens to by the required input elsewhere.

**`make_compact(self)`**
> Internal method really - converts the data structure so that len(dmcList)==1, by concatenating arrays as needed.

**`name(self, channel)`**
> Returns a string as a name for a channel - this is optional and provided more for human usage. Will return None if the channel in question has no name.

**`tupleInputC(self)`**
> Helper method, that can be overriden - returns a tuple containing the inputs needed for the exemplar.

## Goal() ##
> Interface that defines the purpose of a decision forest - defines what the tree is optimising, what statistics to store at each node and what is returned to the user as the answer when they provide a novel feature to the forest (i.e. how to combine the statistics).

**`answer(self, stats_list, which, es, index, trees)`**
> Given a feature then using a forest a list of statistics entitys can be obtained from the leaf nodes that the feature ends up in, one for each tree (Could be as low as just one entity.). This converts that statistics entity list into an answer, to be passed to the user, possibly using the es with the index of the one entry that the stats list is for as well. As multiple answer types exist (As provided by the answer\_types method.) you provide the one(s) you want to the which variable - if which is a string then that answer type is returned, if it is a list of strings then a tuple aligned with it is returned, containing multiple answers. If multiple types are needed then returning a list should hopefuly be optimised by this method to avoid duplicate calculation. Also requires the trees themselves, as a list aligned with stats\_list.

**`answer_batch(self, stats_lists, which, es, indices, trees)`**
> A batch version of answer, that does multiple stat lists at once. The stats\_list now consists of a list of lists, where the outer list matches tne entrys in index (A numpy array), and the inner list are the samples, aligned with the trees list. es is the exemplar object that matches up with index, and which gives the output(s) to provide. Return value is a list, matching index, that contains the answer for each, which can be a tuple if which is alist/tuple. A default implimentation is provided.

**`answer_types(self)`**
> When classifying a new feature an answer is to be provided, of which several possibilities exist. This returns a dictionary of those possibilities (key==name, value=human readable description of what it is.), from which the user can select. By convention 'best' must always exist, as the best guess that the algorithm can give (A point estimate of the answer the user is after.). If a probability distribution over 'best' can be provided then that should be avaliable as 'prob' (It is highly recomended that this be provided.).

**`clone(self)`**
> Returns a deep copy of this object.

**`codeC(self, name, escl)`**
> Returns a dictionary of strings containing C code, that impliment the Goal's methods in C - name is a prefix on the names used, escl the result of listCodeC on the exemplar set from which it will get its data. The contents of its return value must contain some of: `{'stats': 'void <name>_stats(PyObject * data, Exemplar * index, void *& out, size_t & outLen)' - data is the list of channels for the exemplar object, index the exemplars to use. The stats object is stuck into out, and the size updated accordingly. If the provided out object is too small it will be free-ed and then a large enough buffer malloc-ed; null is handled correctly if outLen is 0., 'updateStats': 'void <name>_updateStats(PyObject * data, Exemplar * index, void *& inout, size_t & inoutLen)' - Same as stats, except the inout data arrives already containing a stats object, which is to be updated with the provided exemplars., 'entropy':'float <name>_entropy(void * stats, size_t statsLen) - Given a stats object returns its entropy.', 'summary': 'void <name>_summary(PyObject * data, Exemplar * index, void *& out, size_t & outLen)' - Basically the same as stats, except this time it is using the exemplars to calculate a summary. Interface works in the same way., 'updateSummary': 'void <name>_updateSummary(PyObject * data, Exemplar * index, void *& inout, size_t & inoutLen)' - Given a summary object, using the inout variables it updates it with the provided exemplars., 'error': 'void <name>_error(void * stats, size_t statsLen, void * summary, size_t summaryLen, float & error, float & weight)' - Given two buffers, representing the stats and the summary, this calculates the error, which is put into the reference error. This should be done incrimentally, such that errors from all nodes in a tree can be merged - error will be initialised at 0, and addtionally weight is provided which can be used as it wishes (Incremental mean is typical.), also initialised as 0.}`. Optional - if it throws the NotImplementedError (The default) everything will be done in python, if some C code is dependent on a missing C method it will also be done in python. The code can be dependent on the associated exempler code where applicable.

**`entropy(self, stats)`**
> Given a statistics entity this returns the associated entropy - this is used to choose which test is best.

**`error(self, stats, summary)`**
> Given a stats entity and a summary entity (i.e. the details of the testing and trainning sets that have reached a leaf) this returns the error of the testing set versus the model learnt from the trainning set. The actual return is a pair - (error, weight), so that the errors from all the leafs can be combined in a weighted average. The error metric is arbitary, but the probability of 'being wrong' is a good choice. An alternate mode exists, where weight is set to None - in this case no averaging occurs and the results from all nodes are just summed together.

**`key(self)`**
> Provides a unique string that can be used to hash the results of codeC, to avoid repeated generation. Must be implimented if codeC is implimented.

**`postTreeGrow(self, root, gen)`**
> After a tree is initially grown (At which point its shape is locked, but incrimental learning could still be applied.) this method is given the root node of the tree, and can do anything it likes to it - a post processing step, in case the stats objects need some extra cleverness. Most Goal-s do not need to impliment this. Also provided the generator for the tests in the tree.

**`stats(self, es, index, weights)`**
> Generates a statistics entity for a node, based on the features that make it to the node. The statistics entity is decided by the task at hand, but must allow the nodes entropy to be calculated, plus a collection of these is used to generate the answer when a feature is given to the decision forest. fs is a feature set, index the indices of the features in fs that have made it to this node. weights is an optional set of weights for the features, weighting how many features they are worth - will be a 1D numpy.float32 array aligned with the feature set, and can contain fractional weights.

**`summary(self, es, index, weights)`**
> Once a tree has been grown a testing set (The 'out-of-bag' set) is typically run through to find out how good it is. This consists of two steps, the first of which is to generate a summary of the oob set that made it to each leaf. This generates the summary, and must be done such that the next step - the use of a stats and summary entity to infer an error metric with a weight for averaging the error metrics from all leafs, can be performed. For incrimental learning it is also required to be able to add new exemplars at a later time.

**`updateStats(self, stats, es, index, weights)`**
> Given a stats entity, as generated by the stats method, this returns a copy of that stats entity that has had additional exemplars factored in, specifically those passed in. This allows a tree to be updated with further trainning examples (Or, at least its stats to be updated - its structure is set in stone once built.) Needed for incrimental learning.

**`updateSummary(self, summary, es, index, weights)`**
> For incrimental learning the summaries need to be updated with further testing examples - this does that. Given a summary and some exemplars it returns a copy of the summary updated with the new exemplars.

## Classification(Goal) ##
> The standard goal of a decision forest - classification. When trainning expects the existence of a discrete channel containing a single feature for each exemplar, the index of which is provided. Each discrete feature indicates a different trainning class, and they should be densly packed, starting from 0 inclusive, i.e. belonging to the set {0, ..., # of classes-1}. Number of classes is typically provided, though None can be provided instead in which case it will automatically resize data structures as needed to make them larger as more classes (Still densly packed.) are seen. A side effect of this mode is when it returns arrays indexed by class the size will be data driven, and from the view of the user effectivly arbitrary - user code will have to handle this.

**`__init__(self, classCount, channel)`**
> You provide firstly how many classes exist (Or None if unknown.), and secondly the index of the channel that contains the ground truth for the exemplars. This channel must contain a single integer value, ranging from 0 inclusive to the number of classes, exclusive.

**`answer(self, stats_list, which, es, index, trees)`**
> Given a feature then using a forest a list of statistics entitys can be obtained from the leaf nodes that the feature ends up in, one for each tree (Could be as low as just one entity.). This converts that statistics entity list into an answer, to be passed to the user, possibly using the es with the index of the one entry that the stats list is for as well. As multiple answer types exist (As provided by the answer\_types method.) you provide the one(s) you want to the which variable - if which is a string then that answer type is returned, if it is a list of strings then a tuple aligned with it is returned, containing multiple answers. If multiple types are needed then returning a list should hopefuly be optimised by this method to avoid duplicate calculation. Also requires the trees themselves, as a list aligned with stats\_list.

**`answer_batch(self, stats_lists, which, es, indices, trees)`**
> A batch version of answer, that does multiple stat lists at once. The stats\_list now consists of a list of lists, where the outer list matches tne entrys in index (A numpy array), and the inner list are the samples, aligned with the trees list. es is the exemplar object that matches up with index, and which gives the output(s) to provide. Return value is a list, matching index, that contains the answer for each, which can be a tuple if which is alist/tuple. A default implimentation is provided.

**`answer_types(self)`**
> When classifying a new feature an answer is to be provided, of which several possibilities exist. This returns a dictionary of those possibilities (key==name, value=human readable description of what it is.), from which the user can select. By convention 'best' must always exist, as the best guess that the algorithm can give (A point estimate of the answer the user is after.). If a probability distribution over 'best' can be provided then that should be avaliable as 'prob' (It is highly recomended that this be provided.).

**`clone(self)`**
> Returns a deep copy of this object.

**`codeC(self, name, escl)`**
> Returns a dictionary of strings containing C code, that impliment the Goal's methods in C - name is a prefix on the names used, escl the result of listCodeC on the exemplar set from which it will get its data. The contents of its return value must contain some of: `{'stats': 'void <name>_stats(PyObject * data, Exemplar * index, void *& out, size_t & outLen)' - data is the list of channels for the exemplar object, index the exemplars to use. The stats object is stuck into out, and the size updated accordingly. If the provided out object is too small it will be free-ed and then a large enough buffer malloc-ed; null is handled correctly if outLen is 0., 'updateStats': 'void <name>_updateStats(PyObject * data, Exemplar * index, void *& inout, size_t & inoutLen)' - Same as stats, except the inout data arrives already containing a stats object, which is to be updated with the provided exemplars., 'entropy':'float <name>_entropy(void * stats, size_t statsLen) - Given a stats object returns its entropy.', 'summary': 'void <name>_summary(PyObject * data, Exemplar * index, void *& out, size_t & outLen)' - Basically the same as stats, except this time it is using the exemplars to calculate a summary. Interface works in the same way., 'updateSummary': 'void <name>_updateSummary(PyObject * data, Exemplar * index, void *& inout, size_t & inoutLen)' - Given a summary object, using the inout variables it updates it with the provided exemplars., 'error': 'void <name>_error(void * stats, size_t statsLen, void * summary, size_t summaryLen, float & error, float & weight)' - Given two buffers, representing the stats and the summary, this calculates the error, which is put into the reference error. This should be done incrimentally, such that errors from all nodes in a tree can be merged - error will be initialised at 0, and addtionally weight is provided which can be used as it wishes (Incremental mean is typical.), also initialised as 0.}`. Optional - if it throws the NotImplementedError (The default) everything will be done in python, if some C code is dependent on a missing C method it will also be done in python. The code can be dependent on the associated exempler code where applicable.

**`entropy(self, stats)`**
> Given a statistics entity this returns the associated entropy - this is used to choose which test is best.

**`error(self, stats, summary)`**
> Given a stats entity and a summary entity (i.e. the details of the testing and trainning sets that have reached a leaf) this returns the error of the testing set versus the model learnt from the trainning set. The actual return is a pair - (error, weight), so that the errors from all the leafs can be combined in a weighted average. The error metric is arbitary, but the probability of 'being wrong' is a good choice. An alternate mode exists, where weight is set to None - in this case no averaging occurs and the results from all nodes are just summed together.

**`key(self)`**
> Provides a unique string that can be used to hash the results of codeC, to avoid repeated generation. Must be implimented if codeC is implimented.

**`postTreeGrow(self, root, gen)`**
> After a tree is initially grown (At which point its shape is locked, but incrimental learning could still be applied.) this method is given the root node of the tree, and can do anything it likes to it - a post processing step, in case the stats objects need some extra cleverness. Most Goal-s do not need to impliment this. Also provided the generator for the tests in the tree.

**`stats(self, es, index, weights)`**
> Generates a statistics entity for a node, based on the features that make it to the node. The statistics entity is decided by the task at hand, but must allow the nodes entropy to be calculated, plus a collection of these is used to generate the answer when a feature is given to the decision forest. fs is a feature set, index the indices of the features in fs that have made it to this node. weights is an optional set of weights for the features, weighting how many features they are worth - will be a 1D numpy.float32 array aligned with the feature set, and can contain fractional weights.

**`summary(self, es, index, weights)`**
> Once a tree has been grown a testing set (The 'out-of-bag' set) is typically run through to find out how good it is. This consists of two steps, the first of which is to generate a summary of the oob set that made it to each leaf. This generates the summary, and must be done such that the next step - the use of a stats and summary entity to infer an error metric with a weight for averaging the error metrics from all leafs, can be performed. For incrimental learning it is also required to be able to add new exemplars at a later time.

**`updateStats(self, stats, es, index, weights)`**
> Given a stats entity, as generated by the stats method, this returns a copy of that stats entity that has had additional exemplars factored in, specifically those passed in. This allows a tree to be updated with further trainning examples (Or, at least its stats to be updated - its structure is set in stone once built.) Needed for incrimental learning.

**`updateSummary(self, summary, es, index, weights)`**
> For incrimental learning the summaries need to be updated with further testing examples - this does that. Given a summary and some exemplars it returns a copy of the summary updated with the new exemplars.

## DensityGaussian(Goal) ##
> Provides the ability to construct a density estimate, using Gaussian distributions to represent the density at each node in the tree. A rather strange thing to be doing with a decision forest, and I am a little suspicious of it, but it does give usable results, at least for low enough dimensionalities where everything remains sane. Due to its nature it can be very memory consuming if your doing incrmental learning - the summary has to store all the provided samples. Requires a channel to contain all the features that are fed into the density estimate (It is to this that a Gaussian is fitted.), which is always in channel 0. Other features can not exist, so typically input data would only have 1 channel. Because the divisions between nodes are sharp (This is a mixture model only between trees, not between leaf nodes within each tree.) the normalisation constant for each Gaussian has to be adjusted to take this into account. This is acheived by sampling - sending samples from the Gaussian down the tree and counting what percentage make the node. Note that when calculating the Gaussian at each node a prior is used, to avoid degeneracies, with a default weight of 1, so if weights are provided they should be scaled accordingly. Using a decision tree for density estimation is a bit hit and miss based on my experiance - you need to pay very close attention to tuning the min train parameter of the pruner, as information gain is a terrible stopping metric in this case. You also need a lot of trees to get something smooth out, which means it is quite computationally expensive.

**`__init__(self, feats, samples = 1024, prior_weight = 1.0)`**
> feats is the number of features to be found in channel 0 of the data, which are uses to fit a Gaussian at each node. samples is how many samples per node it sends down the tree, to weight that node according to the samples that can actually reach it. prior\_weight is the weight assigned to a prior used on each node to avoid degeneracies - it defaults to 1, with 0 removing it entirly (Not recomended.).

**`answer(self, stats_list, which, es, index, trees)`**
> Given a feature then using a forest a list of statistics entitys can be obtained from the leaf nodes that the feature ends up in, one for each tree (Could be as low as just one entity.). This converts that statistics entity list into an answer, to be passed to the user, possibly using the es with the index of the one entry that the stats list is for as well. As multiple answer types exist (As provided by the answer\_types method.) you provide the one(s) you want to the which variable - if which is a string then that answer type is returned, if it is a list of strings then a tuple aligned with it is returned, containing multiple answers. If multiple types are needed then returning a list should hopefuly be optimised by this method to avoid duplicate calculation. Also requires the trees themselves, as a list aligned with stats\_list.

**`answer_batch(self, stats_lists, which, es, indices, trees)`**
> A batch version of answer, that does multiple stat lists at once. The stats\_list now consists of a list of lists, where the outer list matches tne entrys in index (A numpy array), and the inner list are the samples, aligned with the trees list. es is the exemplar object that matches up with index, and which gives the output(s) to provide. Return value is a list, matching index, that contains the answer for each, which can be a tuple if which is alist/tuple. A default implimentation is provided.

**`answer_types(self)`**
> When classifying a new feature an answer is to be provided, of which several possibilities exist. This returns a dictionary of those possibilities (key==name, value=human readable description of what it is.), from which the user can select. By convention 'best' must always exist, as the best guess that the algorithm can give (A point estimate of the answer the user is after.). If a probability distribution over 'best' can be provided then that should be avaliable as 'prob' (It is highly recomended that this be provided.).

**`clone(self)`**
> Returns a deep copy of this object.

**`codeC(self, name, escl)`**
> Returns a dictionary of strings containing C code, that impliment the Goal's methods in C - name is a prefix on the names used, escl the result of listCodeC on the exemplar set from which it will get its data. The contents of its return value must contain some of: `{'stats': 'void <name>_stats(PyObject * data, Exemplar * index, void *& out, size_t & outLen)' - data is the list of channels for the exemplar object, index the exemplars to use. The stats object is stuck into out, and the size updated accordingly. If the provided out object is too small it will be free-ed and then a large enough buffer malloc-ed; null is handled correctly if outLen is 0., 'updateStats': 'void <name>_updateStats(PyObject * data, Exemplar * index, void *& inout, size_t & inoutLen)' - Same as stats, except the inout data arrives already containing a stats object, which is to be updated with the provided exemplars., 'entropy':'float <name>_entropy(void * stats, size_t statsLen) - Given a stats object returns its entropy.', 'summary': 'void <name>_summary(PyObject * data, Exemplar * index, void *& out, size_t & outLen)' - Basically the same as stats, except this time it is using the exemplars to calculate a summary. Interface works in the same way., 'updateSummary': 'void <name>_updateSummary(PyObject * data, Exemplar * index, void *& inout, size_t & inoutLen)' - Given a summary object, using the inout variables it updates it with the provided exemplars., 'error': 'void <name>_error(void * stats, size_t statsLen, void * summary, size_t summaryLen, float & error, float & weight)' - Given two buffers, representing the stats and the summary, this calculates the error, which is put into the reference error. This should be done incrimentally, such that errors from all nodes in a tree can be merged - error will be initialised at 0, and addtionally weight is provided which can be used as it wishes (Incremental mean is typical.), also initialised as 0.}`. Optional - if it throws the NotImplementedError (The default) everything will be done in python, if some C code is dependent on a missing C method it will also be done in python. The code can be dependent on the associated exempler code where applicable.

**`entropy(self, stats)`**
> Given a statistics entity this returns the associated entropy - this is used to choose which test is best.

**`error(self, stats, summary)`**
> Given a stats entity and a summary entity (i.e. the details of the testing and trainning sets that have reached a leaf) this returns the error of the testing set versus the model learnt from the trainning set. The actual return is a pair - (error, weight), so that the errors from all the leafs can be combined in a weighted average. The error metric is arbitary, but the probability of 'being wrong' is a good choice. An alternate mode exists, where weight is set to None - in this case no averaging occurs and the results from all nodes are just summed together.

**`key(self)`**
> Provides a unique string that can be used to hash the results of codeC, to avoid repeated generation. Must be implimented if codeC is implimented.

**`postTreeGrow(self, root, gen)`**
> After a tree is initially grown (At which point its shape is locked, but incrimental learning could still be applied.) this method is given the root node of the tree, and can do anything it likes to it - a post processing step, in case the stats objects need some extra cleverness. Most Goal-s do not need to impliment this. Also provided the generator for the tests in the tree.

**`stats(self, es, index, weights)`**
> Generates a statistics entity for a node, based on the features that make it to the node. The statistics entity is decided by the task at hand, but must allow the nodes entropy to be calculated, plus a collection of these is used to generate the answer when a feature is given to the decision forest. fs is a feature set, index the indices of the features in fs that have made it to this node. weights is an optional set of weights for the features, weighting how many features they are worth - will be a 1D numpy.float32 array aligned with the feature set, and can contain fractional weights.

**`summary(self, es, index, weights)`**
> Once a tree has been grown a testing set (The 'out-of-bag' set) is typically run through to find out how good it is. This consists of two steps, the first of which is to generate a summary of the oob set that made it to each leaf. This generates the summary, and must be done such that the next step - the use of a stats and summary entity to infer an error metric with a weight for averaging the error metrics from all leafs, can be performed. For incrimental learning it is also required to be able to add new exemplars at a later time.

**`updateStats(self, stats, es, index, weights)`**
> Given a stats entity, as generated by the stats method, this returns a copy of that stats entity that has had additional exemplars factored in, specifically those passed in. This allows a tree to be updated with further trainning examples (Or, at least its stats to be updated - its structure is set in stone once built.) Needed for incrimental learning.

**`updateSummary(self, summary, es, index, weights)`**
> For incrimental learning the summaries need to be updated with further testing examples - this does that. Given a summary and some exemplars it returns a copy of the summary updated with the new exemplars.

## Pruner() ##
> This abstracts the decision of when to stop growing a tree. It takes various statistics and stops growing when some condition is met.

**`clone(self)`**
> Returns a copy of this object.

**`keep(self, depth, trueCount, falseCount, infoGain, node)`**
> Each time a node is split this method is called to decide if the split should be kept or not - it returns True to keep (And hence the children will be recursivly split, and have keep called on them, etc..) and False to discard the nodes children and stop. depth is how deep the node in question is, where the root node is 0, its children 1, and do on. trueCount and falseCount indicate how many data points in the training set go each way, whilst infoGain is the information gained by the split. Finally, node is the actual node incase some more complicated analysis is desired - at the time of passing in its test and stats will exist, but everything else will not.

## PruneCap(Pruner) ##
> A simple but effective Pruner implimentation - simply provides a set of thresholds on depth, number of training samples required to split and information gained - when any one of the thresholds is tripped it stops further branching.

**`__init__(self, maxDepth = 8, minTrain = 8, minGain = 0.001, minDepth = 2)`**
> maxDepth is the maximum depth of a node in the tree, after which it stops - remember that the maximum node count based on this threshold increases dramatically as this number goes up, so don't go too crazy. minTrain is the smallest size node it will consider for further splitting. minGain is a lower limit on how much information gain a split must provide to be accepted. minDepth overrides the minimum node size - as long as the node count does not reach zero in either branch it will always split to the given depth - used to force it to at least learn something.

**`clone(self)`**
> Returns a copy of this object.

**`keep(self, depth, trueCount, falseCount, infoGain, node)`**
> Each time a node is split this method is called to decide if the split should be kept or not - it returns True to keep (And hence the children will be recursivly split, and have keep called on them, etc..) and False to discard the nodes children and stop. depth is how deep the node in question is, where the root node is 0, its children 1, and do on. trueCount and falseCount indicate how many data points in the training set go each way, whilst infoGain is the information gained by the split. Finally, node is the actual node incase some more complicated analysis is desired - at the time of passing in its test and stats will exist, but everything else will not.

**`setMaxDepth(self, maxDepth)`**
> Sets the depth cap on the trees.

**`setMinDepth(self, minDepth)`**
> Sets the minimum tree growing depth - trees will be grown at least this deep, baring insurmountable issues.

**`setMinGain(self, mingain)`**
> Sets the minimum gain that is allowed for a split to be accepted.

**`setMinTrain(self, minTrain)`**
> Sets the minimum number of nodes allowed to be split.

## Test() ##
> Interface for a test definition. This provides the concept of a test that an exemplar either passes or fails. The test is actually defined by some arbitary entity made by a matching generator, but this object is required to actually do the test - contains the relevant code and any shared parameters to keep memory consumption low, as there could be an aweful lot of tests. The seperation of test from generator is required as there are typically many methods to generate a specific test - generators inherit from the relevant test object.

**`do(self, test, es, index = slice(None, -1, None))`**
> Does the test. Given the entity that defines the actual test and an ExemplarSet to run the test on. An optional index for the features can be provided (Passed directly through to the exemplar sets [.md](.md) operator, so its indexing rules apply.), but if omitted it runs for all. Return value is a boolean numpy array indexed by the relative exemplar indices, that gives True if it passsed the test, False if it failed.

**`testCodeC(self, name, exemplar_list)`**
> Provides C code to perform the test - provides a C function that is given the test object as a pointer to the first byte and the index of the exemplar to test; it then returns true or false dependeing on if it passes the test or not. Returned string contains a function with the calling convention `bool <name>(PyObject * data, void * test, size_t test_length, int exemplar)`. data is a python tuple indexed by channel containning the object to be fed to the exemplar access function. To construct this function it needs the return value of listCodeC for an ExemplarSet, so it can get the calling convention to access the channel. When compiled the various functions must be avaliable.

## AxisSplit(Test) ##
> Possibly the simplest test you can apply to continuous data - an axis-aligned split plane. Can also be applied to discrete data if that happens to make sense. This stores which channel to apply the tests to, whilst each test entity is a 8 byte string, encoding an int32 then a float32 - the first indexes the feature to use from the channel, the second the offset, such that an input has this value subtracted and then fails the test if the result is less than zero or passes if it is greater than or equal to.

**`__init__(self, channel)`**
> Needs to know which channel this test is applied to.

**`do(self, test, es, index = slice(None, None, None))`**
> Does the test. Given the entity that defines the actual test and an ExemplarSet to run the test on. An optional index for the features can be provided (Passed directly through to the exemplar sets [.md](.md) operator, so its indexing rules apply.), but if omitted it runs for all. Return value is a boolean numpy array indexed by the relative exemplar indices, that gives True if it passsed the test, False if it failed.

**`testCodeC(self, name, exemplar_list)`**
> Provides C code to perform the test - provides a C function that is given the test object as a pointer to the first byte and the index of the exemplar to test; it then returns true or false dependeing on if it passes the test or not. Returned string contains a function with the calling convention `bool <name>(PyObject * data, void * test, size_t test_length, int exemplar)`. data is a python tuple indexed by channel containning the object to be fed to the exemplar access function. To construct this function it needs the return value of listCodeC for an ExemplarSet, so it can get the calling convention to access the channel. When compiled the various functions must be avaliable.

## LinearSplit(Test) ##
> Does a linear split of data based on some small set of values. Can be applied to discrete data, though that would typically be a bit strange. This object stores both the channel to which the test is applied and how many dimensions are used, whilst the test entity is a string encoding three things in sequence. First are the int32 indices of the features from the exemplars channel to use, second are the float32 values forming the vector that is dot producted with the extracted values to project to the line perpendicular to the plane, and finally the float32 offset, subtracted from the line position to make it a negative to fail, zero or positive to pass decision.

**`__init__(self, channel, dims)`**
> Needs to know which channel it is applied to and how many dimensions are to be considered.

**`do(self, test, es, index = slice(None, None, None))`**
> Does the test. Given the entity that defines the actual test and an ExemplarSet to run the test on. An optional index for the features can be provided (Passed directly through to the exemplar sets [.md](.md) operator, so its indexing rules apply.), but if omitted it runs for all. Return value is a boolean numpy array indexed by the relative exemplar indices, that gives True if it passsed the test, False if it failed.

**`testCodeC(self, name, exemplar_list)`**
> Provides C code to perform the test - provides a C function that is given the test object as a pointer to the first byte and the index of the exemplar to test; it then returns true or false dependeing on if it passes the test or not. Returned string contains a function with the calling convention `bool <name>(PyObject * data, void * test, size_t test_length, int exemplar)`. data is a python tuple indexed by channel containning the object to be fed to the exemplar access function. To construct this function it needs the return value of listCodeC for an ExemplarSet, so it can get the calling convention to access the channel. When compiled the various functions must be avaliable.

## DiscreteBucket(Test) ##
> For discrete values. The test is applied to a single value, and consists of a list of values such that if it is equal to one of them it passes, but if it is not equal to any of them it fails. Basically a binary split of categorical data. The test entity is a string encoding first a int32 of the index of which feature to use, followed by the remainder of the string forming a list of int32's that constitute the values that result in success.

**`__init__(self, channel)`**
> Needs to know which channel this test is applied to.

**`do(self, test, es, index = slice(None, None, None))`**
> Does the test. Given the entity that defines the actual test and an ExemplarSet to run the test on. An optional index for the features can be provided (Passed directly through to the exemplar sets [.md](.md) operator, so its indexing rules apply.), but if omitted it runs for all. Return value is a boolean numpy array indexed by the relative exemplar indices, that gives True if it passsed the test, False if it failed.

**`testCodeC(self, name, exemplar_list)`**
> Provides C code to perform the test - provides a C function that is given the test object as a pointer to the first byte and the index of the exemplar to test; it then returns true or false dependeing on if it passes the test or not. Returned string contains a function with the calling convention `bool <name>(PyObject * data, void * test, size_t test_length, int exemplar)`. data is a python tuple indexed by channel containning the object to be fed to the exemplar access function. To construct this function it needs the return value of listCodeC for an ExemplarSet, so it can get the calling convention to access the channel. When compiled the various functions must be avaliable.

## Generator() ##
> A generator - provides lots of test entities designed to split an exemplar set via a (python) generator method (i.e. using yield). When a tree is constructed it is provided with a generator and each time it wants to split the generator is given the exemplar set and an index into the relevent exemplars to split on, plus an optional weighting. It then yields a set of test entities, which are applied and scored via the goal, such that the best can be selected. This is more inline with extremelly random decision forests, but there is nothing stopping the use of a goal-aware test generator that does do some kind of optimisation, potentially yielding just one test entity. The generator will contain the most important parameters of the decision forest, as it controls how the test entities are created and how many are tried - selecting the right generator and its associated parameters is essential for performance. An actual Generator is expected to also inherit from its associated Test object, such that it provides the do method. This is necesary as a test entity requires access to its associated Test object to work.

**`clone(self)`**
> Returns a (deep) copy of this object.

**`genCodeC(self, name, exemplar_list)`**
> Provides C code for a generator. The return value is a 2-tuple, the first entry containing the code and the second entry `<state>`, the name of a state object to be used by the system. The state object has two public variables for use by the user - `void * test` and `size_t length`. The code itself will contain the definition of `<state>` and two functions: `void <name>_init(<state> & state, PyObject * data, Exemplar * test_set)` and `bool <name>_next(<state> & state, PyObject * data, Exemplar * test_set)`. Usage consists of creating an instance of State and calling `<name>_init` on it, then repeatedly calling `<name>_next` - each time it returns true you can use the variables in `<state>` to get at the test, but when it returns false it is time to stop (And the `<State>` will have been cleaned up.). If it is not avaliable then a NotImplementedError will be raised.

**`itertests(self, es, index, weights)`**
> Generates test entities that split the provided features into two sets, yielding them one at a time such that the caller can select the best, according to the current goal. It really is allowed to do whatever it likes to create these test entities, which means it provides an insane amount of customisation potential, if possibly rather too much choice. es is the exemplar set, whilst index is the set of exemplars within the exemplar set it is creating a test for. weights optionally provides a weight for each exemplar, aligned with es.

## MergeGen(Generator) ##
> As most generators only handle a specific kind of data (discrete, continuous, one channel at a time.) the need arises to merge multiple generators for a given problem, in the sense that when iterating the generators tests it provides the union of all tests by all of the contained generators. Alternativly, the possibility exists to get better results by using multiple generators with different properties, as the best test from all provided will ultimatly be selected. This class merges upto 256 generators as one. The 256 limit comes from the fact the test entities provided by it have to encode which generator made them, so that the do method can send the test entity to the right test object, and it only uses a byte - in the unlikelly event that more are needed a hierarchy can be used, though your almost certainly doing it wrong if you get that far.

**`__init__(self, *args)`**
> By default constructs the object without any generators in it, but you can provide generators to it as parameters to the constructor.

**`add(self, gen)`**
> Adds a generator to the provided set. Generators can be in multiple MergeGen/RandomGen objects, just as long as a loop is not formed.

**`clone(self)`**
> Returns a (deep) copy of this object.

**`do(self, test, es, index = slice(None, None, None))`**
> None

**`genCodeC(self, name, exemplar_list)`**
> Provides C code for a generator. The return value is a 2-tuple, the first entry containing the code and the second entry `<state>`, the name of a state object to be used by the system. The state object has two public variables for use by the user - `void * test` and `size_t length`. The code itself will contain the definition of `<state>` and two functions: `void <name>_init(<state> & state, PyObject * data, Exemplar * test_set)` and `bool <name>_next(<state> & state, PyObject * data, Exemplar * test_set)`. Usage consists of creating an instance of State and calling `<name>_init` on it, then repeatedly calling `<name>_next` - each time it returns true you can use the variables in `<state>` to get at the test, but when it returns false it is time to stop (And the `<State>` will have been cleaned up.). If it is not avaliable then a NotImplementedError will be raised.

**`itertests(self, es, index, weights)`**
> Generates test entities that split the provided features into two sets, yielding them one at a time such that the caller can select the best, according to the current goal. It really is allowed to do whatever it likes to create these test entities, which means it provides an insane amount of customisation potential, if possibly rather too much choice. es is the exemplar set, whilst index is the set of exemplars within the exemplar set it is creating a test for. weights optionally provides a weight for each exemplar, aligned with es.

**`testCodeC(self, name, exemplar_list)`**
> None

## RandomGen(Generator) ##
> This generator contains several generators, and randomly selects one to provide the tests each time itertests is called - not entirly sure what this could be used for, but it can certainly add some more randomness, for good or for bad. Supports weighting and merging multiple draws from the set of generators contained within. Has the same limit of 256 that MergeGen has, for the same reasons.

**`__init__(self, draws = 1, *args)`**
> draws is the number of draws from the list of generators to merge to provide the final output. Note that it is drawing with replacement, and will call an underlying generator twice if it gets selected twice. After the draws parameter you can optionally provide generators, which will be put into the created object, noting that they will all have a selection weight of 1.

**`add(self, gen, weight = 1.0)`**
> Adds a generator to the provided set. Generators can be in multiple MergeGen/RandomGen objects, just as long as a loop is not formed. You can also provide a weight, to bias how often particular generators are selected.

**`clone(self)`**
> Returns a (deep) copy of this object.

**`do(self, test, es, index = slice(None, None, None))`**
> None

**`genCodeC(self, name, exemplar_list)`**
> Provides C code for a generator. The return value is a 2-tuple, the first entry containing the code and the second entry `<state>`, the name of a state object to be used by the system. The state object has two public variables for use by the user - `void * test` and `size_t length`. The code itself will contain the definition of `<state>` and two functions: `void <name>_init(<state> & state, PyObject * data, Exemplar * test_set)` and `bool <name>_next(<state> & state, PyObject * data, Exemplar * test_set)`. Usage consists of creating an instance of State and calling `<name>_init` on it, then repeatedly calling `<name>_next` - each time it returns true you can use the variables in `<state>` to get at the test, but when it returns false it is time to stop (And the `<State>` will have been cleaned up.). If it is not avaliable then a NotImplementedError will be raised.

**`itertests(self, es, index, weights)`**
> Generates test entities that split the provided features into two sets, yielding them one at a time such that the caller can select the best, according to the current goal. It really is allowed to do whatever it likes to create these test entities, which means it provides an insane amount of customisation potential, if possibly rather too much choice. es is the exemplar set, whilst index is the set of exemplars within the exemplar set it is creating a test for. weights optionally provides a weight for each exemplar, aligned with es.

**`testCodeC(self, name, exemplar_list)`**
> None

## AxisMedianGen(Generator, AxisSplit, Test) ##
> Provides a generator for axis-aligned split planes that split the data set in half, i.e. uses the median. Has random selection of the dimension to split the axis on.

**`__init__(self, channel, count, ignoreWeights = False)`**
> channel is which channel to select the values from, whilst count is how many tests it will return, where each has been constructed around a randomly selected feature from the channel. Setting ignore weights to True means it will not consider the weights when calculating the median.

**`clone(self)`**
> Returns a (deep) copy of this object.

**`do(self, test, es, index = slice(None, None, None))`**
> Does the test. Given the entity that defines the actual test and an ExemplarSet to run the test on. An optional index for the features can be provided (Passed directly through to the exemplar sets [.md](.md) operator, so its indexing rules apply.), but if omitted it runs for all. Return value is a boolean numpy array indexed by the relative exemplar indices, that gives True if it passsed the test, False if it failed.

**`genCodeC(self, name, exemplar_list)`**
> Provides C code for a generator. The return value is a 2-tuple, the first entry containing the code and the second entry `<state>`, the name of a state object to be used by the system. The state object has two public variables for use by the user - `void * test` and `size_t length`. The code itself will contain the definition of `<state>` and two functions: `void <name>_init(<state> & state, PyObject * data, Exemplar * test_set)` and `bool <name>_next(<state> & state, PyObject * data, Exemplar * test_set)`. Usage consists of creating an instance of State and calling `<name>_init` on it, then repeatedly calling `<name>_next` - each time it returns true you can use the variables in `<state>` to get at the test, but when it returns false it is time to stop (And the `<State>` will have been cleaned up.). If it is not avaliable then a NotImplementedError will be raised.

**`itertests(self, es, index, weights)`**
> Generates test entities that split the provided features into two sets, yielding them one at a time such that the caller can select the best, according to the current goal. It really is allowed to do whatever it likes to create these test entities, which means it provides an insane amount of customisation potential, if possibly rather too much choice. es is the exemplar set, whilst index is the set of exemplars within the exemplar set it is creating a test for. weights optionally provides a weight for each exemplar, aligned with es.

**`testCodeC(self, name, exemplar_list)`**
> Provides C code to perform the test - provides a C function that is given the test object as a pointer to the first byte and the index of the exemplar to test; it then returns true or false dependeing on if it passes the test or not. Returned string contains a function with the calling convention `bool <name>(PyObject * data, void * test, size_t test_length, int exemplar)`. data is a python tuple indexed by channel containning the object to be fed to the exemplar access function. To construct this function it needs the return value of listCodeC for an ExemplarSet, so it can get the calling convention to access the channel. When compiled the various functions must be avaliable.

## LinearMedianGen(Generator, LinearSplit, Test) ##
> Provides a generator for split planes that uses the median of the features projected perpendicular to the plane direction, such that it splits the data set in half. Randomly selects which dimensions to work with and the orientation of the split plane.

**`__init__(self, channel, dims, dimCount, dirCount, ignoreWeights = False)`**
> channel is which channel to select for and dims how many features (dimensions) to test on for any given test. dimCount is how many sets of dimensions to randomly select to generate tests for, whilst dirCount is how many random dimensions (From a uniform distribution over a hyper-sphere.) to try. It actually generates the two independantly and trys every combination, as generating uniform random directions is somewhat expensive. Setting ignore weights to True means it will not consider the weights when calculating the median.

**`clone(self)`**
> Returns a (deep) copy of this object.

**`do(self, test, es, index = slice(None, None, None))`**
> Does the test. Given the entity that defines the actual test and an ExemplarSet to run the test on. An optional index for the features can be provided (Passed directly through to the exemplar sets [.md](.md) operator, so its indexing rules apply.), but if omitted it runs for all. Return value is a boolean numpy array indexed by the relative exemplar indices, that gives True if it passsed the test, False if it failed.

**`genCodeC(self, name, exemplar_list)`**
> Provides C code for a generator. The return value is a 2-tuple, the first entry containing the code and the second entry `<state>`, the name of a state object to be used by the system. The state object has two public variables for use by the user - `void * test` and `size_t length`. The code itself will contain the definition of `<state>` and two functions: `void <name>_init(<state> & state, PyObject * data, Exemplar * test_set)` and `bool <name>_next(<state> & state, PyObject * data, Exemplar * test_set)`. Usage consists of creating an instance of State and calling `<name>_init` on it, then repeatedly calling `<name>_next` - each time it returns true you can use the variables in `<state>` to get at the test, but when it returns false it is time to stop (And the `<State>` will have been cleaned up.). If it is not avaliable then a NotImplementedError will be raised.

**`itertests(self, es, index, weights)`**
> Generates test entities that split the provided features into two sets, yielding them one at a time such that the caller can select the best, according to the current goal. It really is allowed to do whatever it likes to create these test entities, which means it provides an insane amount of customisation potential, if possibly rather too much choice. es is the exemplar set, whilst index is the set of exemplars within the exemplar set it is creating a test for. weights optionally provides a weight for each exemplar, aligned with es.

**`testCodeC(self, name, exemplar_list)`**
> Provides C code to perform the test - provides a C function that is given the test object as a pointer to the first byte and the index of the exemplar to test; it then returns true or false dependeing on if it passes the test or not. Returned string contains a function with the calling convention `bool <name>(PyObject * data, void * test, size_t test_length, int exemplar)`. data is a python tuple indexed by channel containning the object to be fed to the exemplar access function. To construct this function it needs the return value of listCodeC for an ExemplarSet, so it can get the calling convention to access the channel. When compiled the various functions must be avaliable.

## AxisRandomGen(Generator, AxisSplit, Test) ##
> Provides a generator for axis-aligned split planes that split the data set at random - uses a normal distribution constructed from the data. Has random selection of the dimension to split the axis on.

**`__init__(self, channel, dimCount, splitCount, ignoreWeights = False)`**
> channel is which channel to select the values from; dimCount is how many dimensions to try splits on; splitCount how many random split points to try for each selected dimension. Setting ignore weights to True means it will not consider the weights when calculating the normal distribution to draw random split points from.

**`clone(self)`**
> Returns a (deep) copy of this object.

**`do(self, test, es, index = slice(None, None, None))`**
> Does the test. Given the entity that defines the actual test and an ExemplarSet to run the test on. An optional index for the features can be provided (Passed directly through to the exemplar sets [.md](.md) operator, so its indexing rules apply.), but if omitted it runs for all. Return value is a boolean numpy array indexed by the relative exemplar indices, that gives True if it passsed the test, False if it failed.

**`genCodeC(self, name, exemplar_list)`**
> Provides C code for a generator. The return value is a 2-tuple, the first entry containing the code and the second entry `<state>`, the name of a state object to be used by the system. The state object has two public variables for use by the user - `void * test` and `size_t length`. The code itself will contain the definition of `<state>` and two functions: `void <name>_init(<state> & state, PyObject * data, Exemplar * test_set)` and `bool <name>_next(<state> & state, PyObject * data, Exemplar * test_set)`. Usage consists of creating an instance of State and calling `<name>_init` on it, then repeatedly calling `<name>_next` - each time it returns true you can use the variables in `<state>` to get at the test, but when it returns false it is time to stop (And the `<State>` will have been cleaned up.). If it is not avaliable then a NotImplementedError will be raised.

**`itertests(self, es, index, weights)`**
> Generates test entities that split the provided features into two sets, yielding them one at a time such that the caller can select the best, according to the current goal. It really is allowed to do whatever it likes to create these test entities, which means it provides an insane amount of customisation potential, if possibly rather too much choice. es is the exemplar set, whilst index is the set of exemplars within the exemplar set it is creating a test for. weights optionally provides a weight for each exemplar, aligned with es.

**`testCodeC(self, name, exemplar_list)`**
> Provides C code to perform the test - provides a C function that is given the test object as a pointer to the first byte and the index of the exemplar to test; it then returns true or false dependeing on if it passes the test or not. Returned string contains a function with the calling convention `bool <name>(PyObject * data, void * test, size_t test_length, int exemplar)`. data is a python tuple indexed by channel containning the object to be fed to the exemplar access function. To construct this function it needs the return value of listCodeC for an ExemplarSet, so it can get the calling convention to access the channel. When compiled the various functions must be avaliable.

## LinearRandomGen(Generator, LinearSplit, Test) ##
> Provides a generator for split planes that it is entirly random. Randomly selects which dimensions to work with, the orientation of the split plane and then where to put the split plane, with this last bit done with a normal distribution.

**`__init__(self, channel, dims, dimCount, dirCount, splitCount, ignoreWeights = False)`**
> channel is which channel to select for and dims how many features (dimensions) to test on for any given test. dimCount is how many sets of dimensions to randomly select to generate tests from, whilst dirCount is how many random dimensions (From a uniform distribution over a hyper-sphere.) to use for selection. It actually generates the two independantly and trys every combination, as generating uniform random directions is somewhat expensive. For each of these splitCount split points are then tried, as drawn from a normal distribution. Setting ignore weights to True means it will not consider the weights when calculating the normal distribution to draw random split points from.

**`clone(self)`**
> Returns a (deep) copy of this object.

**`do(self, test, es, index = slice(None, None, None))`**
> Does the test. Given the entity that defines the actual test and an ExemplarSet to run the test on. An optional index for the features can be provided (Passed directly through to the exemplar sets [.md](.md) operator, so its indexing rules apply.), but if omitted it runs for all. Return value is a boolean numpy array indexed by the relative exemplar indices, that gives True if it passsed the test, False if it failed.

**`genCodeC(self, name, exemplar_list)`**
> Provides C code for a generator. The return value is a 2-tuple, the first entry containing the code and the second entry `<state>`, the name of a state object to be used by the system. The state object has two public variables for use by the user - `void * test` and `size_t length`. The code itself will contain the definition of `<state>` and two functions: `void <name>_init(<state> & state, PyObject * data, Exemplar * test_set)` and `bool <name>_next(<state> & state, PyObject * data, Exemplar * test_set)`. Usage consists of creating an instance of State and calling `<name>_init` on it, then repeatedly calling `<name>_next` - each time it returns true you can use the variables in `<state>` to get at the test, but when it returns false it is time to stop (And the `<State>` will have been cleaned up.). If it is not avaliable then a NotImplementedError will be raised.

**`itertests(self, es, index, weights)`**
> Generates test entities that split the provided features into two sets, yielding them one at a time such that the caller can select the best, according to the current goal. It really is allowed to do whatever it likes to create these test entities, which means it provides an insane amount of customisation potential, if possibly rather too much choice. es is the exemplar set, whilst index is the set of exemplars within the exemplar set it is creating a test for. weights optionally provides a weight for each exemplar, aligned with es.

**`testCodeC(self, name, exemplar_list)`**
> Provides C code to perform the test - provides a C function that is given the test object as a pointer to the first byte and the index of the exemplar to test; it then returns true or false dependeing on if it passes the test or not. Returned string contains a function with the calling convention `bool <name>(PyObject * data, void * test, size_t test_length, int exemplar)`. data is a python tuple indexed by channel containning the object to be fed to the exemplar access function. To construct this function it needs the return value of listCodeC for an ExemplarSet, so it can get the calling convention to access the channel. When compiled the various functions must be avaliable.

## DiscreteRandomGen(Generator, DiscreteBucket, Test) ##
> Defines a generator for discrete data. It basically takes a single discrete feature and randomly assigns just one value to pass and all others to fail the test. The selection is from the values provided by the data passed in, weighted by how many of them there are.

**`__init__(self, channel, featCount, valueCount)`**
> channel is the channel to build discrete tests for. featCount is how many different features to select to generate tests for whilst valueCount is how many values to draw and offer as tests for each feature selected.

**`clone(self)`**
> Returns a (deep) copy of this object.

**`do(self, test, es, index = slice(None, None, None))`**
> Does the test. Given the entity that defines the actual test and an ExemplarSet to run the test on. An optional index for the features can be provided (Passed directly through to the exemplar sets [.md](.md) operator, so its indexing rules apply.), but if omitted it runs for all. Return value is a boolean numpy array indexed by the relative exemplar indices, that gives True if it passsed the test, False if it failed.

**`genCodeC(self, name, exemplar_list)`**
> Provides C code for a generator. The return value is a 2-tuple, the first entry containing the code and the second entry `<state>`, the name of a state object to be used by the system. The state object has two public variables for use by the user - `void * test` and `size_t length`. The code itself will contain the definition of `<state>` and two functions: `void <name>_init(<state> & state, PyObject * data, Exemplar * test_set)` and `bool <name>_next(<state> & state, PyObject * data, Exemplar * test_set)`. Usage consists of creating an instance of State and calling `<name>_init` on it, then repeatedly calling `<name>_next` - each time it returns true you can use the variables in `<state>` to get at the test, but when it returns false it is time to stop (And the `<State>` will have been cleaned up.). If it is not avaliable then a NotImplementedError will be raised.

**`itertests(self, es, index, weights)`**
> Generates test entities that split the provided features into two sets, yielding them one at a time such that the caller can select the best, according to the current goal. It really is allowed to do whatever it likes to create these test entities, which means it provides an insane amount of customisation potential, if possibly rather too much choice. es is the exemplar set, whilst index is the set of exemplars within the exemplar set it is creating a test for. weights optionally provides a weight for each exemplar, aligned with es.

**`testCodeC(self, name, exemplar_list)`**
> Provides C code to perform the test - provides a C function that is given the test object as a pointer to the first byte and the index of the exemplar to test; it then returns true or false dependeing on if it passes the test or not. Returned string contains a function with the calling convention `bool <name>(PyObject * data, void * test, size_t test_length, int exemplar)`. data is a python tuple indexed by channel containning the object to be fed to the exemplar access function. To construct this function it needs the return value of listCodeC for an ExemplarSet, so it can get the calling convention to access the channel. When compiled the various functions must be avaliable.

## AxisClassifyGen(Generator, AxisSplit, Test) ##
> Provides a generator that creates axis-aligned split planes that have their position selected to maximise the information gain with respect to the task of classification.

**`__init__(self, channel, catChannel, count)`**
> channel is which channel to select the values from; catChannel contains the true classes of the features so the split can be optimised; and count is how many tests it will return, where each has been constructed around a randomly selected feature from the channel.

**`clone(self)`**
> Returns a (deep) copy of this object.

**`do(self, test, es, index = slice(None, None, None))`**
> Does the test. Given the entity that defines the actual test and an ExemplarSet to run the test on. An optional index for the features can be provided (Passed directly through to the exemplar sets [.md](.md) operator, so its indexing rules apply.), but if omitted it runs for all. Return value is a boolean numpy array indexed by the relative exemplar indices, that gives True if it passsed the test, False if it failed.

**`genCodeC(self, name, exemplar_list)`**
> Provides C code for a generator. The return value is a 2-tuple, the first entry containing the code and the second entry `<state>`, the name of a state object to be used by the system. The state object has two public variables for use by the user - `void * test` and `size_t length`. The code itself will contain the definition of `<state>` and two functions: `void <name>_init(<state> & state, PyObject * data, Exemplar * test_set)` and `bool <name>_next(<state> & state, PyObject * data, Exemplar * test_set)`. Usage consists of creating an instance of State and calling `<name>_init` on it, then repeatedly calling `<name>_next` - each time it returns true you can use the variables in `<state>` to get at the test, but when it returns false it is time to stop (And the `<State>` will have been cleaned up.). If it is not avaliable then a NotImplementedError will be raised.

**`itertests(self, es, index, weights)`**
> Generates test entities that split the provided features into two sets, yielding them one at a time such that the caller can select the best, according to the current goal. It really is allowed to do whatever it likes to create these test entities, which means it provides an insane amount of customisation potential, if possibly rather too much choice. es is the exemplar set, whilst index is the set of exemplars within the exemplar set it is creating a test for. weights optionally provides a weight for each exemplar, aligned with es.

**`testCodeC(self, name, exemplar_list)`**
> Provides C code to perform the test - provides a C function that is given the test object as a pointer to the first byte and the index of the exemplar to test; it then returns true or false dependeing on if it passes the test or not. Returned string contains a function with the calling convention `bool <name>(PyObject * data, void * test, size_t test_length, int exemplar)`. data is a python tuple indexed by channel containning the object to be fed to the exemplar access function. To construct this function it needs the return value of listCodeC for an ExemplarSet, so it can get the calling convention to access the channel. When compiled the various functions must be avaliable.

## LinearClassifyGen(Generator, LinearSplit, Test) ##
> Provides a generator for split planes that projected the features perpendicular to a random plane direction but then optimises where to put the split plane to maximise classification performance. Randomly selects which dimensions to work with and the orientation of the split plane.

**`__init__(self, channel, catChannel, dims, dimCount, dirCount)`**
> channel is which channel to select for and catChannel the channel to get the classification answers from. dims is how many features (dimensions) to test on for any given test. dimCount is how many sets of dimensions to randomly select to generate tests for, whilst dirCount is how many random dimensions (From a uniform distribution over a hyper-sphere.) to try. It actually generates the two independantly and trys every combination, as generating uniform random directions is somewhat expensive.

**`clone(self)`**
> Returns a (deep) copy of this object.

**`do(self, test, es, index = slice(None, None, None))`**
> Does the test. Given the entity that defines the actual test and an ExemplarSet to run the test on. An optional index for the features can be provided (Passed directly through to the exemplar sets [.md](.md) operator, so its indexing rules apply.), but if omitted it runs for all. Return value is a boolean numpy array indexed by the relative exemplar indices, that gives True if it passsed the test, False if it failed.

**`genCodeC(self, name, exemplar_list)`**
> Provides C code for a generator. The return value is a 2-tuple, the first entry containing the code and the second entry `<state>`, the name of a state object to be used by the system. The state object has two public variables for use by the user - `void * test` and `size_t length`. The code itself will contain the definition of `<state>` and two functions: `void <name>_init(<state> & state, PyObject * data, Exemplar * test_set)` and `bool <name>_next(<state> & state, PyObject * data, Exemplar * test_set)`. Usage consists of creating an instance of State and calling `<name>_init` on it, then repeatedly calling `<name>_next` - each time it returns true you can use the variables in `<state>` to get at the test, but when it returns false it is time to stop (And the `<State>` will have been cleaned up.). If it is not avaliable then a NotImplementedError will be raised.

**`itertests(self, es, index, weights)`**
> Generates test entities that split the provided features into two sets, yielding them one at a time such that the caller can select the best, according to the current goal. It really is allowed to do whatever it likes to create these test entities, which means it provides an insane amount of customisation potential, if possibly rather too much choice. es is the exemplar set, whilst index is the set of exemplars within the exemplar set it is creating a test for. weights optionally provides a weight for each exemplar, aligned with es.

**`testCodeC(self, name, exemplar_list)`**
> Provides C code to perform the test - provides a C function that is given the test object as a pointer to the first byte and the index of the exemplar to test; it then returns true or false dependeing on if it passes the test or not. Returned string contains a function with the calling convention `bool <name>(PyObject * data, void * test, size_t test_length, int exemplar)`. data is a python tuple indexed by channel containning the object to be fed to the exemplar access function. To construct this function it needs the return value of listCodeC for an ExemplarSet, so it can get the calling convention to access the channel. When compiled the various functions must be avaliable.

## DiscreteClassifyGen(Generator, DiscreteBucket, Test) ##
> Defines a generator for discrete data. It basically takes a single discrete feature and then greedily optimises to get the best classification performance, As it won't necesarilly converge to the global optimum multiple restarts are provided. The discrete values must form a contiguous set, starting at 0 and going upwards. When splitting it only uses values it can see - unseen values will fail the test, though it always arranges for the most informative half to be the one that passes the test.

**`__init__(self, channel, catChannel, featCount, initCount)`**
> channel is the channel to build discrete tests for; featCount is how many random features to randomly select and initCount how many random initialisations to try for each feature.

**`clone(self)`**
> Returns a (deep) copy of this object.

**`do(self, test, es, index = slice(None, None, None))`**
> Does the test. Given the entity that defines the actual test and an ExemplarSet to run the test on. An optional index for the features can be provided (Passed directly through to the exemplar sets [.md](.md) operator, so its indexing rules apply.), but if omitted it runs for all. Return value is a boolean numpy array indexed by the relative exemplar indices, that gives True if it passsed the test, False if it failed.

**`genCodeC(self, name, exemplar_list)`**
> Provides C code for a generator. The return value is a 2-tuple, the first entry containing the code and the second entry `<state>`, the name of a state object to be used by the system. The state object has two public variables for use by the user - `void * test` and `size_t length`. The code itself will contain the definition of `<state>` and two functions: `void <name>_init(<state> & state, PyObject * data, Exemplar * test_set)` and `bool <name>_next(<state> & state, PyObject * data, Exemplar * test_set)`. Usage consists of creating an instance of State and calling `<name>_init` on it, then repeatedly calling `<name>_next` - each time it returns true you can use the variables in `<state>` to get at the test, but when it returns false it is time to stop (And the `<State>` will have been cleaned up.). If it is not avaliable then a NotImplementedError will be raised.

**`itertests(self, es, index, weights)`**
> Generates test entities that split the provided features into two sets, yielding them one at a time such that the caller can select the best, according to the current goal. It really is allowed to do whatever it likes to create these test entities, which means it provides an insane amount of customisation potential, if possibly rather too much choice. es is the exemplar set, whilst index is the set of exemplars within the exemplar set it is creating a test for. weights optionally provides a weight for each exemplar, aligned with es.

**`testCodeC(self, name, exemplar_list)`**
> Provides C code to perform the test - provides a C function that is given the test object as a pointer to the first byte and the index of the exemplar to test; it then returns true or false dependeing on if it passes the test or not. Returned string contains a function with the calling convention `bool <name>(PyObject * data, void * test, size_t test_length, int exemplar)`. data is a python tuple indexed by channel containning the object to be fed to the exemplar access function. To construct this function it needs the return value of listCodeC for an ExemplarSet, so it can get the calling convention to access the channel. When compiled the various functions must be avaliable.

## SVMClassifyGen(Generator, Test) ##
> Allows you to use the SVM library as a classifier for a node. Note that it detects if the SVM library is avaliable - if not then this class will not exist. Be warned that its quite memory intensive, as it just wraps the SVM objects without any clever packing. Works by randomly selecting a class to seperate and training a one vs all classifier, with random parameters on random features. Parameters are quite complicated, due to all the svm options and randomness control being extensive.

**`__init__(self, params, paramDraw, catChannel, catDraw, featChannel, featCount, featDraw)`**
> There are three parts - the svm parameters to use, the class to seperate and the features to train on, all of which allow for the introduction of randomness. The svm parameters are controlled by params - it must be either a single svm.Params or a list of them, which includes things like parameter sets provided by the svm library. For each test generation paramDraw parameter options are selected randomly from params and tried combinatorically with the other two parts. The class of each feature must be provided, as an integer in channel catChannel. For each test generation it selects one class randomly from the classes exhibited by the features, which it does catDraw times, combinatorically with the other two parts. The features to train on are found in channel featChannel, and it randomly selects featCount of them to be used for each trainning run, which it does featDraw times combinatorically with the other two parameters. Each time classifiers are generated it will produce the product of the three **Draw parameters generators, where it draws each set once and then tries all combinations between the three.**

**`clone(self)`**
> Returns a (deep) copy of this object.

**`do(self, test, es, index = slice(None, None, None))`**
> Does the test. Given the entity that defines the actual test and an ExemplarSet to run the test on. An optional index for the features can be provided (Passed directly through to the exemplar sets [.md](.md) operator, so its indexing rules apply.), but if omitted it runs for all. Return value is a boolean numpy array indexed by the relative exemplar indices, that gives True if it passsed the test, False if it failed.

**`genCodeC(self, name, exemplar_list)`**
> Provides C code for a generator. The return value is a 2-tuple, the first entry containing the code and the second entry `<state>`, the name of a state object to be used by the system. The state object has two public variables for use by the user - `void * test` and `size_t length`. The code itself will contain the definition of `<state>` and two functions: `void <name>_init(<state> & state, PyObject * data, Exemplar * test_set)` and `bool <name>_next(<state> & state, PyObject * data, Exemplar * test_set)`. Usage consists of creating an instance of State and calling `<name>_init` on it, then repeatedly calling `<name>_next` - each time it returns true you can use the variables in `<state>` to get at the test, but when it returns false it is time to stop (And the `<State>` will have been cleaned up.). If it is not avaliable then a NotImplementedError will be raised.

**`itertests(self, es, index, weights)`**
> Generates test entities that split the provided features into two sets, yielding them one at a time such that the caller can select the best, according to the current goal. It really is allowed to do whatever it likes to create these test entities, which means it provides an insane amount of customisation potential, if possibly rather too much choice. es is the exemplar set, whilst index is the set of exemplars within the exemplar set it is creating a test for. weights optionally provides a weight for each exemplar, aligned with es.

**`testCodeC(self, name, exemplar_list)`**
> Provides C code to perform the test - provides a C function that is given the test object as a pointer to the first byte and the index of the exemplar to test; it then returns true or false dependeing on if it passes the test or not. Returned string contains a function with the calling convention `bool <name>(PyObject * data, void * test, size_t test_length, int exemplar)`. data is a python tuple indexed by channel containning the object to be fed to the exemplar access function. To construct this function it needs the return value of listCodeC for an ExemplarSet, so it can get the calling convention to access the channel. When compiled the various functions must be avaliable.

## Node() ##
> Defines a node - these are the bread and butter of the system. Each decision tree is made out of nodes, each of which contains a binary test - if a feature vector passes the test then it travels to the true child node; if it fails it travels to the false child node (Note lowercase to avoid reserved word clash.). Eventually a leaf node is reached, where test==None, at which point the stats object is obtained, merged with the equivalent for all decision trees, and then provided as the answer to the user. Note that this python object uses the slots techneque to keep it small - there will often be many thousands of these in a trained model.

**`__init__(self, goal, gen, pruner, es, index = slice(None, None, None), weights, depth = 0, stats, entropy, code)`**
> This recursivly grows the tree until the pruner says to stop. goal is a Goal object, so it knows what to optimise, gen a Generator object that provides tests for it to choose between and pruner is a Pruner object that decides when to stop growing. The exemplar set to train on is then provided, optionally with the indices of which members to use and weights to assign to them (weights align with the exemplar set, not with the relative exemplar indices defined by index. depth is the depth of this node - part of the recursive construction and used by the pruner as a possible reason to stop growing. stats is optionally provided to save on duplicate calculation, as it will be calculated as part of working out the split. entropy should match up with stats. The static method initC can be called to generate code that can be used by this constructor to accelerate test selection, but only if it is passed in.

**`addTrain(self, goal, gen, es, index = slice(None, None, None), weights, code)`**
> This allows you to update the nodes with more data, as though it was used for trainning. The actual tests are not affected, only the statistics at each node - part of incrimental learning. You can optionally proivde code generated by the addTrainC method to give it go faster stripes.

**`clone(self)`**
> Returns a deep copy of this node. Note that it only copys the nodes - test, stats and summary are all assumed to contain invariant entities that are always replaced, never editted.

**`error(self, goal, gen, es, index = slice(None, None, None), weights, inc = False, store, code)`**
> Once a tree is trained this method allows you to determine how good it is, using a test set, which would typically be its out-of-bag (oob) test set. Given a test set, possibly weighted, it will return its error rate, as defined by the goal. goal is the Goal object used for trainning, gen the Generator. Also supports incrimental testing, where the information gleened from the test set is stored such that new test exemplars can be added. This is the inc variable - True to store this (potentially large) quantity of information, and update it if it already exists, False to not store it and therefore disallow incrimental learning whilst saving memory. Note that the error rate will change by adding more training data as well as more testing data - you can call it with es==None to get an error score without adding more testing exemplars, assuming it has previously been called with inc==True. store is for internal use only. code can be provided by the relevent parameter, as generated by the errorC method, allowing a dramatic speedup.

**`evaluate(self, out, gen, es, index = slice(None, None, None), code)`**
> Given a set of exemplars, and possibly an index, this outputs the infered stats entities. Requires the generator so it can apply the tests. The output goes into out, a list indexed by exemplar position. If code is set to a string generated by evaluateC it uses that, for speed.

**`give_birth(self, goal, gen, pruner, es, index = slice(None, None, None), weights, depth = 0, entropy, code)`**
> This recursivly grows the tree until the pruner says to stop. goal is a Goal object, so it knows what to optimise, gen a Generator object that provides tests for it to choose between and pruner is a Pruner object that decides when to stop growing. The exemplar set to train on is then provided, optionally with the indices of which members to use and weights to assign to them (weights align with the exemplar set, not with the relative exemplar indices defined by index. depth is the depth of this node - part of the recursive construction and used by the pruner as a possible reason to stop growing. entropy should match up with self.stats. The static method initC can be called to generate code that can be used to accelerate test selection, but only if it is passed in.

**`grow(self, goal, gen, pruner, es, index = slice(None, None, None), weights, depth = 0, code)`**
> This is called on a tree that has already grown - it recurses to the children and continues as though growth never stopped. This can be to grow the tree further using a less stritc pruner or to grow the tree after further information has been added. code can be passed in as generated by the initC static method, and will be used to optimise test generation.

**`removeIncError(self)`**
> Culls the information for incrimental testing from the data structure, either to reset ready for new information or just to shrink the data structure after learning is finished.

**`size(self)`**
> Returns how many nodes this (sub-)tree consists of.