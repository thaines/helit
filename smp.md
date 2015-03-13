# Sparse Multinomial Posterior #

## Overview ##
**Sparse Multinomial Posterior**

Implements a system to estimate a multinomial given multiple samples drawn from it where those samples are 'sparse', i.e. have had some of the counts thrown away. Actually focuses on obtaining the mean draw for each parameter of the multinomial, as this often allows the draw to be collapsed out in a larger system.

Core is implemented in C++, with a Python wrapper, using scipy.weave.


Files:

`smp.py` - Contains the python interface to the system.

`smp_cpp.py` - The C++ code that does the actual work. Designed so it can be used by scipy.weave code for other modules if need be.

`flag_index_array.py` - A support system; handles the combinations of flags indicating which counts in a draw are known and which are unknown.


`test.py` - Simple test file.


`readme.txt` - This file, which is copied into the html documentation.

`make_doc.py` - Script to generate the html documentation.


---


# Variables #

**`smp_code`**
> String containing the C++ code that does the actual work for the system.


# Classes #

## SMP() ##
> Impliments a Python wrapper around the C++ code for the Sparse Multinomial Posterior. Estimates the multinomial distribution from which various samples have been drawn, where those samples are sparse, i.e. not all counts are provided.

**`__init__(self, fia)`**
> Initialises with a FlagIndexArray object (Which must of had the addSingles method correctly called.) - this specifies the various combinations of counts being provided that are allowed.

**`add(self, fi, counts)`**
> Given the flag index returned from the relevant fia and an array of counts this adds it to the smp.

**`mean(self)`**
> Returns an estimate of the mean for each value of the multinomial, as an array, given the evidence provided. (Will itself sum to one - a necesary consequence of being an average of points constrained to the simplex.

**`reset(self)`**
> Causes a reset, so you may add a new set of samples.

**`setPrior(self, conc, mn)`**
> Sets the prior, as a Dirichlet distribution represented by a concentration and a multinomial distribution. Can leave out the multinomial to just update the concentration.

**`setSampleCount(self, count)`**
> Sets the number of samples to use when approximating the integral.

## FlagIndexArray() ##
> Provides a register for flag lists - given a list of true/false flags gives a unique number for each combination. Requesting the numebr associated with a combination that has already been entered will always return the same number. All flag lists should be the same length and you can obtain a numpy matrix of {0,1} valued unsigned chars where each row corresponds to the flag list with that index. Also has a function to add the flags for each case of only one flag being on, which if called before anything else puts them so the index of the flag and the index of the flag list correspond - a trick required by the rest of the system.

**`__init__(self, length, addSingles = False)`**
> Requires the length of the flag lists. Alternativly it can clone another FlagIndexArray. Will call the addSingles method for you if the flag is set.

**`addFlagIndexArray(self, fia, remap)`**
> Given a flag index array this merges its flags into the new flags, returning a dictionary indexed by fia's indices that converts them to the new indices in self. remap is optionally a dictionary converting flag indices in fia to flag indexes in self - remap[index](fia.md) = self index.

**`addSingles(self)`**
> Adds the entries where only a single flag is set, with the index of the flag list set to match the index of the flag that is set. Must be called first, before flagIndex is ever called.

**`flagCount(self)`**
> Returns the number of flag lists that are in the system.

**`flagIndex(self, flags, create = True)`**
> Given a flag list returns its index - if it has been previously supplied then it will be the same index, otherwise a new one. Can be passed any entity that can be indexed via [.md](.md) to get the integers {0,1}. Returns a natural. If the create flag is set to False in the event of a previously unseen flag list it will raise an exception instead of assigning it a new natural.

**`getFlagMatrix(self)`**
> Returns a 2D numpy array of type numpy.uint8 containing {0,1}, indexed by [index,flag entry](flag.md) - basically all the flags stacked into a single matrix and indexed by the entries returned by flagIndex. Often refered to as a 'flag index array' (fia).

**`getLength(self)`**
> Return the length that all flag lists should be.