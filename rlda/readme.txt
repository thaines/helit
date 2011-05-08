RLDA implementation

The code for the paper 'Video Topic Modelling with Behavioural Segmentation' by T. S. F. Haines and T. Xiang.

Theoretically performs a behavioural segmentation whilst constructing a topic model, in a single formulation. In practise its better described as a topic model with a slightly more complicated topic distribution, with some crazy interactions. Whilst the idea is sound this model and implementation do not perform as well as I would of liked (Tested on traffic cctv data.) - its better than straight LDA, but not by much (Mostly because it has better generalisation to larger regions, i.e. its less sensitive to vehicles that are large or driving in the right way but slightly out of position.).

Implemented using Gibbs sampling, using python with scipy, including scipy.weave. All of the tests use open cv for visualisation. Interface is extremely similar to my LDA implementation.

rlda.py - pulls together everything into a single convenient namespace, and handles details with regards to selecting a solver.

document.py - contains the data structure to represent a document.
corpus.py - defines a corpus, as a set of documents and a model if it has been solved.
params.py - the object that represents the parameters for running the algorithm.

solve_shared.py - The data structure and other shared stuff used by all solving methods (Even though their is really only one.).
solve_weave.py - The single process solver.
solve_weave_mp.py - The multi-process solver.

test_cross.py - A very simple test case. Probably a good reference to figuring out how to use the implementation.
test_cross_dual.py - A variant of the simple test case.
test_cross_fat.py - Another variant of the simple test case.
test_junction.py - More sophisticated test that simulates a traffic junction.

readme.txt - This file, which is also copied into the html documentation.
make_doc.py - The file that generates the html documentation.
