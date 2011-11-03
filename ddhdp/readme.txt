Delta-Dual Hierarchical Dirichlet Processes

An extension of DHDP, which is published in the paper 'Delta-Dual Hierarchical Dirichlet Processes: A pragmatic abnormal behaviour detector' by T. S. F. Haines and T. Xiang. A lot of code was copied from my DHDP implementation, so much of the interface is similar/identical. Uses scipy.weave for all the Gibbs sampling.

Files:

ddhdp.py - Includes everything needed to use the system.

dp_conc.py - Contains the values required for the concentration parameter of a DP - its prior and initial value.
params.py - Provides the parameters for the solver that are not specific to the problem.
document.py - Provides an object representing a document and the words it contains.
corpus.py - Contains an object to represent a Corpus - basically all the documents and a bunch of parameters to define exactly what the problem does.
model.py - Contains all the objects that represent a model of the data, as provided by the solvers.


solvers.py - Internal use file; contains the code to detect the best solver available.
solve_shared.py - Contains the python-side representation of the state of a Gibbs sampler.
solve_weave.py - Solver that uses scipy.weave.
solve_weave_mp.py - Multiprocess version of the scipy.weave based solver.

ds_cpp.py - Contains the cpp side representation of the state of a Gibbs sampler.
ds_link_cpp.py - Contains the code to convert between the python and cpp states.

test_lines.py - Simple test file, to verify that it works. Also acts as an example of usage.
test_abnorm_lines.py - Another test file, that tests its ability to learn topics for abnormal behaviour.

readme.txt - This file, which is included in the html documentation.
make_doc.py - Creates the html documentation.

