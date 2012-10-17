Support Vector Machine implementation, using the Sequential Minimal Optimization technique.

Implemented in python using numpy and scipy.weave.
Only provides classification.

The key papers behind this are:
'Support-vector networks' by Cortes and Vapnik provides the basic soft svm idea.
'Fast Training of Support Vector Machines using Sequential Minimal Optimization' by Platt provides the core solution method.
'Working Set Selection Using Second Order Information for Training Support Vector Machines' by Fan, Chen and Lin provides a better pair selection method for the above algorithm.

svm.py provides all the functionality, and is the only file you need import. Standard usage is to create 3 objects - a parameters object, a dataset object and a model object.

The model object should nearly always be MultiModel, even if you only have two classes, as it provides the best interface (Mostly because it supports model selection and multi-processor solving.).

The Dataset object is constructed using one or more data matrix/label vector pairs.

For parameters you either choose the parameters explicitly using a Params() object, or you construct a ParamsSet() object and do model selection. See their respective interfaces for details.

All objects are designed to be compatible with pickle, for file i/o purposes.
If you are reading readme.txt then you can generate documentation by running make_doc.py, which will include all the relevant interfaces.



svm.py - import file, as covered above.

params.py - parameters object
params_sets.py - objects that emulate being lists of parameters, for model selection.
dataset.py - dataset, of feature vectors with labels.

smo.py - actual solver object.
smo_aux.py - C code strings for solver.
loo.py - Does leave one out, to access how good specific parameters are. Primarilly used for model selection.

model.py - Two class model as provided by the solver.
multiclass.py - Wrapper for multiple models to support one vs one classification; additionally provides model selection and multiprocessor support.

readme.txt - this file, which is copied into the start of svm.html if generated.
make_doc.py - creates the svm.html help file.

test_simple.py - Very simple test of a linear classifier on overlapping uniform distributions.
test_mult.py - Model selection test with four classes with overlapping uniform distributions.
test_curve.py - Test on a sin curve with model selection. Note this requires opencv 2.x to be installed, as it renders out some images.
test_2d.py - Test on a four-class spiral with model selection; also outputs images requiring opencv 2.x.
