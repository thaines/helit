Fast Random Forest

A straight random forest implementation created for when I just want a classifier or a regressor. Not designed for flexibility like my other two implementations in other words, though still plenty powerful. Was actually created in frustration at the scikit learn one - I found myself spending more time loading the model from disk (because scikit learn requires the use, at least at the time of writing, of the awful Python pickle system) than actually using it. 'Fast' actually refers to its ability to load a model really quickly - I designed it so the in-memory and on-disk data layouts are identical, so loading is a straight copy into memory with no clever logic required (each Tree in the model provides the memoryview interface, with full read/write support!).

Explore the test files to see use cases. Typical usage is to create a Forest() object, then call the configure method. The configure method is probably the most fiddly bit - it defines the inputs and outputs (you can have multiple outputs, though that's generally not useful) using three strings of codes (one character per code), where the codes are in the documentation/provided by the info.py script. The first string specifies the summary type, which is what is being learnt for each output. For instance 'C' means one categorical output, which would typically be used for a classification forest. The second string specifies what it is greedily optimising when learning, one code per output (first and second string must be same length). 'C' for this string would mean one output, categorical, for which the system has an entropy based objective. This separation is so you can have different objectives with the same output type, though only entropy ones are provided at this time. The final string tells the system how it can use the inputs to the random forest - effectively the kinds of test to generate for each input feature when deciding which branch to go down. 'OSS' would be a length three feature vector where the first is categorical, for which it uses one vs all tests, and the second and third are both real, for which it generates split tests based on a comparison. The Forest object also has a load of variables, which control things like maximum tree depth.

After the Forest is setup the train(x, y, # of trees to add) method will add trees. Be aware that tree objects can be moved from one Forest object to another and serialised - this is so learning using multiple cores is trivial (You can serialise the Forest object as well, so you only have to configure it once!). This method can be called repeatedly, to keep adding trees. Data set does not have to be the same each time - usually that would be used for incremental learning, where you train new trees with the extra data, then cull trees with poor OOB performance. The train method returns the OOB. Finally, once a Forest is trained the predict(x) method will return the predictions for the given data matrix. Note that the entire system support passing in tuples/lists of data matrices (each of which is a 2D numpy arrays), so you can have both discrete (int) and real (float) features at the same time. You can also weight the exemplars. The Forest and Tree object additionally have loads of extra methods for diagnostics, configuration and i/o - see documentation for details.

I/O is one of the strong points of the system - see the save_forest and load_forest functions in frf.py for examples of how it works.

If you are reading readme.txt then you can generate documentation by running make_doc.py


Contains the following key files:

frf.py - The file a user imports - provides the Forest class, the Tree class (can be ignored) and two methods for file i/o.

info.py - Dynamically generated information about the summary, information and learner types available to the system.

test_*.py - Some test scripts.

readme.txt - This file, which is included in the html documentation.
make_doc.py - Builds the html documentation.

