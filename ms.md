# Mean Shift #

## Overview ##
**Mean Shift**

Primarily provides a mean shift implementation, but also includes kernel density estimation and subspace constrained mean shift using the same object, such that they are all using the same underlying density estimate. Includes multiple spatial indexing schemes and kernel types, including one for directional data. Clustering is supported, with a choice of cluster intersection tests, as well as the ability to interpret exemplar indexing dimensions of the data matrix as extra features, so it can handle the traditional image segmentation scenario.

If you are reading readme.txt then you can generate documentation by running make\_doc.py
Note that this module includes a setup.py that allows you to package/install it (The dependency on utils is only for the tests and automatic compilation if you have not installed it - it is not required.)
It is strongly recommended that you look through the various test**files to see examples of how to use the system.**


Contains the following key files:

`ms.py` - The file a user imports - provides a single class - MeanShift.


`info.py` - File that iterates and prints out information on all the modular components of the system.


`test_*.py` - Many test scripts.


`readme.txt` - This file, which is included in the html documentation.

`make_doc.py` - Builds the html documentation.

`setup.py` - Allows you to create a package/build/install this module.


---


# Classes #

## MeanShift(object) ##
> An object implimenting mean shift; also includes kernel density estimation and subspace constrained mean shift using the same object, such that they are all using the same underlying density estimate. Includes multiple spatial indexing schemes and kernel types, including one for directional data. Clustering is supported, with a choice of cluster intersection tests, as well as the ability to interpret exemplar indexing dimensions of the data matrix as extra features, so it can handle the traditional image segmentation scenario.

**`assign_cluster(?)`**
> After the cluster method has been called this can be called with a single feature vector. It will then return the index of the cluster to which it has been assigned, noting that this will map to the mode array returned by the cluster method. In the event it does not map to a pre-existing cluster it will return a negative integer - this usually means it is so far from the provided data that the kernel does not include any samples.

**`assign_clusters(?)`**
> After the cluster method has been called this can be called with a data matrix. It will then return the indices of the clusters to which each feature vector has been assigned, as a 1D numpy array, noting that this will map to the mode array returned by the cluster method. In the event any entry does not map to a pre-existing cluster it will return a negative integer for it - this usually means it is so far from the provided data that the kernel does not include any samples.

**`balls(?)`**
> Returns a list of ball indexing techneques - this is the structure used when clustering to represent the hyper-sphere around the mode that defines a cluster in terms of merging distance.

**`cluster(?)`**
> Clusters the exemplars provided by the data matrix - returns a two tuple (data matrix of all the modes in the dataset, indexed [mode, feature], A matrix of integers, indicating which mode each one has been assigned to by indexing the mode array. Indexing of this array is identical to the provided data matrix, with any feature dimensions removed.). The clustering is replaced each time this is called - do not expect cluster indices to remain consistant after calling this.

**`exemplars(?)`**
> Returns how many exemplars are in the hallucinated data matrix.

**`features(?)`**
> Returns how many features are in the hallucinated data matrix.

**`get_balls(?)`**
> Returns the current ball indexing structure, as a string.

**`get_kernel(?)`**
> Returns the string that identifies the current kernel.

**`get_spatial(?)`**
> Returns the string that identifies the current spatial indexing structure.

**`info(?)`**
> A static method that is given the name of a kernel, spatial or ball. It then returns a human readable description of that entity.

**`kernels(?)`**
> A static method that returns a list of kernel types, as strings.

**`manifold(?)`**
> Given a feature vector and the dimensionality of the manifold projects the feature vector onto the manfold using subspace constrained mean shift. Returns an array with the same shape as the input.

**`manifolds(?)`**
> Given a data matrix [exemplar, feature] and the dimensionality of the manifold projects the feature vectors onto the manfold using subspace constrained mean shift. Returns a data matrix with the same shape as the input.

**`manifolds_data(?)`**
> Given the dimensionality of the manifold projects the feature vectors that are defining the density estimate onto the manfold using subspace constrained mean shift. The return value will be indexed in the same way as the provided data matrix, but without the feature dimensions, with an extra dimension at the end to index features.

**`mode(?)`**
> Given a feature vector returns its mode as calculated using mean shift - essentially the maxima in the kernel density estimate to which you converge by climbing the gradient.

**`modes(?)`**
> Given a data matrix [exemplar, feature] returns a matrix of the same size, where each feature has been replaced by its mode, as calculated using mean shift.

**`modes_data(?)`**
> Runs mean shift on the contained data set, returning a feature vector for each data point. The return value will be indexed in the same way as the provided data matrix, but without the feature dimensions, with an extra dimension at the end to index features. Note that the resulting output will contain a lot of effective duplication, making this a very inefficient method - your better off using the cluster method.

**`prob(?)`**
> Given a feature vector returns its probability, as calculated by the kernel density estimate that is defined by the data and kernel. Be warned that the return value can be zero.

**`probs(?)`**
> Given a data matrix returns an array (1D) containing the probability of each feature, as calculated by the kernel density estimate that is defined by the data and kernel. Be warned that the return value can be zero.

**`set_balls(?)`**
> Sets the current ball indexing structure, as identified by a string.

**`set_data(?)`**
> Sets the data matrix, which defines the probability distribution via a kernel density estimate that everything is using. First parameter is a numpy matrix (Any normal numerical type), the second a string with its length matching the number of dimensions of the matrix. The characters in the string define the meaning of each dimension: 'd' (data) - changing the index into this dimension changes which exemplar you are indexing; 'f' (feature) - changing the index into this dimension changes which feature you are indexing; 'b' (both) - same as d, except it also contributes an item to the feature vector, which is essentially the position in that dimension (used on the dimensions of an image for instance, to include pixel position in the feature vector). The system unwraps all data indices and all feature indices in row major order to hallucinate a standard data matrix, with all 'both' features at the start of the feature vector. Note that calling this resets scale. A third optional parameter sets an index into the original feature vector that is to be the weight of the feature vector - this effectivly reduces the length of the feature vector, as used by all other methods, by one.

**`set_kernel(?)`**
> Sets the current kernel, as identified by a string. An optional second parameter exists, for the alpha parameter, which is passed through to the kernel. Most kernels ignore this parameter - right now only the 'fisher' kernel uses it, where it is the concentration parameter of the von-Mises Fisher distribution that is used.

**`set_scale(?)`**
> Given two parameters. First is an array indexed by feature to get a multiplier that is applied before the kernel (Which is always of radius 1, or some approximation of.) is considered - effectivly an inverse bandwidth in kernel density estimation terms. Second is an optional scale for the weight assigned to each feature vector via the set\_data method (In the event that no weight is assigned this parameter is the weight of each feature vector, as the default is 1).

**`set_spatial(?)`**
> Sets the current spatial indexing structure, as identified by a string.

**`spatials(?)`**
> A static method that returns a list of spatial indexing structures you can use, as strings.

**`weight(?)`**
> Returns the total weight of the included data, taking into account a weight channel if provided.

**`alpha`** = Arbitrary parameter that is passed through to the kernel - most kernels ignore it. Only current use is for the 'fisher' kernel, where it is the concentration of the von-Mises Fisher distribution.

**`epsilon`** = For convergance detection - when the step size is smaller than this it stops.

**`ident_dist`** = If two exemplars are found at any point to have a distance less than this from each other whilst clustering it is assumed they will go to the same destination, saving computation.

**`iter_cap`** = Maximum number of iterations to do before stopping, a hard limit on computation.

**`merge_check_step`** = When clustering this controls how many mean shift iterations it does between checking for convergance - simply a tradeoff between wasting time doing mean shift when it has already converged and doing proximity checks for convergance. Should only affects runtime.

**`merge_range`** = Controls how close two mean shift locations have to be to be merged in the clustering method.

**`quality`** = Value between 0 and 1, inclusive - for kernel types that have an infinite domain this controls how much of that domain to use for the calculations - 0 for lowest quality, 1 for the highest quality. (Ignored by kernel types that have a finite kernel.)