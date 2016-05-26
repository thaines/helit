ply 2

This specifies a more flexible version of the ply file format, extended primarily to store data other than 3D models. It was designed because I wanted a file format that was human readable (unlike hdf5) and suitable for data sets too large for a json file. Being that the ply file format already has these properties it made sense to extend it. It should be noted that my choice of 'ply2' is entirely independent of the original authors of the ply format - I hope this does not annoy them. As the original ply format suffers from a certain amount of variability in implementation, often resulting in incompatibilities between software, I have written this to reduce the variability; it is entirely incompatible with the original format as a result.

This library reads and writes ply 2 files, as represented internally by a nest of Python dictionaries, where the data itself is stored as numpy arrays. Given the dictionary representing the ply 2 file has the variable name 'data', then the following entries are used:

data['format'] = 'ascii', 'binary_little_endian' or 'binary_big_endian' to indicate how the file is to be stored; if omitted defaults to ascii.
data['type'] = A list of types (arbitrary strings), indicating what kind of data the file represents.
data['meta'] - A dictionary indexed by the key of each meta item, going to the meta items, so that data['meta']['author'] = 'Cthulhu' indicates that the header includes 'meta string:nat32 author 7 Cthulhu\n'. Encoding is automatically inferred from the python type.
data['comment'] - A dictionary indexed by natural numbers, to get comment 0, comment 1 etc. as strings.
data['compress'] = None, '', 'gzip', 'bzip2'. If omitted or the first two options that means no compression.

data['element'] - A dictionary indexed by the name of each element type.
data['element'][element name] - A dictionary indexed by property.
data['element'][element name][property name] - A numpy array with the shape of the element in question, containing all of the data for this property type.

See specification.txt for what is actually written to disk.


Contains the following files:

ply2.py - Contains the read and write functions.
test.py - Contains lots of unit tests to make sure it works. Includes many examples of using the library.

readme.txt - This file, which is included in the documentation.
make_doc.py - Builds the documentation.

