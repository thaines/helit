# Utilities/Miscellaneous #

## Overview ##
**Utils**

A selection of miscellaneous utilities, for doing various boring tasks.


`cvarray.py` - Code for converting between open cv and numpy arrays.

`prog_bar.py` - A simple progress bar object, for command line interface feedback.

`mp_map.py` - A multiprocess map. Note that this file contains unit tests.

`setprocName.py` - Function to change the process name of a python program.

`make.py` - A wrapper around distutils for making modules when they have changed. Its a bit like scipy.weave, except you write a real module and it just compiles it on demand for the current platform.


`start_cpp.py` - Helper to add line numbers to scipy.weave code, to make the error messages helpful.

`numpy_help_cpp.py` - Helper functions for accessing numpy arrays in scipy.weave.

`python_obj_cpp.py` - Helper functions for manipulating python objects in scipy.weave.

`gamma_cpp.py` - Implimentations of the Gamma and related functions in C.

`matrix_cpp.py` - Some matrix operations in C++.


`readme.txt` - This file, which is copied into the html documentation.

`make_doc.py` - code that generates the html documentation.

---


# Variables #

**`numpy_help_cpp.numpy_util_code`**
> Assorted utility functions for accessing numpy arrays within scipy.weave C++ code.

**`python_obj_cpp.python_obj_code`**
> Assorted utility functions for interfacing with python objects from scipy.weave C++ code.

**`matrix_cpp.matrix_code`**
> Matrix manipulation routines for use in scipy.weave C++

**`gamma_cpp.gamma_code`**
> Gamma and related functions for use in scipy.weave C++


# Functions #

**`make_mod(name, base, source, openCL = False)`**
> Uses distutils to compile a python module - really just a set of hacks to allow this to be done 'on demand', so it only compiles if the module does not exist or is older than the current source, and after compilation the program can continue on its merry way, and immediatly import the just compiled module. Note that on failure erros can be thrown - its your choice to catch them or not. name is the modules name, i.e. what you want to use with the import statement. base is the base directory for the module, which contains the source file - often you would want to set this to 'os.path.dirname(file)', assuming the .py file that imports the module is in the same directory as the code. It is this directory that the module is output to. source is the filename of the source code to compile, or alternativly a list of filenames. openCL indicates if OpenCL is used by the module, in which case it does all the necesary setup - done like this so these setting can be kept centralised, so when they need to be different for a new platform they only have to be changed in one place.

**`cv2array(im)`**
> Converts a cv array to a numpy array.

**`array2cv(a)`**
> Converts a numpy array to a cv array, if possible.

**`repeat(x)`**
> A generator that repeats the input forever - can be used with the mp\_map function to give data to a function that is constant.

**`mp_map(func, *iters, **keywords)`**
> A multiprocess version of the map function. Note that func must limit itself to the data provided - if it accesses anything else (globals, locals to its definition.) it will fail. There is a repeat generator provided in this module to work around such issues. Note that, unlike map, this iterates the length of the shortest of inputs, rather than the longest - whilst this makes it not a perfect substitute it makes passing constant argumenmts easier as they can just repeat for infinity.

**`setProcName(name)`**
> Sets the process name, linux only - useful for those programs where you might want to do a killall, but don't want to slaughter all the other python processes. Note that there are multiple mechanisms, and that the given new name can be shortened by differing amounts in differing cases.

**`start_cpp(hash_str)`**
> This method does two things - firstly it adds the correct line numbers to scipy.weave code (Good for debugging) and secondly it can optionaly inserts a hash code of some other code into the code. This latter feature is useful for working around the fact the scipy.weave only recompiles if the hash of the code changes, but ignores the support\_code - passing the support\_code into start\_cpp avoids this problem by putting its hash into the code and forcing a recompile when that code changes. Usage is <code variable> = start\_cpp([variable](support_code.md)) + <3 quotations to start big comment with code in, typically going over many lines.>

**`make_mod(name, base, source, openCL = False)`**
> Uses distutils to compile a python module - really just a set of hacks to allow this to be done 'on demand', so it only compiles if the module does not exist or is older than the current source, and after compilation the program can continue on its merry way, and immediatly import the just compiled module. Note that on failure erros can be thrown - its your choice to catch them or not. name is the modules name, i.e. what you want to use with the import statement. base is the base directory for the module, which contains the source file - often you would want to set this to 'os.path.dirname(file)', assuming the .py file that imports the module is in the same directory as the code. It is this directory that the module is output to. source is the filename of the source code to compile, or alternativly a list of filenames. openCL indicates if OpenCL is used by the module, in which case it does all the necesary setup - done like this so these setting can be kept centralised, so when they need to be different for a new platform they only have to be changed in one place.


# Classes #

## ProgBar() ##
> Simple console progress bar class. Note that object creation and destruction matter, as they indicate when processing starts and when it stops.

**`__init__(self, width = 60, onCallback)`**
> None

**`__del__(self)`**
> None

**`callback(self, nDone, nToDo)`**
> Hand this into the callback of methods to get a progress bar - it works by users repeatedly calling it to indicate how many units of work they have done (nDone) out of the total number of units required (nToDo).

## DocGen() ##
> A helper class that is used to generate documentation for the system. Outputs multiple formats simultaneously, specifically html for local reading with a webbrowser and the markup used by the wiki system on Google code.

**`__init__(self, name, title, summary)`**
> name is the module name - primarilly used for the file names. title is the title used as applicable - if not provide it just uses the name. summary is an optional line to go below the title.

**`__del__(self)`**
> None

**`addClass(self, cls)`**
> Adds a class to the documentation. You provide the actual class object.

**`addFile(self, fn, title, fls = True)`**
> Given a filename and section title adds the contents of said file to the output. Various flags influence how this works.

**`addFunction(self, func)`**
> Adds a function to the documentation. You provide the actual function instance.

**`addVariable(self, var, desc)`**
> Adds a variable to the documentation. Given the nature of this you provide it as a pair of strings - one referencing the variable, the other some kind of description of its use etc..