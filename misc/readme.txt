Misc

A selection of miscellaneous algorithms:

philox.h/.c - The random number generator from the paper "Parallel Random Numbers: As Easy as 1, 2, 3" by J. K. Salmon et al. This algorithm was designed for GPU use, but after using it for my background subtraction paper I started using it on the CPU as well. Its a great approach as it can go from any number to a random value, so you can easily do the normal thing, of requesting numbers in sequence, but also assign random numbers to, e.g. locations in a grid, and evaluate them efficiently without needing a cache. Has a reasonable selection of samplers for various PDFs. I use this to make my code deterministic, and it also means that if I ever write a GPU version of the code I should be able to obtain bit-identical behaviour. Has no Python interface - its for use by other C/C++ modules that may have a Python interface.

tps.py - A straight forward pure-Python thin plate spline implementation.
