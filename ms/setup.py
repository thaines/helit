#! /usr/bin/env python

# Copyright 2013 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import numpy

try:
  from setuptools import setup
except:
  from distutils.core import setup

from distutils.core import Extension



depends = ['philox.h', 'bessel.h', 'eigen.h', 'mult.h', 'kernels.h', 'convert.h', 'data_matrix.h', 'spatial.h', 'balls.h', 'mean_shift.h', 'ms_c.h']
code = ['philox.c', 'bessel.c', 'eigen.c', 'mult.c', 'kernels.c', 'convert.c', 'data_matrix.c', 'spatial.c', 'balls.c',  'mean_shift.c', 'ms_c.c']

ext = Extension('ms_c', code, depends=depends)

setup(name='ms',
      version='1.1.0',
      description='Mean Shift',
      author='Tom SF Haines',
      author_email='thaines@gmail.com',
      url='http://code.google.com/p/haines/',
      py_modules=['ms'],
      ext_modules=[ext],
      include_dirs=[numpy.get_include()]
      )
