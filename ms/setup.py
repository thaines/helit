#! /usr/bin/env python

# Copyright 2013 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

from distutils.core import setup, Extension



depends = ['bessel.h', 'eigen.h', 'kernels.h', 'data_matrix.h', 'spatial.h', 'balls.h', 'mean_shift.h', 'ms_c.h']
code = ['bessel.c', 'eigen.c', 'kernels.c', 'data_matrix.c', 'spatial.c', 'balls.c',  'mean_shift.c', 'ms_c.c']

ext = Extension('ms_c', code, depends=depends)

setup(name='ms',
      version='1.0.0',
      description='Mean Shift',
      author='Tom SF Haines',
      author_email='thaines@gmail.com',
      url='http://code.google.com/p/haines/',
      py_modules=['ms'],
      ext_modules=[ext],
      )
