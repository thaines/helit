#! /usr/bin/env python

# Copyright 2016 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import numpy

try:
  from setuptools import setup
except:
  from distutils.core import setup

from distutils.core import Extension



depends = ['philox.h', 'data_matrix.h', 'summary.h', 'information.h', 'learner.h', 'index_set.h', 'tree.h', 'frf_c.h']
code = ['philox.c', 'data_matrix.c', 'summary.c', 'information.c', 'learner.c', 'index_set.c', 'tree.c', 'frf_c.c']

ext = Extension('frf_c', code, depends=depends)

setup(name='frf',
      version='1.0.0',
      description='Fast Random Forest',
      author='Tom SF Haines',
      author_email='thaines@gmail.com',
      url='https://github.com/thaines/helit',
      py_modules=['frf'],
      ext_modules=[ext],
      include_dirs=[numpy.get_include()]
      )
