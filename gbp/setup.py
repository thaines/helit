#! /usr/bin/env python

# Copyright 2014 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

from distutils.core import setup, Extension



depends = ['gbp_c.h']
code = ['gbp_c.c']

ext = Extension('gbp_c', code, depends=depends)

setup(name='gbp',
      version='1.0.0',
      description='Gaussian belief propagation',
      author='Tom SF Haines',
      author_email='thaines@gmail.com',
      url='https://github.com/thaines/helit',
      py_modules=['gbp'],
      ext_modules=[ext],
     )
