# Copyright 2013 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



# Compile the code if need be...
try:
  from utils.make import make_mod
  import os.path

  make_mod('ms_c', os.path.dirname(__file__), ['bessel.h', 'bessel.c', 'eigen.h', 'eigen.c', 'kernels.h', 'kernels.c', 'data_matrix.h', 'data_matrix.c', 'spatial.h', 'spatial.c', 'balls.h', 'balls.c', 'mean_shift.h', 'mean_shift.c', 'ms_c.h', 'ms_c.c'])
except: pass



# Import the compiled module into this space, so we can pretend they are one and the same, just with automatic compilation...
from ms_c import MeanShift
