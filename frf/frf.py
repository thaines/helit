# Copyright 2014 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



from utils.make import make_mod
import os.path

make_mod('frf_c', os.path.dirname(__file__), ['philox.h', 'philox.c', 'data_matrix.h', 'data_matrix.c', 'summary.h', 'summary.c', 'information.h', 'information.c', 'learner.h', 'learner.c', 'index_set.h', 'index_set.c', 'node.h', 'node.c', 'frf_c.h', 'frf_c.c'])



from frf_c import *
