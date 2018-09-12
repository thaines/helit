# Copyright 2016 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import os.path

from utils.make import make_mod



# Compile the code if need be...
make_mod('line_graph_c', os.path.dirname(__file__), ['line_graph_c.h', 'line_graph_c.c'], numpy=True)



# Import the compiled module into this space, so we can pretend they are one and the same, just with automatic compilation...
from line_graph_c import *
