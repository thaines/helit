# -*- coding: utf-8 -*-

# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



# Simple file that imports the various implimentations and provides a default implimentation of kmeans, which happens to be the most sophisticated version...

from kmeans0 import KMeans0
from kmeans1 import KMeans1
from kmeans2 import KMeans2
from kmeans3 import KMeans3

KMeansShort = KMeans1
KMeans = KMeans3
