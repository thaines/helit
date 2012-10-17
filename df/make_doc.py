#! /usr/bin/env python

# Copyright 2012 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import df

from utils import doc_gen



# Setup...
doc = doc_gen.DocGen('df', 'Decision Forests', 'Extensive random forests implimentation')
doc.addFile('readme.txt', 'Overview')


# Classes...
doc.addClass(df.DF)
doc.addClass(df.ExemplarSet)
doc.addClass(df.MatrixES)
doc.addClass(df.MatrixGrow)
doc.addClass(df.Goal)
doc.addClass(df.Classification)
doc.addClass(df.DensityGaussian)
doc.addClass(df.Pruner)
doc.addClass(df.PruneCap)
doc.addClass(df.Test)
doc.addClass(df.AxisSplit)
doc.addClass(df.LinearSplit)
doc.addClass(df.DiscreteBucket)
doc.addClass(df.Generator)
doc.addClass(df.MergeGen)
doc.addClass(df.RandomGen)
doc.addClass(df.AxisMedianGen)
doc.addClass(df.LinearMedianGen)
doc.addClass(df.AxisRandomGen)
doc.addClass(df.LinearRandomGen)
doc.addClass(df.DiscreteRandomGen)
doc.addClass(df.AxisClassifyGen)
doc.addClass(df.LinearClassifyGen)
doc.addClass(df.DiscreteClassifyGen)
doc.addClass(df.SVMClassifyGen)
doc.addClass(df.Node)
