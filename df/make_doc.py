#! /usr/bin/env python

# Copyright 2012 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import df
import pydoc

doc = pydoc.HTMLDoc()


# Open the document...
out = open('df.html','w')
out.write('<html>\n')
out.write('<head>\n')
out.write('<title>Decision Forests</title>\n')
out.write('</head>\n')
out.write('<body>\n')


# Openning blob...
readme = open('readme.txt','r').read()
readme = readme.replace('\n','<br/>')
out.write(doc.bigsection('Overview','#ffffff','#7799ee',readme))



# Class...
classes = ''
classes += doc.docclass(df.DF)
classes += doc.docclass(df.ExemplarSet)
classes += doc.docclass(df.MatrixES)
classes += doc.docclass(df.MatrixGrow)
classes += doc.docclass(df.Goal)
classes += doc.docclass(df.Classification)
classes += doc.docclass(df.DensityGaussian)
classes += doc.docclass(df.Pruner)
classes += doc.docclass(df.PruneCap)
classes += doc.docclass(df.Test)
classes += doc.docclass(df.AxisSplit)
classes += doc.docclass(df.LinearSplit)
classes += doc.docclass(df.DiscreteBucket)
classes += doc.docclass(df.Generator)
classes += doc.docclass(df.MergeGen)
classes += doc.docclass(df.RandomGen)
classes += doc.docclass(df.AxisMedianGen)
classes += doc.docclass(df.LinearMedianGen)
classes += doc.docclass(df.AxisRandomGen)
classes += doc.docclass(df.LinearRandomGen)
classes += doc.docclass(df.DiscreteRandomGen)
classes += doc.docclass(df.AxisClassifyGen)
classes += doc.docclass(df.LinearClassifyGen)
classes += doc.docclass(df.DiscreteClassifyGen)
classes += doc.docclass(df.SVMClassifyGen)
classes += doc.docclass(df.Node)
classes = classes.replace('&nbsp;',' ')
out.write(doc.bigsection('Classes','#ffffff','#ee77aa',classes))


# Close the document...
out.write('</body>\n')
out.write('</html>\n')
out.close()
