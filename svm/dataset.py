# -*- coding: utf-8 -*-

# Copyright 2010 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import numpy
import operator



class Dataset:
  """Contains a dataset - lots of pairs of feature vectors and labels. For conveniance labels can be arbitrary python objects, or at least python objects that work for indexing a dictionary."""
  def __init__(self):
    # labels are internally stored as consecutive integers - this does the conversion...
    self.labelToNum = dict()
    self.numToLabel = []

    # Store of data blocks - each block is a data matrix and label list pair (A lot of blocks could be of length one of course.)...
    self.blocks = []


  def add(self,featVect,label):
    """Adds a single feature vector and label."""
    if label in self.labelToNum:
      l = self.labelToNum[label]
    else:
      l = len(self.numToLabel)
      self.numToLabel.append(label)
      self.labelToNum[label] = l
    
    self.blocks.append((featVect.reshape((1,featVect.shape[0])).astype(numpy.double),[l]))

  def addMatrix(self,dataMatrix,labels):
    """This adds a data matrix alongside a list of labels for it. The number of rows in the matrix should match the number of labels in the list."""
    assert(dataMatrix.shape[0]==len(labels))

    # Add any labels not yet seen...
    for l in labels:
      if l not in self.labelToNum.keys():
        num = len(self.numToLabel)
        self.numToLabel.append(l)
        self.labelToNum[l] = num

    # Convert the given labels list to a list of numerical labels...
    ls = map(lambda l:self.labelToNum[l],labels)

    # Store...
    self.blocks.append((dataMatrix.astype(numpy.double),ls))


  def getLabels(self):
    """returns a list of all the labels in the data set."""
    return self.numToLabel

  def getCounts(self):
    """Returns a how many features with each label have been seen - as a list which aligns with the output of getLabels."""
    ret = [0]*len(self.numToLabel)
    for block in self.blocks:
      for label in block[1]: ret[label] += 1
    return ret


  def getTrainData(self,lNeg,lPos):
    """Given two labels this returns a pair of a data matrix and a y vector, where lPos features have +1 and lNeg features have -1. Features that do not have one of these two labels will not be included."""
    # Convert the given labels to label numbers...
    if lNeg in self.labelToNum:
      ln = self.labelToNum[lNeg]
    else:
      ln = -1
    if lPos in self.labelToNum:
      lp = self.labelToNum[lPos]
    else:
      lp = -1

    # Go through the blocks and extract the relevant info...
    dataList = []
    yList = []
    for dataMatrix, labels in self.blocks:
      y = filter(lambda l:l==lp or l==ln,labels)
      if len(y)!=0:
        def signRes(l):
          if l==lp: return 1.0
          else: return -1.0
        y = numpy.array(map(signRes,y), dtype=numpy.float_)

        inds = map(operator.itemgetter(0), filter(lambda l:l[1]==lp or l[1]==ln, enumerate(labels)))
        data = dataMatrix[numpy.array(inds),:]

        dataList.append(data)
        yList.append(y)

    # Glue it all together into big blocks, and return 'em...
    return (numpy.vstack(dataList),numpy.concatenate(yList))
