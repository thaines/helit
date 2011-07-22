# -*- coding: utf-8 -*-

# Copyright 2011 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



import math

import numpy



def modelSelectBIC(feats, model, maxCount=-1):
  """Provides model selection, i.e. selection of the number of clusters, using the bayesian information criterion (BIC). You provide it with the features, and a model to train with (It just reuses it.) and the maximum size to consider (It starts at 2.). When it returns the given model will be left trained with the optimal result. If maxSize is less than 2, the default, it goes up to twice the logarithm of the feature vector count, rounded up and inclusive. if this is 2 or less you will get back the model trained with just two clusters."""

  if isinstance(feats,numpy.ndarray): size = feats.shape[0]
  else: size = len(feats)
  
  if maxCount<2:
    maxCount = int(math.ceil(2.0*math.log(size)))
    if maxCount<2: maxCount = 2

  best = None
  bestScore = None
  for c in xrange(2,maxCount+1):
    model.train(feats,c)
    score = 2.0*model.getNLL(feats) + model.parameters()*math.log(size)
    if bestScore==None or bestScore>score:
      bestScore = score
      best = model.getData()

  model.setData(best)
